require 'cunn'
require 'image'
require 'nms'
require 'Anchors'

local Detector = torch.class('Detector')

function Detector:__init(model)
  local cfg = model.cfg
  self.model = model
  self.anchors = Anchors.new(model.pnet, model.cfg.scales)
  self.localizer = Localizer.new(model.pnet.outnode.children[#cfg.scales+1]) -- fixed
  self.lsm = nn.LogSoftMax():cuda()
  self.amp = nn.SpatialAdaptiveMaxPooling(cfg.roi_pooling.kw, cfg.roi_pooling.kh):cuda()
end

function Detector:detect(input, draw_img)
  local cfg = self.model.cfg
  local pnet = self.model.pnet
  local cnet = self.model.cnet
  local spanet = self.model.spanet
  local spa_post_process_net = self.model.spa_post_process_net

  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local bgclass = cfg.class_count + 1   -- background class
  local amp = self.amp
  local lsm = self.lsm
  local cnet_input_planes = self.model.layers[#self.model.layers].filters
  
  local input_size = input:size()
  local input_rect = Rect.new(0, 0, input_size[3], input_size[2])
  
  -- pass image through network
  pnet:evaluate()
  input = input:cuda()
  local outputs = pnet:forward(input)
   
   -- analyse network output for non-background classification
  local matches = {}
  
  local aspect_ratios = 3
  for i=1,#cfg.scales do
    local layer = outputs[i]
    local layer_size = layer:size()
    for y=1,layer_size[2] do
      for x=1,layer_size[3] do
        local c = layer[{{}, y, x}]
        for a=1,aspect_ratios do

          local ofs = (a-1) * 6
          local cls_out = c[{{ofs + 1, ofs + 2}}] 
          local reg_out = c[{{ofs + 3, ofs + 6}}]
                    
          -- classification
          local c = lsm:forward(cls_out)
          --print(c[1])
          --print(c[2])
--          if c[1] > c[2] then
          if math.exp(c[1]) > 0.95 then
            -- regression
            local a = self.anchors:get(i,a,y,x)
            local r = Anchors.anchorToInput(a, reg_out)
            if r:overlaps(input_rect) then
              table.insert(matches, { p=c[1], a=a, r=r, l=i })
            end
          end
          
        end
      end
    end      
  end
  
  local winners = {}
  
  if #matches > 0 then
    -- print(matches)  -- FIXME!
    
    -- NON-MAXIMUM SUPPRESSION
    local bb = torch.Tensor(#matches, 4)
    local score = torch.Tensor(#matches, 1)
    for i=1,#matches do
      bb[i] = matches[i].r:totensor()
      score[i] = matches[i].p
    end
    
    local iou_threshold = 0.25
    local pick = nms(bb, iou_threshold, score)
    --local pick = nms(bb, iou_threshold, 'area')
    local candidates = {}
    pick:apply(function (x) table.insert(candidates, matches[x]) end )
    
--    print(string.format('candidates: %d', #candidates))
--    local green = torch.Tensor({0,1,0})
--    for i,m in ipairs(candidates) do
--      draw_rectangle(draw_img, m.r, green)
--    end
--    image.saveJPG(string.format('debug/eval_vis/output%d.jpg', glb_cnt), draw_img)

    -- REGION CLASSIFICATION 
    cnet:evaluate()
    if cfg.use_stn then
      spanet:evaluate()
      spa_post_process_net:evaluate()
    end
    
    local bbox_out, cls_out
    local yclass = {}
    local stn_cnt = 0
    local stn_candidates = {}
    
    if cfg.use_stn then
      -- create a temporary cnet input batch
      local po_stn_proposals = torch.CudaTensor(#candidates, cfg.roi_pooling.kh * cfg.roi_pooling.kw * cnet_input_planes)
      
      for i, x in ipairs(candidates) do
        local r = x.r:snapToInt()
        if r.minX < 1 or r.minY < 1 or r.maxY > input:size(2) or r.maxX > input:size(3) or
          r.maxY - r.minY < cfg.roi_pooling.kh or r.maxX - r.minX < cfg.roi_pooling.kw then
          goto check_fail
        end
        local patch = input[{{}, {r.minY, math.max(r.maxY-1, r.minY)}, {r.minX, math.max(r.maxX-1, r.minX)}}]                
        local spa_out = spanet:forward(patch) -- stn needs input of dim (N, C, H, W) or (C, H, W)  
        po_stn_proposals[i] = spa_post_process_net:forward(spa_out):view(cfg.roi_pooling.kh * cfg.roi_pooling.kw * cnet_input_planes)
        
        stn_candidates[#stn_candidates+1] = x
        stn_cnt = stn_cnt + 1
        ::check_fail::
      end
      
      if stn_cnt > 0 then
        po_stn_proposals = po_stn_proposals:narrow(1, 1, stn_cnt)
        --print(po_stn_proposals)
        local coutputs_stn_proposals = cnet:forward(po_stn_proposals)
        bbox_out = coutputs_stn_proposals[1] -- regression of spatially transformed proposals
        cls_out = coutputs_stn_proposals[2]-- classification of spatially transformed proposals
        
        for i,x in ipairs(stn_candidates) do
          x.r2 = Anchors.anchorToInput(x.r, bbox_out[i])
          
          local cprob = cls_out[i]
          local p,c = torch.sort(cprob, 1, true) -- get probabilities and class indicies
          
          x.class = c[1]
          x.confidence = p[1]
    --      print(x.class)
          if x.class ~= bgclass and math.exp(x.confidence) > 0.2 then
            if not yclass[x.class] then
              yclass[x.class] = {}
            end
            table.insert(yclass[x.class], x)
          end
        end
      end
    end
 
    
    -- create cnet input batch
    local cinput = torch.CudaTensor(#candidates, cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
    for i,v in ipairs(candidates) do
      -- pass through adaptive max pooling operation
      local pi, idx = extract_roi_pooling_input(v.r, self.localizer, outputs[#cfg.scales+1]) -- fixed
      cinput[i] = amp:forward(pi):view(cfg.roi_pooling.kw * cfg.roi_pooling.kh * cnet_input_planes)
    end
    
    -- send extracted roi-data through classification network
    local coutputs = cnet:forward(cinput)
    bbox_out = coutputs[1]
    cls_out = coutputs[2]
      
--    print(#candidates)
--    print(bbox_out)
--    print(cls_out)
    
    for i,x in ipairs(candidates) do
      x.r2 = Anchors.anchorToInput(x.r, bbox_out[i])
      
      local cprob = cls_out[i]
      local p,c = torch.sort(cprob, 1, true) -- get probabilities and class indicies
      
      x.class = c[1]
      x.confidence = p[1]
--      print(x.class)
      if x.class ~= bgclass and math.exp(x.confidence) > 0.2 then
        if not yclass[x.class] then
          yclass[x.class] = {}
        end
        
        table.insert(yclass[x.class], x)
      end
    end
    
    -- run per class NMS
    for i,c in pairs(yclass) do
      -- fill rect tensor
      bb = torch.Tensor(#c, 5)
      for j,r in ipairs(c) do
        bb[{j, {1,4}}] = r.r2:totensor()
        bb[{j, 5}] = r.confidence
      end
      
      pick = nms(bb, 0.1, bb[{{}, 5}])
      pick:apply(function (x) table.insert(winners, c[x]) end ) 
     
    end
    
    --print(winners)
  end
  
  return winners
end
