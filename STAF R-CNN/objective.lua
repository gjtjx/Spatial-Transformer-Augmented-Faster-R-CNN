require 'cunn'
require 'BatchIterator'
require 'Localizer'
require 'image'
color = require 'trepl.colorize'


local glb_w1, glb_w2


function extract_roi_pooling_input(input_rect, localizer, feature_layer_output)
  local r = localizer:inputToFeatureRect(input_rect)
  -- the use of math.min ensures correct handling of empty rects, 
  -- +1 offset for top, left only is conversion from half-open 0-based interval
  local s = feature_layer_output:size()
  r = r:clip(Rect.new(0, 0, s[3], s[2]))
  local idx = { {}, { math.min(r.minY + 1, r.maxY), r.maxY }, { math.min(r.minX + 1, r.maxX), r.maxX } }
  return feature_layer_output[idx], idx
end

function create_objective(opt, model, weights, gradient, batch_iterator, stats)
  local cfg = model.cfg
  local pnet = model.pnet
  local cnet = model.cnet
  local spanet = model.spanet
  local spa_cnet = model.cnet:clone('weight','bias') -- make a copy that shares the weights and biases
  local bgclass = cfg.class_count + 1   -- background class
  local anchors = batch_iterator.anchors    
  local localizer = Localizer.new(pnet.outnode.children[#cfg.scales+1]) -- fixed 
  local softmax = nn.CrossEntropyCriterion():cuda()
  local cnll = nn.ClassNLLCriterion():cuda()
  local smoothL1 = nn.SmoothL1Criterion():cuda()
  local spa_cnll = cnll:clone('weight') -- make a copy that shares the weights
  local spa_smoothL1 = nn.SmoothL1Criterion():cuda()
  local spa_post_process_net = model.spa_post_process_net
  
  smoothL1.sizeAverage = false
  spa_smoothL1.sizeAverage = false
  
  local kh, kw = cfg.roi_pooling.kh, cfg.roi_pooling.kw
  local cnet_input_planes = model.layers[#model.layers].filters
  local amp = nn.SpatialAdaptiveMaxPooling(kw, kh):cuda()

  local function cleanAnchors(examples, outputs)
    local i = 1
    while i <= #examples do
      local anchor = examples[i][1]
      local fmSize = outputs[anchor.layer]:size()
      if anchor.index[2] > fmSize[2] or anchor.index[3] > fmSize[3] then
        table.remove(examples, i)   -- accessing would cause ouf of range exception
      else
        i = i + 1
      end
    end
  end

  local function lossAndGradient(w)
    if w ~= weights then
      weights:copy(w)
    end
    gradient:zero()

    -- statistics for proposal stage      
    local cls_loss, reg_loss = 0, 0
    local cls_count, reg_count = 0, 0
    local delta_outputs = {}

    -- statistics for fine-tuning and classification stage
    local creg_loss, creg_count = 0, 0
    local ccls_loss, ccls_count = 0, 0

    -- enable dropouts 
    pnet:training()
    cnet:training()
    if cfg.use_stn then
      spa_post_process_net:training()
    end
    
    local batch, num_img = batch_iterator:nextTraining()
    glb_img_scanned = glb_img_scanned + num_img

    for i,x in ipairs(batch) do
      local img = x.img:cuda()    -- convert batch to cuda if we are running on the gpu
      local p = x.positive        -- get positive and negative anchors examples
      local n = x.negative

      -- run forward convolution
      local outputs = pnet:forward(img)
--      print(outputs)
      
      -- ensure all example anchors lie withing existing feature planes 
      cleanAnchors(p, outputs)
      cleanAnchors(n, outputs)

      -- clear delta values for each new image
      for i,out in ipairs(outputs) do
        if not delta_outputs[i] then
          delta_outputs[i] = torch.FloatTensor():cuda()
        end
        delta_outputs[i]:resizeAs(out)
        delta_outputs[i]:zero()
      end

      local roi_pool_state = {}
      local input_size = img:size()
      local cnetgrad
      local pos_num = 0
      local neg_num = 0
      local stn_proposal_num = 0
      local stn_out = {}
      local stn_patch = {}
      
      -- process positive set
      for i,x in ipairs(p) do
        local anchor = x[1]
        local roi = x[2]
        local l = anchor.layer

        local out = outputs[l]
        local delta_out = delta_outputs[l]

        local idx = anchor.index
        local v = out[idx]
        local d = delta_out[idx]

        -- box regression
        local reg_out = v[{{3, 6}}]
        local reg_target = Anchors.inputToAnchor(anchor, roi.rect):cuda()  -- regression target
        local reg_proposal = Anchors.anchorToInput(anchor, reg_out)
        local po_stn_proposal

        -- utilize spatial transformer network at fine-tune stage
        if cfg.use_stn then
          -- positive example
          local r = reg_proposal:snapToInt()
--                print(x.reg_proposal)
--                print(r)
--                print(img:size()) -- img is of size (C, H, W)
          if r.minX < 1 or r.minY < 1 or r.maxY > img:size(2) or r.maxX > img:size(3) then
            goto check_fail
          end

          if Rect.IoU(r, roi.rect) < 0.4 then -- threshold needs to be adjusted
            goto check_fail
          end

          local patch = img[{{}, {r.minY, math.max(r.maxY-1, r.minY)}, {r.minX, math.max(r.maxX-1, r.minX)}}]                
          local spa_out = spanet:forward(patch) -- stn needs input of dim (N, C, H, W) or (C, H, W)
          
          if cfg.vis_stn then

            if glb_b_cnt % opt.stn_vis_every == 0 then
              draw_img = torch.FloatTensor(2, 3, 120, 120)
              draw_img[1] = image.scale(torch.FloatTensor(patch:size()):copy(patch), 120, 120)
              draw_img[2] = image.scale(torch.FloatTensor(spa_out:size()):copy(spa_out), 120, 120)
  --            glb_w1 = image.display({image=draw_img, nrow=2, legend='Comparison', win=glb_w1})
  --              glb_w2 = image.display({image=img, nrow=1, legend='original image', win=glb_w2})
              image.save(string.format(opt.save..'stn/'..'batch%d_patch%d.jpg', glb_b_cnt, stn_proposal_num+1), draw_img[1])
              image.save(string.format(opt.save..'stn/'..'batch%d_patch%d_stn.jpg', glb_b_cnt, stn_proposal_num+1), draw_img[2])
  --            glb_w1 = image.display({image=image.scale(float_img, 58, 58), nrow=1, legend='Original Patch', win=glb_w1})
  --            glb_w2 = image.display({image=spa_out, nrow=1, legend='Spatial Transformed Patch', win=glb_w2})              
            end
          end
          
----          -- get feature points of proposal in output conv map of the last layer in RPN
--          local pi_stn_proposal, _ = extract_roi_pooling_input(reg_proposal, localizer, outputs[#cfg.scales+1])
--          local po_stn = amp:forward(pi_stn_proposal):view(kh * kw * cnet_input_planes)          
--          print(patch:size()) -- 3X58X58
--          print(spa_out:size()) -- 384X8X8
--          print(pi_stn_proposal:size()) -- 384X12X10
--          print(po_stn:size()) --13824

          po_stn_proposal = spa_post_process_net:forward(spa_out):view(kh * kw * cnet_input_planes)
--          print(po_stn_proposal:size())
          stn_patch[#stn_patch+1] = patch -- store patches for backward propogation  
          stn_out[#stn_out+1] = spa_out -- store proposals for backward propogation  
          stn_proposal_num = stn_proposal_num + 1

          ::check_fail::
        end

        -- classification loss
        cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 1)
        local dc = softmax:backward(v[{{1, 2}}], 1)
        d[{{1,2}}]:add(dc)

        -- regression loss
        reg_loss = reg_loss + smoothL1:forward(reg_out, reg_target) * 10
        local dr = smoothL1:backward(reg_out, reg_target) * 10
        d[{{3,6}}]:add(dr)

        -- pass through adaptive max pooling operation
        local pi, idx = extract_roi_pooling_input(roi.rect, localizer, outputs[#cfg.scales+1]) -- fixed 

--        print(pi) -- 18x30X40
--        print(amp:forward(pi)) -- 18x6x6
--        print(kh * kw * cnet_input_planes) -- 13824
        local po = amp:forward(pi):view(kh * kw * cnet_input_planes)

        local clone_po_stn_proposal
        if po_stn_proposal ~= nil then
          clone_po_stn_proposal = po_stn_proposal:clone()
        end

        table.insert(roi_pool_state, { input = pi, input_idx = idx, anchor = anchor, reg_proposal = reg_proposal, roi = roi, output = po:clone(), output_stn_proposal = clone_po_stn_proposal, indices = amp.indices:clone() })

        pos_num = pos_num + 1
      end

      -- process negative
      for i,x in ipairs(n) do
        local anchor = x[1]
        local l = anchor.layer
        local out = outputs[l]
        local delta_out = delta_outputs[l]
        local idx = anchor.index
        local v = out[idx]
        local d = delta_out[idx]

        cls_loss = cls_loss + softmax:forward(v[{{1, 2}}], 2)
        local dc = softmax:backward(v[{{1, 2}}], 2)
        d[{{1,2}}]:add(dc)

        -- pass through adaptive max pooling operation
        local pi, idx = extract_roi_pooling_input(anchor, localizer, outputs[#cfg.scales+1])

        local po = amp:forward(pi):view(kh * kw * cnet_input_planes)
        table.insert(roi_pool_state, { input = pi, input_idx = idx, output = po:clone(), indices = amp.indices:clone() })

        neg_num = neg_num + 1
      end

      -- fine-tuning STAGE
      -- pass extracted roi-data through classification network

      -- create cnet input batch
      if #roi_pool_state > 0 then
        local cinput = torch.CudaTensor(#roi_pool_state, kh * kw * cnet_input_planes)
        local cctarget = torch.CudaTensor(#roi_pool_state)
        local crtarget = torch.CudaTensor(#roi_pool_state, 4):zero()
        local cinput_stn_proposal = torch.CudaTensor(stn_proposal_num, kh * kw * cnet_input_planes)
        local cctarget_stn_proposal = torch.CudaTensor(stn_proposal_num)
        local crtarget_stn_proposal = torch.CudaTensor(stn_proposal_num, 4):zero()
        local delta_proposals = torch.CudaTensor(stn_proposal_num, 3, 58, 58)
        local stn_cnt = 1

--        print(string.format('%d imaged spatially transformed', stn_proposal_num))

        for i,x in ipairs(roi_pool_state) do
          cinput[i] = x.output
          if cfg.use_stn and x.output_stn_proposal then
            cinput_stn_proposal[stn_cnt] = x.output_stn_proposal
          end

          if x.roi then
            -- positive example
            cctarget[i] = x.roi.class_index
            crtarget[i] = Anchors.inputToAnchor(x.reg_proposal, x.roi.rect)   -- base fine tuning on proposal
            if cfg.use_stn and x.output_stn_proposal then
              cctarget_stn_proposal[stn_cnt] = cctarget[i]
              crtarget_stn_proposal[stn_cnt] = crtarget[i]
              stn_cnt = stn_cnt + 1 -- keep a counter on stn transformed proposals
            end
          else
            -- negative example
            cctarget[i] = bgclass  -- negative example: do not send to STN?
          end
        end
        
        -- process classification batch 
        local coutputs = cnet:forward(cinput)
        local crout = coutputs[1] -- regression
        local ccout = coutputs[2] -- log softmax classification
--        print(crout:size())

        local coutputs_stn_proposal
        local crout_stn_proposal, ccout_stn_propsoal
        if cfg.use_stn and stn_proposal_num > 0 then
          coutputs_stn_proposal = spa_cnet:forward(cinput_stn_proposal)
          crout_stn_proposal = coutputs_stn_proposal[1] -- regression of spatially transformed proposals
          ccout_stn_propsoal = coutputs_stn_proposal[2]-- classification of spatially transformed proposals
--          print(crout_stn_proposal:size())
        end  
        
        crout[{{#p + 1, #roi_pool_state}, {}}]:zero() -- ignore negative examples
        
        if cfg.use_stn and stn_proposal_num > 0 then
          local frac = math.floor(pos_num/stn_proposal_num)
                    
          -- calculate regression loss using smooth L1
          creg_loss = creg_loss + smoothL1:forward(crout, crtarget) * 10 
                                + spa_smoothL1:forward(crout_stn_proposal, crtarget_stn_proposal) * frac
--          -- debug
--          print('Reg Proposal Regression Loss:')
--          print(smoothL1:forward(crout, crtarget) * 10 )
--          print('Spatially Transformed Proposal Regression Loss:')
--          print(spa_smoothL1:forward(crout_stn_proposal, crtarget_stn_proposal) * frac)

          -- dscore of regression coordinates
          local crdelta = smoothL1:backward(crout, crtarget) * 10
          local cr_stn_proposal_delta = spa_smoothL1:backward(crout_stn_proposal, crtarget_stn_proposal) * frac
          
          -- calculate classification loss using cross entropy
          local loss = cnll:forward(ccout, cctarget) 
                     + spa_cnll:forward(ccout_stn_propsoal, cctarget_stn_proposal) * 10
          ccls_loss = ccls_loss + loss 
          
--          -- debug
--          print('Reg Proposal Classification Loss:')
--          print(cnll:forward(ccout, cctarget))
--          print('Spatially Transformed Proposal Classification Loss:')
--          print(spa_cnll:forward(ccout_stn_propsoal, cctarget_stn_proposal) * 10)
          
          local ccdelta = cnll:backward(ccout, cctarget)  -- dscore of classification scores
          local cc_stn_proposal_delta = spa_cnll:backward(ccout_stn_propsoal, cctarget_stn_proposal) * 10
          
          local post_roi_delta = cnet:backward(cinput, { crdelta, ccdelta })
          local post_roi_stn_proposal_delta = spa_cnet:backward(cinput_stn_proposal, { cr_stn_proposal_delta, cc_stn_proposal_delta })
          
          -- run backward pass over spatially transformed proposals
          for i = 1, stn_proposal_num do
            
--            local patch = img[{{}, {r.minY, math.max(r.maxY-1, r.minY)}, {r.minX, math.max(r.maxX-1, r.minX)}}]                
--            local spa_out = spanet:forward(patch) -- stn needs input of dim (N, C, H, W) or (C, H, W)
--            po_stn_proposal = spa_post_process_net:forward(spa_out):view(kh * kw * cnet_input_planes)
            
--            print(string.format('Back propogating on %d-th proposal', i))

            delta_proposals[i] = spa_post_process_net:backward(stn_out[i], post_roi_stn_proposal_delta[i]:view(cnet_input_planes, kh, kw))
            
--            print(stn_patch[i]:size())
--            print(delta_proposals[i]:size())
--            image.display(stn_patch[i])
--            image.display(delta_proposals[i])

            local dimg_proposal = spanet:backward(stn_patch[i], delta_proposals[i]) -- update weights for proposals
            
--            image.display(dimg_proposal)
--            print(dimg_proposal:size()) -- dstn_out
          end
--          -- debug
--          local stn_params, stn_gradParams = spanet:getParameters()
--          local post_params, post_gradParams = spa_post_process_net:getParameters()
--          print(color.blue '==>'..'post Params:'..torch.norm(post_params))
--          print(color.red '==>'..'post gradParams:'..torch.norm(post_gradParams))
--          print(color.blue '==>'..'stn Params:'..torch.norm(stn_params))
--          print(color.red '==>'..'stn gradParams:'..torch.norm(stn_gradParams))
          spa_post_process_net:updateParameters(stn_proposal_num*1e-5)
          spanet:updateParameters(stn_proposal_num*1e-5)
          
          -- run backward pass over rois
          for i,x in ipairs(roi_pool_state) do
            amp.indices = x.indices
            delta_outputs[#cfg.scales+1][x.input_idx]:add(amp:backward(x.input, post_roi_delta[i]:view(cnet_input_planes, kh, kw))) -- fixed 
          end                    
          
        else
          -- calculate regression loss using smooth L1
          creg_loss = creg_loss + smoothL1:forward(crout, crtarget) * 10
          local crdelta = smoothL1:backward(crout, crtarget) * 10 -- dscore of regression coordinates
          -- calculate classification loss using cross entropy
          local loss = cnll:forward(ccout, cctarget)
          ccls_loss = ccls_loss + loss 
          local ccdelta = cnll:backward(ccout, cctarget)  -- dscore of classification scores

          local post_roi_delta = cnet:backward(cinput, { crdelta, ccdelta })

          -- run backward pass over rois
          for i,x in ipairs(roi_pool_state) do
            amp.indices = x.indices
            delta_outputs[#cfg.scales+1][x.input_idx]:add(amp:backward(x.input, post_roi_delta[i]:view(cnet_input_planes, kh, kw))) -- fixed 
          end          
        end
      end

      -- backward pass of proposal network
      local gi = pnet:backward(img, delta_outputs)
      -- print(string.format('%f; pos: %d; neg: %d', gradient:max(), #p, #n))
      reg_count = reg_count + #p
      cls_count = cls_count + #p + #n

      creg_count = creg_count + #p
      ccls_count = ccls_count + 1

    end

    -- scale gradient
    gradient:div(cls_count)

    local pcls = cls_loss / cls_count     -- proposal classification (bg/fg)
    local preg = reg_loss / reg_count     -- proposal bb regression
    local dcls = ccls_loss / ccls_count   -- detection classification
    local dreg = creg_loss / creg_count   -- detection bb finetuning

--      print(string.format('prop: cls: %f (%d), reg: %f (%d); det: cls: %f, reg: %f', 
--        pcls, cls_count, preg, reg_count, dcls, dreg)
--      )

    table.insert(stats.pcls, pcls)
    table.insert(stats.preg, preg)
    table.insert(stats.dcls, dcls)
    table.insert(stats.dreg, dreg)

    local loss = pcls + preg
    return loss, gradient
  end


  return lossAndGradient
end
