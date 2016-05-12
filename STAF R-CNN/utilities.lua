require 'lfs' -- lua file system for directory listings
require 'nn'
require 'image'

function list_files(directory_path, max_count, abspath)
  local l = {}
  for fn in lfs.dir(directory_path) do
    if max_count and #l >= max_count then
      break
    end
    local full_fn = path.join(directory_path, fn)
    if lfs.attributes(full_fn, 'mode') == 'file' then 
      table.insert(l, abspath and full_fn or fn)
    end
  end
  return l
end


function clamp(x, lo, hi)
  return math.max(math.min(x, hi), lo)
end


function saturate(x)
  return clam(x, 0, 1)
end


function lerp(a, b, t)
  return (1-t) * a + t * b
end


function shuffle_n(array, count)
  count = math.max(count, count or #array)
  local r = #array    -- remaining elements to pick from
  local j, t
  for i=1,count do
    j = math.random(r) + i - 1
    t = array[i]    -- swap elements at i and j
    array[i] = array[j]
    array[j] = t
    r = r - 1
  end
end


function shuffle(array)
  local i, t
  for n=#array,2,-1 do
    i = math.random(n)
    t = array[n]
    array[n] = array[i]
    array[i] = t
  end
  return array
end


function shallow_copy(t)
  local t2 = {}
  for k,v in pairs(t) do
    t2[k] = v
  end
  return t2
end


function deep_copy(obj, seen)
  if type(obj) ~= 'table' then 
    return obj 
  end
  if seen and seen[obj] then 
    return seen[obj] 
  end
  local s = seen or {}
  local res = setmetatable({}, getmetatable(obj))
  s[obj] = res
  for k, v in pairs(obj) do 
    res[deep_copy(k, s)] = deep_copy(v, s) 
  end
  return res
end


function reverse(array)
  local n = #array, t 
  for i=1,n/2 do
    t = array[i]
    array[i] = array[n-i+1]
    array[n-i+1] = t
  end
  return array
end


function remove_tail(array, num)
  local t = {}
  for i=num,1,-1 do
    t[i] = table.remove(array)
  end
  return t, array
end


function keys(t)
  local l = {}
  for k,v in pairs(t) do
    table.insert(l, k)
  end
  return l
end


function values(t)
  local l = {}
  for k,v in pairs(t) do
    table.insert(l, v)
  end
  return l
end


function save_obj(file_name, obj)
  local f = torch.DiskFile(file_name, 'w')
  f:writeObject(obj)
  f:close()
end


function load_obj(file_name)
  local f = torch.DiskFile(file_name, 'r')
  local obj = f:readObject()
  f:close()
  return obj
end


function save_model(file_name, weights, options, stats)
  save_obj(file_name,
  {
    version = 0,
    weights = weights,
    options = options,
    stats = stats
  })
end


function combine_and_flatten_parameters(...)
  local nets = { ... }
  local parameters,gradParameters = {}, {}
  for i=1,#nets do
    local w, g = nets[i]:parameters()
    for i=1,#w do
      table.insert(parameters, w[i])
      table.insert(gradParameters, g[i])
    end
  end
  return nn.Module.flatten(parameters), nn.Module.flatten(gradParameters)
end


function draw_rectangle(img, rect, color)
  local sz = img:size()
  
  local x0 = math.max(1, rect.minX)
  local x1 = math.min(sz[3], rect.maxX)
  local w = math.floor(x1) - math.floor(x0)
  if w >= 0 then
    local v = color:view(3,1):expand(3, w + 1)
    if rect.minY > 0 and rect.minY <= sz[2] then
      img[{{}, rect.minY, {x0, x1}}] = v
    end
    if rect.maxY > 0 and rect.maxY <= sz[2] then
      img[{{}, rect.maxY, {x0, x1}}] = v
    end
  end
  
  local y0 = math.max(1, rect.minY)
  local y1 = math.min(sz[2], rect.maxY)
  local h = math.floor(y1) - math.floor(y0)
  if h >= 0 then
    local v = color:view(3,1):expand(3, h + 1)
    if rect.minX > 0 and rect.minX <= sz[3] then
      img[{{}, {y0, y1}, rect.minX}] = v 
    end
    if rect.maxX > 0 and rect.maxX <= sz[3] then
      img[{{}, {y0, y1}, rect.maxX}] = v
    end
  end
end


function remove_quotes(s)
  return s:gsub('^"(.*)"$', "%1")
end


function normalize_debug(t)
  local lb, ub = t:min(), t:max()
  return (t -lb):div(ub-lb+1e-10)
end


function find_target_size(orig_w, orig_h, target_smaller_side, max_pixel_size)
  local w, h
  if orig_h < orig_w then
    -- height is smaller than width, set h to target_size
    w = math.min(orig_w * target_smaller_side/orig_h, max_pixel_size)
    h = math.floor(orig_h * w/orig_w + 0.5)
    w = math.floor(w + 0.5)
  else
    -- width is smaller than height, set w to target_size
    h = math.min(orig_h * target_smaller_side/orig_w, max_pixel_size)
    w = math.floor(orig_w * h/orig_h + 0.5)
    h = math.floor(h + 0.5)
  end
  assert(w >= 1 and h >= 1)
  return w, h
end


function load_image(fn, color_space, base_path)
  if not path.isabs(fn) and base_path then
    fn = path.join(base_path, fn)
  end
  local img = image.load(fn, 3, 'float')
  if color_space == 'yuv' then
    img = image.rgb2yuv(img)
  elseif color_space == 'lab' then
    img = image.rgb2lab(img)
  elseif color_space == 'hsv' then
    img = image.rgb2hsv(img)
  end
  return img
end


function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end


function swap(array, index1, index2)
    array[index1], array[index2] = array[index2], array[index1]
end


function shuffle(array)
  local counter = #array
  while counter > 1 do
    local index = math.random(counter)
    swap(array, index, counter)
    counter = counter - 1
  end
end


function range(init, limit, step)
  step = step or 1
  return function(lim, value)
    value = value + step
    if lim * step >= value * step then
      return value
    end
  end, limit, init - step
end


function tensor2table_1D(src)
  local dst = {}
  for i=1,src:size(1) do
    dst[i] = src[i]
  end
  return dst
end


function tensor2table_2D(src)
  local dst = {}
  for i=1,src:size(1) do
    dst[i] = {}
    for j=1,src:size(2) do
      dst[i][j] = src[i][j]
    end
  end
  return dst
end


--------------------------------------------------------------------------------
-- Evaluation toolbox
--------------------------------------------------------------------------------
local function ASLap(rec,prec)
  
  local mrec = rec:totable()
  local mpre = prec:totable()
  table.insert(mrec,1,0); table.insert(mrec,1)
  table.insert(mpre,1,0); table.insert(mpre,0)
  for i=#mpre-1,1,-1 do
      mpre[i]=math.max(mpre[i],mpre[i+1])
  end
  
  local ap = 0
  for i=1,#mpre-1 do
    if mrec[i] ~= mrec[i+1] then
      ap = ap + (mrec[i+1]-mrec[i])*mpre[i+1]
    end
  end
  return ap
end


local function boxoverlap(a,b)
  --local b = anno.objects[j]
  local b = b.xmin and {b.xmin,b.ymin,b.xmax,b.ymax} or b
    
  local x1 = a:select(2,1):clone()
  x1[x1:lt(b[1])] = b[1] 
  local y1 = a:select(2,2):clone()
  y1[y1:lt(b[2])] = b[2]
  local x2 = a:select(2,3):clone()
  x2[x2:gt(b[3])] = b[3]
  local y2 = a:select(2,4):clone()
  y2[y2:gt(b[4])] = b[4]
  
  local w = x2-x1+1;
  local h = y2-y1+1;
  local inter = torch.cmul(w,h):float()
  local aarea = torch.cmul((a:select(2,3)-a:select(2,1)+1) ,
                           (a:select(2,4)-a:select(2,2)+1)):float()
  local barea = (b[3]-b[1]+1) * (b[4]-b[2]+1);
  
  -- intersection over union overlap
  local o = torch.cdiv(inter , (aarea+barea-inter))
  -- set invalid entries to 0 overlap
  o[w:lt(0)] = 0
  o[h:lt(0)] = 0
  
  return o
end


function ASL_det_eval(test_matches, test_GT_rois, IoU_th)
  local num_pr = 0
  local energy = {}
  local correct = {} -- flags indicating if scored box is correct over for specific roi over all test images
  local count = 0 -- # of rois
  
  for i=1, #test_GT_rois do  -- over all test images
    local bbox = {} -- ground truth boxes for single test image
    local det = {} -- flag indicating if certain rois were detected for single test image
    local matches = test_matches[i]
    local GT_rois = test_GT_rois[i]
    
    for idx,obj in ipairs(GT_rois) do
      if obj.class_name == 'Hand' then
        local r = obj.rect                         
        table.insert(bbox,{r.minX, r.minY,
                           r.maxX, r.maxY})
        table.insert(det,0)
        count = count + 1
      end
    end
    
    bbox = torch.Tensor(bbox)
    det = torch.Tensor(det)
    
    local num = tablelength(matches)
    for j=1,num do -- for one test image, go over all scored boxes
      local r = matches[j].r
      local bbox_pred = torch.Tensor({r.minX, r.minY,
                                      r.maxX, r.maxY,
                                      math.exp(matches[j].confidence)})
                           
      num_pr = num_pr + 1 -- an iterator of scored boxes over all test images
      table.insert(energy, bbox_pred[5]) -- energy stores all prediction scores for all test images
      
      if bbox:numel() > 0 then -- if ground truth data exists
        local o = boxoverlap(bbox, bbox_pred[{{1,4}}])
        local maxo,index = o:max(1)
        maxo = maxo[1]
        index = index[1]
        if maxo >= IoU_th and det[index] == 0 then -- current roi is not detected yet and IoU is good
          correct[num_pr] = 1
          det[index] = 1
        else
          correct[num_pr] = 0
        end
      else
          correct[num_pr] = 0        
      end
    end
    
  end
  
  if #energy == 0 then -- no confidence score was given
    return 0,torch.Tensor(),torch.Tensor()
  end
  
  energy = torch.Tensor(energy)
  correct = torch.Tensor(correct)
  
  -- sort confidence and corresponding correct flag in descending order
  local threshold,index = energy:sort(true)

  correct = correct:index(1,index)

  local n = threshold:numel() -- total matched boxes over all test images
  
  local recall = torch.zeros(n)
  local precision = torch.zeros(n)

  local num_correct = 0

  for i = 1,n do -- for each of top i candidates, calculate precision and recall
      --compute precision
      num_positive = i -- top i of candidates, i.e. TP + FP
      num_correct = num_correct + correct[i] -- total # of correctly detected boxes, i.e. TP
      if num_positive ~= 0 then
          precision[i] = num_correct / num_positive; -- TP / (TP + FP)
      else
          precision[i] = 0;
      end
      
      --compute recall
      recall[i] = num_correct / count -- TP / # of rois = TP / (TP + FN)
  end

  ap = ASLap(recall, precision)
--  io.write(('AP = %.4f\n'):format(ap));

  return ap, recall, precision
end
