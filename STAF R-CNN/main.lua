require 'torch'
require 'pl'
require 'optim'
require 'image'
require 'nngraph'
require 'cunn'
require 'nms'
require 'gnuplot'
require 'utilities'
require 'Anchors'
require 'BatchIterator'
require 'objective'
require 'Detector'
require 'utilities'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'csvigo'
color = require 'trepl.colorize'


-- command line options
cmd = torch.CmdLine()
cmd:addTime()

cmd:text()
cmd:text('Training a convnet for region proposals')
cmd:text()

cmd:text('=== Training ===')
cmd:option('-cfg', 'config/ASL_hand.lua', 'configuration file')
cmd:option('-model', 'models/hand.lua', 'model factory file')
cmd:option('-name', 'stn_low_lr_enable_post', 'experiment name, snapshot prefix') 
cmd:option('-train', '/media/wei/DATA/datasets/vlm/annotations/ASL_det_mix.t7', 'training data file name')
cmd:option('-restore', '', 'network snapshot file name to load')
cmd:option('-snapshot', 500, 'snapshot interval')
cmd:option('-lr', 1E-4, 'learn rate')
cmd:option('-rms_decay', 0.9, 'RMSprop moving average dissolving factor')
cmd:option('-opti', 'rmsprop', 'Optimizer')
cmd:option('-max_epoch', 30, 'number of epoches to train')
cmd:option('-backend', 'cuda', 'NOTE: nngraph is not supposed to work with nngraph now')
cmd:text('=== Misc ===')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-gpuid', 0, 'device ID (CUDA), (use -1 for CPU, 0 for GPU)')
cmd:option('-seed', 0, 'random seed (0 = no fixed seed)')
-- added by Wei:
cmd:text('=== Debug ===')
cmd:option('-save', 'debug/stn_low_lr_enable_post/', 'path to save evaluation log files')
-- during training
cmd:option('-eval', 1000, 'validation batch evaluation interval')
cmd:option('-plot', 200, 'plot training progress interval')
cmd:option('-eval_num', 100, 'number of validation images to evaluate')
cmd:option('-eval_vis_num', 10, 'number of validation images to evaluate and visualize')
cmd:option('-stn_vis_every', 200, 'interval to save visualization results of stn')
-- during evaluation
cmd:option('-save_eval_every', 100, 'interval to save evaluation results (mAP)')

-- parse configuration
print('Command line args:')
local opt = cmd:parse(arg or {})
print(opt)

print('Options:')
local cfg = dofile(opt.cfg)
print(cfg)

-- system configuration
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpuid + 1)  -- nvidia tools start counting at 0
torch.setnumthreads(opt.threads)
if opt.seed ~= 0 then
  torch.manualSeed(opt.seed)
  cutorch.manualSeed(opt.seed)
end

-- Log configuration
if cfg.test_logger == true then
  print('Will save logs at '..opt.save)
  paths.mkdir(opt.save)
  paths.mkdir(opt.save..'eval_vis')
  paths.mkdir(opt.save..'tmp')
  paths.mkdir(opt.save..'anchors')
  paths.mkdir(opt.save..'eval')
  if cfg.use_stn then
    paths.mkdir(opt.save..'stn')
  end
  trainloss_logger = optim.Logger(paths.concat(opt.save, 'train_loss.log'))
  trainloss_logger:setNames{'Loss (train set)'}
  trainloss_logger.showPlot = true
end


function plot_training_progress(prefix, stats)
  local fn = opt.save..prefix..'_progress.png'
  gnuplot.pngfigure(fn)
  gnuplot.title('Traning progress over time')
  
  local xs = torch.range(1, #stats.pcls)
  
  gnuplot.plot(
    { 'pcls', xs, torch.Tensor(stats.pcls), '-' },
    { 'preg', xs, torch.Tensor(stats.preg), '-' },
    { 'dcls', xs, torch.Tensor(stats.dcls), '-' },
    { 'dreg', xs, torch.Tensor(stats.dreg), '-' }
  )
 
  gnuplot.axis({ 0, #stats.pcls, 0, 10 })
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')
  
  gnuplot.plotflush()
end

function plot_det_result(prefix, batchNO, det_mAP)
  local fn = opt.save..prefix..'_det_mAP.png'
  gnuplot.pngfigure(fn)
  gnuplot.title('Detection Accuracy Over Time (IoU=0.3)')
--  gnuplot.plot(torch.Tensor(det_mAP)) -- FIXME
  local xs = torch.range(256, batchNO[#batchNO], 256)
  gnuplot.plot({'', xs, torch.Tensor(det_mAP), '-'})
  gnuplot.axis({ 0, batchNO[#batchNO], 0, 70 })
  gnuplot.xlabel('batch #')
  gnuplot.ylabel('mAP (%)')
  gnuplot.plotflush()
end

function load_model(cfg, model_path, network_filename, cuda)

  -- get configuration & model
  local model_factory = dofile(model_path)
  local model = model_factory(cfg)
  
  if cuda then
    model.cnet:cuda()
    model.pnet:cuda()
    if cfg.use_stn then
      model.spanet:cuda()
      model.spa_post_process_net:cuda()
    end
  end
  
  -- combine parameters from pnet and cnet into flat tensors
  local weights = nil
  local gradient = nil
  if cfg.use_stn then
    weights, gradient = combine_and_flatten_parameters(model.pnet, model.cnet, model.spanet, model.spa_post_process_net)
  else
    weights, gradient = combine_and_flatten_parameters(model.pnet, model.cnet)
  end
  local training_stats
  if network_filename and #network_filename > 0 then
    print('Restoring networks from '..network_filename)
    local stored = load_obj(network_filename)
    training_stats = stored.stats
    weights:copy(stored.weights)
  end

  return model, weights, gradient, training_stats
end


function print_outputs(m)
  print(m)
  print(m.output)
end


function graph_training(cfg, model_path, snapshot_prefix, training_data_filename, network_filename)
  print('Reading training data file \'' .. training_data_filename .. '\'.')
  local training_data = load_obj(training_data_filename)
  local file_names = keys(training_data.ground_truth)
  print(string.format("Training data loaded. Dataset: '%s'; Total files: %d; classes: %d; Background: %d)", 
      training_data.dataset_name, 
      #file_names,
      #training_data.class_names,
      #training_data.background_files))
  
  -- create/load model
  local model, weights, gradient, training_stats = load_model(cfg, model_path, network_filename, true)
  if not training_stats then
    training_stats = { pcls={}, preg={}, dcls={}, dreg={} }
  end
  
  local batch_iterator = BatchIterator.new(model, training_data)
  local eval_objective_grad = create_objective(opt, model, weights, gradient, batch_iterator, training_stats)
  
  local rmsprop_state = { learningRate = opt.lr, alpha = opt.rms_decay }
  --local nag_state = { learningRate = opt.lr, weightDecay = 0, momentum = opt.rms_decay }
  --local sgd_state = { learningRate = opt.lr, weightDecay = 0.0005, momentum = 0.9 }
  
  print(color.blue'==>' ..' configuring optimizer')
  print(rmsprop_state)
  
  -- For ASL final (train on lb+tb, test on gl) dataset:
  -- Total images: 101445; train_set: 74618; validation_set: 26827; (Background: 0)	
  -- For ASL test (train and test on lb+tb) dataset:
  -- Total images: 74618; train_set: 59694; validation_set: 14924; (Background: 0)	
  
  -- in ASL dataset, we need approximately 10 images to extract 256 pos+neg samples, 
  local batches_per_epoch = math.floor(tablelength(training_data.training_set)/(cfg.batch_size / 8))
  
  if cfg.use_stn then
    if cfg.vis_stn then
      require 'image'
    end
  end
      
  
  local det_mAP = {}
  glb_img_scanned = 0
  glb_b_cnt = 1
  
  -- restore option
  local restore_ind = 1
  if network_filename and #network_filename > 0 then
    restore_ind = tonumber(string.match(network_filename, '%d%d%d%d%d%d')) + 1 -- +1 starts from next batch
    print(color.blue '==>'..'Restore and start from batch # '..restore_ind)
  end

  glb_b_cnt = restore_ind
  
  -- decay corresponding learning rate
  restore_ind = restore_ind - 1
  while restore_ind / 5000 > 0 do
    opt.lr = opt.lr / 2
    rmsprop_state.lr = opt.lr
    restore_ind = restore_ind - 5000
  end
  
  for epoch = 1, opt.max_epoch do
    print(color.blue '==>'..'epoch # '..epoch..', batch # to train till next evaluation '..opt.eval..' [batchSize = ' .. cfg.batch_size .. ']')
    
    while glb_img_scanned < #training_data.training_set do
      
      print(color.blue '==>'..'image # scanned '..glb_img_scanned)
      
      xlua.progress((glb_b_cnt-1)%opt.eval+1, opt.eval)
      
      if glb_b_cnt % 5000 == 0 then -- learning rate decay every 5000 batches
        opt.lr = opt.lr / 2
        rmsprop_state.lr = opt.lr
      end
      
      --print(color.blue '==>'..'epoch # '..epoch..', batch # '..i..' [batchSize = ' .. cfg.batch_size .. ']')

      local _, loss = optim.rmsprop(eval_objective_grad, weights, rmsprop_state)
      --local _, loss = optim.nag(eval_objective_grad, weights, nag_state)
      --local _, loss = optim.sgd(eval_objective_grad, weights, sgd_state)
        
      print(color.red '==>'..'loss: '..loss[1]) -- 1st is the value before optimization
      
      if trainloss_logger then
        trainloss_logger:add{loss[1]}
      end
      
      if glb_b_cnt % opt.eval == 0 then -- evaluate every opt.eval batches
        local mAP = evaluation_batch(batch_iterator, model, opt.eval_num, math.floor(glb_b_cnt/opt.eval), opt.eval_vis_num) 
        det_mAP[#det_mAP+1] = mAP
        local batchNO = {}
        for j = 1, #det_mAP do
          batchNO[#batchNO+1] = j*cfg.batch_size
        end
        plot_det_result(snapshot_prefix, batchNO, det_mAP)
        print(color.red '==>'..string.format('mAP at IoU 0.3: %.2f', mAP))
        
        local t = {batchNO = batchNO, mAP = det_mAP}    
        csvigo.save{path=(opt.save..'eval_batch.txt'), data=t}
      end
            
      if glb_b_cnt % opt.eval == 0 and trainloss_logger then
        trainloss_logger:style{'-'}
        trainloss_logger:plot()
        local base64im
        do
          os.execute(('convert -density 200 %s/train_loss.log.eps %s/train_loss.png'):format(opt.save,opt.save))
          os.execute(('openssl base64 -in %s/train_loss.png -out %s/train_loss.base64'):format(opt.save,opt.save))
          local f = io.open(opt.save..'/train_loss.base64')
          if f then base64im = f:read'*all' end
        end

        local file = io.open(opt.save..'/report.html','w')
        file:write(([[
        <!DOCTYPE html>
        <html>
        <body>
        <title>%s - %s</title>
        <img src="data:image/png;base64,%s">
        <h4>Parameters:</h4>
        <table>
        ]]):format(opt.save,epoch,base64im))
        for k,v in pairs(cfg) do
          if torch.type(v) == 'number' then
            file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
          end
        end
        for k,v in pairs(opt) do
          if torch.type(v) == 'number' then
            file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
          end
        end
        file:write'</table>'
        file:write'<h4>Performance:</h4>'
        file:write'<table>'
        local precision = 0.0 -- TODO
        local recall = 0.0 -- TODO
        file:write(('<tr><td>Detection mAP, IoU=0.3: </td><td>%.2f%%</td></tr>\n'):format(det_mAP[#det_mAP]))
        file:write(('<tr><td>Precision: </td><td>%.2f%%</td></tr>\n'):format(100*precision))    
        file:write(('<tr><td>Recall: </td><td>%.2f%%</td></tr>\n'):format(100*recall))
        file:write'</table>\n'
        file:write'<h4>Proposal Network:</h4>'
        file:write'<pre>\n'
        file:write(model.cnet:__tostring()..'\n')
        file:write'</pre>\n'
        file:write'<h4>Classification Network:</h4>'
        file:write'<pre>\n'
        file:write(model.pnet:__tostring()..'\n')
        file:write'</pre></body></html>'
        file:close()
      end

      if glb_b_cnt % opt.plot == 0 then
        plot_training_progress(snapshot_prefix, training_stats)
      end
      
      if glb_b_cnt % opt.snapshot == 0 then
        -- save snapshot
        save_model(string.format((opt.save..'tmp/%s_%06d.t7'), snapshot_prefix, glb_b_cnt), weights, opt, training_stats)
      end
    
      glb_b_cnt = glb_b_cnt + 1
    end
    
    glb_img_scanned = 0 -- reset global image count
  end

  -- compute positive anchors, add anchors to ground-truth file
end


function load_image_auto_size(fn, target_smaller_side, max_pixel_size, color_space)
  local img = image.load(path.join(base_path, fn), 3, 'float')
  local dim = img:size()
  
  local w, h
  if dim[2] < dim[3] then
    -- height is smaller than width, set h to target_size
    w = math.min(dim[3] * target_smaller_side/dim[2], max_pixel_size)
    h = dim[2] * w/dim[3]
  else
    -- width is smaller than height, set w to target_size
    h = math.min(dim[2] * target_smaller_side/dim[1], max_pixel_size)
    w = dim[3] * h/dim[2]
  end
  
  img = image.scale(img, w, h)
  
  if color_space == 'yuv' then
    img = image.rgb2yuv(img)
  elseif color_space == 'lab' then
    img = image.rgb2lab(img)
  elseif color_space == 'hsv' then
    img = image.rgb2hsv(img)
  end

  return img, dim
end


--batch_size: number of images you want to evaluate from validation dataset
--            images are taken in order using nextValidation()
--vis_num: visualized the first say 50 validation results
--eval_idx: n-th time the model was evaluated
function evaluation_batch(batch_iterator, model, batch_size, eval_idx, vis_num) 
  
  -- imgs = batch_iterator:nextValidation(batch_size) -- validate in rotation
  imgs = batch_iterator:batch_validation(batch_size) -- validate certain batch (the first batch_size in validation set )
  
  local red = torch.Tensor({1,0,0})
  local green = torch.Tensor({0,1,0})
  local blue = torch.Tensor({0,0,1})
  local white = torch.Tensor({1,1,1})
  local colors = { red, green, blue, white }
  local test_matches = {}
  local test_GT_rois = {}
  local IoU_ths = {}
  for i = 0, 1.0, 0.05 do
    IoU_ths[#IoU_ths+1] = i
  end
  
  -- create detector
  local d = Detector(model)
  
  --evaluate on first batch_size images
  print(('Test on first %d validation images:'):format(batch_size))
--  glb_cnt = 1
  
  for i = 1, batch_size do
--    glb_cnt = glb_cnt + 1
    xlua.progress(i, batch_size)
    
    -- pick random validation image
    local b = imgs[i]
    local img = b.img
    local matches = d:detect(img, b.ori_img)
    
    test_matches[i] = matches
    test_GT_rois[i] = b.rois
      
    if i <= vis_num then
      local vis_img = b.ori_img
      -- draw bounding boxes and save image
      for i,m in ipairs(matches) do
        draw_rectangle(vis_img, m.r, green)
      end
      for i,m in ipairs(b.rois) do
        draw_rectangle(vis_img, m.rect, white)
      end
      image.saveJPG(string.format((opt.save..'eval_vis/eval%d_output%d.jpg'), eval_idx, i), vis_img)
    end
    
  end
  
  local mAPs = {}
  for j = 1, #IoU_ths do
    local ap = ASL_det_eval(test_matches, test_GT_rois, IoU_ths[j]) -- in an accumutive style 
    io.write(('mAP(IoU=%f): %.4f%%\n'):format(IoU_ths[j], ap*100))
    -- NOTE: we only evaluate on mAP!!!
    mAPs[#mAPs+1] = ap*100
  end
  local t = {IoU = IoU_ths, mAP = mAPs}    
  csvigo.save{path=(opt.save..'eval.txt'), data=t}
  
  return mAPs[7] -- return mAP when IoU == 0.3
end



function evaluation_demo(cfg, model_path, training_data_filename, network_filename, save_eval_every)
  -- load trainnig data
  local training_data = load_obj(training_data_filename)
  
  -- load model
  local model = load_model(cfg, model_path, network_filename, true)
  local batch_iterator = BatchIterator.new(model, training_data)
    
  local red = torch.Tensor({1,0,0})
  local green = torch.Tensor({0,1,0})
  local blue = torch.Tensor({0,0,1})
  local white = torch.Tensor({1,1,1})
  local colors = { red, green, blue, white }
  
  -- create detector
  local d = Detector(model)
  
  local val_num = #training_data.validation_set
--  local val_num = 8
  
  print(color.blue '==>'..'Evaluating on '..val_num..' images:')
  
  local test_matches = {}
  local test_GT_rois = {}
  local IoU_ths = {}
  for i = 0, 1.0, 0.05 do
    IoU_ths[#IoU_ths+1] = i
  end
  print('Evaluating with IoU:')
  print(IoU_ths)

  for i=1, val_num do
    xlua.progress(i, val_num)
    
    -- pick random validation image
    local b = batch_iterator:nextValidation(1)[1]
    local img = b.img
    local matches = d:detect(img, b.ori_img)
    
    test_matches[i] = matches
    test_GT_rois[i] = b.rois
    
    -- debug
--    print(matches)
--    print(b.rois)
--    print('candidate # '..tablelength(matches))
    
    -- every interval, write evaluation results to temporary files
    if i % save_eval_every == 0 then
      local mAPs = {}
      for j = 1, #IoU_ths do
        local ap = ASL_det_eval(test_matches, test_GT_rois, IoU_ths[j]) -- in an accumutive style 
        io.write(('mAP(IoU=%f): %.4f%%\n'):format(IoU_ths[j], ap*100))
        -- NOTE: we only evaluate on mAP!!!
        mAPs[#mAPs+1] = ap*100
      end
      
      local t = {IoU = IoU_ths, mAP = mAPs}    
      local save_path = string.format((opt.save..'eval%d.txt'), i/save_eval_every)
      csvigo.save{path=save_path, data=t}
    end
    
--    print(matches)
    -- draw bounding boxes of matches and ground truth boxes, then save image
    local vis_img = b.ori_img
    for i,m in ipairs(matches) do
      draw_rectangle(vis_img, m.r, green)
    end
    for i,m in ipairs(b.rois) do
      draw_rectangle(vis_img, m.rect, white)
    end
    image.saveJPG(string.format((opt.save..'eval/output%d.jpg'), i), vis_img)
  end
  
--  local ap, recall, precision = ASL_det_eval(test_matches, test_GT_rois, IoU_th) 
--  local t_recall = tensor2table_1D(recall)
--  local t_precision = tensor2table_1D(precision)
--  local t = {precision = t_precision, recall = t_recall, mAP = {ap}}
  
  -- write result when evaluation finished
  local mAPs = {}
  for j = 1, #IoU_ths do
    local ap = ASL_det_eval(test_matches, test_GT_rois, IoU_ths[j]) -- in an accumutive style 
    io.write(color.red ''..('mAP(IoU=%f): %.4f%%\n'):format(IoU_ths[j], ap*100))
    -- NOTE: we only evaluate on mAP!!!
    mAPs[#mAPs+1] = ap*100
  end
  local t = {IoU = IoU_ths, mAP = mAPs}    
  csvigo.save{path=(opt.save..'eval.txt'), data=t}
  
end

------ ASL_det_test | ASL_det_final | ASL_det_mix| ASL_det_test_100val | ASL_det_test_2000val
--opt.train = '/media/wei/DATA/datasets/vlm/annotations/ASL_det_mix.t7' 
--graph_training(cfg, opt.model, opt.name, opt.train, opt.restore)

-- ASL_det_test | ASL_det_final | ASL_det_mix| ASL_det_test_100val | ASL_det_test_2000val
opt.train = '/media/wei/DATA/datasets/vlm/annotations/ASL_det_mix.t7' 
opt.restore = opt.save..'tmp/stn_low_lr_enable_post_007500.t7'
evaluation_demo(cfg, opt.model, opt.train, opt.restore, opt.save_eval_every)

