local torch = require 'torch'
require 'hdf5'
require 'nn'
require 'cunn';
require 'cutorch';
-- require 'cudnn';
require 'paths'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'image'
color = require 'trepl.colorize'

base_path = '/media/wei/DATA/datasets/vlm/'
model_path = '/home/wei/Dropbox/research/gesture & sl & action/my_hand_detector/models/'

train_dataset_name = 'large'
test_dataset_name = 'val' -- 'gb1113' | 'val' (in 'tb1113+lb1113')| 'test' (in 'tb1113+lb1113')

img_size = {}
img_size['H'] = 58
img_size['W'] = 58

opt = {}
opt['save'] = 'Logs/t1/'
opt['batchSize'] = 500
opt['learningRateDecay'] = 0.95
opt['learningRate'] = 0.0001
opt['weightDecay'] = 0.0001 -- 0.0001
opt['momentum'] = 0.9

opt['model'] = 'LeNet'
opt['epoch_step'] = 30
opt['max_epoch'] = 150
opt['backend'] = 'cudnn'
opt['save_every'] = 30
opt['use_stn'] = true
opt['vis_stn'] = false
opt['use_rot'] = true
opt['use_sca'] = true
opt['use_tra'] = true

local w1, w2

classes = {}
classes[1] = 'Non-hand'
classes[2] = 'Hand'
print('Classes: 1-' .. classes[1] .. ' 2-' .. classes[2])


function setup_model()
  print(color.blue '==>' ..' configuring model')
  model = nn.Sequential()
  -- Copy : add a copy of the input with type casting;
  model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
  model:add(dofile('models/'..opt.model..'.lua'):cuda())

  if opt.use_stn then 
     require 'stn'
     paths.dofile('models/spatial_transformer.lua')
     model:insert(spanet:cuda(),2)
     print(spanet:__tostring())
  end

  -- do not update gradients of Copy layer
  model:get(1).updateGradInput = function(input) return end 

  if opt.backend == 'cudnn' then
     require 'cudnn'
     -- convert ReLU to cudnn
     cudnn.convert(model:get(2), cudnn) 
     if opt.use_stn then 
       cudnn.convert(model:get(3), cudnn)
     end
  end


  print(opt.model..'\n'..model:__tostring());
  --print(model)

  parameters,gradParameters = model:getParameters()

  confusion = optim.ConfusionMatrix(2)
  print('Will save at '..opt.save)
  paths.mkdir(opt.save)
  testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
  testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
  testLogger.showPlot = true

  print(color.blue'==>' ..' setting criterion')
  criterion = nn.CrossEntropyCriterion():cuda()

  print(color.blue'==>' ..' configuring optimizer')
  optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
  }
  
  -- initial model information, for debug:
  model_info = {
    model = opt.model, 
    batchSize = opt.batchSize, 
    n_epoch = opt.max_epoch, 
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
  }
  print(optimState)
end




function load_train_dataset()
  trainset = torch.load(base_path .. 'subgestures/' .. 'ASL_torch_hand_train_' .. train_dataset_name .. '.t7')
  print('Number of training samples:' .. trainset.data:size(1))

  trainset.data = trainset.data:float() -- convert the data from a ByteTensor to a FloatTensor.
  setmetatable(trainset, 
      {__index = function(t, i) 
                      return {t.data[i], t.label[i]} 
                  end}
  );


  mean = {} -- store the mean, to normalize the test set in the future
  stdv  = {} -- store the standard-deviation for the future
  for i=1,3 do -- over each image channel
      mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
      print('Channel ' .. i .. ', Mean: ' .. mean[i])
      trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
      
      stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
      print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
      trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end
end


function train()
  -- sets the mode of the Module (or sub-modules) to train=true
  model:training()
  epoch = epoch or 1
  
  -- for debug:
  --w1=image.display({image=trainset.data:narrow(1, 1, 250), nrow=16, legend='original input, epoch : '..epoch, win=w1})

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(color.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(trainset.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = trainset.data:index(1,v)
    targets:copy(trainset.label:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    --optim.sgd(feval, parameters, optimState)
    optim.adam(feval, parameters, optimState)
  end
  
  if opt.use_stn and opt.vis_stn then
    --print(spanet.output:size())
    --print(tranet:get(2).output:size())
    -- in order to draw successfully the output images, they should be stored in dim of Nx3xHxW
    -- spanet.output is of dim (N, C, H, W), tranet:get(2).output is of (N, H, W, C)--which should be tranposed twice
    w1=image.display({image=spanet.output:narrow(1, 1, 250), nrow=16, legend='STN-transformed inputs, epoch : '..epoch, win=w1}) --
    w2=image.display({image=tranet:get(2).output:transpose(2, 4):transpose(3, 4):narrow(1, 1, 250), nrow=16, legend='Inputs, epoch : '..epoch, win=w2})
  end
  
  confusion:updateValids()
  print(('Train accuracy: '..color.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
  
end


function test()
  -- disable flips, dropouts and batch normalization
  -- sets the mode of the Module (or sub-modules) to train=false
  model:evaluate()
  print((color.blue '==>'.." testing on %s"):format(test_dataset_name))
  
  local bs = 125
  
  for i=1,testset.data:size(1),bs do 
    if i + bs - 1 > testset.data:size(1) then
      bs = testset.data:size(1) - i + 1
    end
    local outputs = model:forward(testset.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, testset.label:narrow(1,i,bs))
  end
  
  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table>'
    file:write'<br>'
    file:write'<h4>Model Information:</h4>'
    file:write'<table>'
    for k,v in pairs(model_info) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table>'
    file:write'<br>'
    file:write(('<h4>Training set:%s</h4>'):format(train_dataset_name))
    file:write(('<h4>Test set:%s</h4>'):format(test_dataset_name))
    file:write(('<h4>%s Performance:</h4>'):format(opt.model))
    file:write'<table>'
    accuracy = (confusion.mat[1][1]+confusion.mat[2][2])/torch.sum(confusion.mat)
    precision = confusion.mat[2][2]/(confusion.mat[2][2]+confusion.mat[1][2])
    sensitivity = confusion.mat[2][2]/(confusion.mat[2][2]+confusion.mat[2][1])
    specificity = confusion.mat[1][1]/(confusion.mat[1][1]+confusion.mat[1][2])
    F_score = 2*(precision*sensitivity)/(precision+sensitivity)
    file:write(('<tr><td>Accuracy: </td><td>%.2f%%</td></tr>\n'):format(100*accuracy))
    file:write(('<tr><td>Precision: </td><td>%.2f%%</td></tr>\n'):format(100*precision))    
    file:write(('<tr><td>Sensitivity(Hand): </td><td>%.2f%%</td></tr>\n'):format(100*sensitivity))
    file:write(('<tr><td>Specificity(Non-Hand): </td><td>%.2f%%</td></tr>\n'):format(100*specificity))
    file:write(('<tr><td>F-score: </td><td>%.2f</td></tr>\n'):format(F_score))
    file:write'</table>'
    file:write'<br>'
    file:write'<pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
  if epoch % opt.save_every == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    --torch.save(filename, model:get(2):clearState())
    torch.save(filename, model:clearState())
  end

  confusion:zero()  
  
end


function load_test_dataset()
  testset = torch.load(base_path .. 'subgestures/' .. 'ASL_torch_hand_'..test_dataset_name..'.t7')
  testset.data = testset.data:float() -- convert the data from a ByteTensor to a FloatTensor.
  setmetatable(testset, 
      {__index = function(t, i) 
                      return {t.data[i], t.label[i]} 
                  end}
  );
  -- print the mean and standard-deviation of example-100
  sample = testset.data[100]
  print(sample:mean(), sample:std())
  print(classes[testset.label[100][1]]) -- tensor element must be accessed by with index [1] in this case
  --itorch.image(valset.data[100])

  -- normalize val samples
  for i=1,3 do -- over each image channel
      testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
      testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end
end


setup_model()
load_train_dataset()
load_test_dataset()


for i=1,opt.max_epoch do
  train()
  test()
end
