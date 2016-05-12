require 'nn'

print(color.blue '==>' ..' configuring model')
--Define a NN model
model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 8, 5, 5, 1, 1, 2, 2))
model:add(nn.SpatialBatchNormalization(8,1e-3))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialConvolution(8, 8, 5, 5, 1, 1, 2, 2)) 
model:add(nn.SpatialBatchNormalization(8,1e-3))
model:add(nn.ReLU())                       -- non-linearity 
---- Will use "ceil" MaxPooling because we want to save as much space as we can
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())     -- A max-pooling operation that looks at 2x2 windows and finds the max.
model:add(nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialBatchNormalization(16,1e-3)) 
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialBatchNormalization(16,1e-3))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())     -- A max-pooling operation that looks at 2x2 windows and finds the max.
model:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialBatchNormalization(32,1e-3))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialBatchNormalization(32,1e-3))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
model:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
model:add(nn.SpatialBatchNormalization(64,1e-3))
model:add(nn.ReLU())                       -- non-linearity 
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
model:add(nn.View(64*14*14))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5

classifier = nn.Sequential()
classifier:add(nn.Linear(64*14*14, 120))             -- fully connected layer (matrix multiplication between input and weights)
classifier:add(nn.ReLU())                       -- non-linearity 
classifier:add(nn.Linear(120, 84))
classifier:add(nn.ReLU())                       -- non-linearity 
classifier:add(nn.Linear(84, 2))            
classifier:add(nn.LogSoftMax())      -- converts the output to a log-probability. Useful for classification problems

model:add(classifier)

--[[or we can read from another source:
model:add(dofile('models/'..opt.model..'.lua'):cuda())
model:get(2).updateGradInput = function(input) return end
]]

-- initialization from MSR:
-- See paper: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n)) --uses uniform random samples rather than gaussian random variables
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution' -- see: https://github.com/torch/torch7/blob/master/init.lua
end

MSRinit(model)

-- do not update gradients of first SpatialBatchNormalization
-- model:get(1).updateGradInput = function(input) return end 

if opt.backend == 'cudnn' then
   require 'cudnn'
   -- convert ReLU to cudnn
   cudnn.convert(model:get(3), cudnn) 
end

model = model:cuda()





