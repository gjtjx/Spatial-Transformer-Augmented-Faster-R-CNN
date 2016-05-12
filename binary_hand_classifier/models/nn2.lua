require 'nn'

local model = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  model:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  model:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  model:add(nn.ReLU(true))
  return model
end

---- Will use "ceil" MaxPooling because we want to save as much space as we can

--Define a NN model
ConvBNReLU(3, 16):add(nn.Dropout(0.5))
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil()) 
ConvBNReLU(16, 32):add(nn.Dropout(0.5))
ConvBNReLU(32, 32):add(nn.Dropout(0.5))
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil()) 
ConvBNReLU(32, 64):add(nn.Dropout(0.5))
ConvBNReLU(64, 64):add(nn.Dropout(0.5))
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
ConvBNReLU(64, 128):add(nn.Dropout(0.5))
ConvBNReLU(128, 128):add(nn.Dropout(0.5))
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
ConvBNReLU(128, 256):add(nn.Dropout(0.5))
ConvBNReLU(256, 256):add(nn.Dropout(0.5))
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
model:add(nn.View(256*2*2))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*2*2, 128))      
classifier:add(nn.BatchNormalization(128))
classifier:add(nn.ReLU(true))               
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(128, 2))            
classifier:add(nn.LogSoftMax())            

model:add(classifier)

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

return model
