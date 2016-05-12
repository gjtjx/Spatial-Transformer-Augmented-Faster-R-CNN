require 'nn'

local model = nn.Sequential()

model:add(nn.SpatialConvolution(3, 6, 5, 5, 1, 1, 2, 2)) 
model:add(nn.ReLU())                
model:add(nn.SpatialMaxPooling(2,2,2,2))    
model:add(nn.SpatialConvolution(6, 16, 5, 5, 1, 1, 2, 2))
model:add(nn.ReLU())                      
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.View(16*14*14))                   
model:add(nn.Linear(16*14*14, 120))            
model:add(nn.ReLU())                     
model:add(nn.Linear(120, 84))
model:add(nn.ReLU())                     
model:add(nn.Linear(84, 2))              
model:add(nn.LogSoftMax())              

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
