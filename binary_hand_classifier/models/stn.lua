require 'nn'
require 'stn'

------
-- prepare both branches of the st
local ct = nn.ConcatTable()
-- This branch does not modify the input, just change the data layout to bhwd
local branch1 = nn.Transpose({3,4},{2,4})
-- This branch will compute the parameters and generate the grid
local branch2 = nn.Sequential()
--branch2:add(localization_network)
-- Here you can restrict the possible transformation with the "use_*" boolean variables
branch2:add(nn.AffineTransformMatrixGenerator(opt.use_rot, opt.use_sca, opt.use_tra))
branch2:add(nn.AffineGridGeneratorBHWD(58, 58))
ct:add(branch1)
ct:add(branch2)

------
-- Wrap the st in one module
local st_module = nn.Sequential()
st_module:add(ct)
st_module:add(nn.BilinearSamplerBHWD())
-- go back to the bdhw layout (used by all default torch modules)
st_module:add(nn.Transpose({2,4},{3,4}))


local model = nn.Sequential()
model:add(st_module)
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