require 'nngraph'
require 'stn'

local input = nn.Identity()()
local stn = nn.Sequential()
local concat = nn.ConcatTable()

-- first branch is there to transpose inputs to BHWD, for the bilinear sampler
local tranet = nn.Sequential()
tranet:add(nn.Identity())  

---- batch setting
--tranet:add(nn.Transpose({2,3},{3,4})) -- (N, C, H, W) to (N, H, C, W) to (N, H, W, C)
-- single setting
tranet:add(nn.Transpose({1,2},{2,3})) -- (C, H, W) to (H, C, W) to (H, W, C)

-- second branch is the localization network
local locnet = nn.Sequential()
locnet:add(nn.SpatialConvolution(3,20,5,5,1,1,2,2))
locnet:add(nn.ReLU(true))
locnet:add(nn.SpatialMaxPooling(2,2,2,2))
locnet:add(nn.SpatialConvolution(20,20,5,5,1,1,2,2))
locnet:add(nn.ReLU(true))
locnet:add(nn.SpatialAdaptiveMaxPooling(6, 6))
locnet:add(nn.View(20*6*6))
locnet:add(nn.Linear(20*6*6,128))
locnet:add(nn.ReLU(true))

-- we initialize the output layer so it gives the identity transform
local outLayer = nn.Linear(128,6)
outLayer.weight:fill(0)
local bias = torch.FloatTensor(6):fill(0)
bias[1]=1
bias[5]=1
outLayer.bias:copy(bias)
locnet:add(outLayer)

-- there we generate the grids
locnet:add(nn.View(2,3))
locnet:add(nn.AffineGridGeneratorBHWD(58,58))

-- we need a table input for the bilinear sampler, so we use concattable
concat:add(tranet)
concat:add(locnet)

stn:add(concat)
stn:add(nn.BilinearSamplerBHWD())

-- and we transpose back to standard BDHW format for subsequent processing by nn modules
------ batch setting
--spanet:add(nn.Transpose({3,4},{2,3}))
-- single setting
stn:add(nn.Transpose({2,3},{1,2}))

local out = stn(input)
local spanet = nn.gModule({ input }, { out })

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

MSRinit(spanet)

return spanet