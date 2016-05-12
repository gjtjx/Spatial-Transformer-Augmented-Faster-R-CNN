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



base_path = '/media/wei/DATA/datasets/vlm/'

----test nn:
--m = nn.SpatialConvolution(1,3,2,2) -- learn 3 2x2 kernels
--print(m.weight) -- initially, the weights are randomly initialized

---- test cuda:
--a = torch.Tensor(5,3) -- construct a 5x3 matrix, uninitialized
--b = torch.rand(3,4)
--c = torch.Tensor(5,4)
--c:mm(a,b) -- store the result of a*b in c
--a = a:cuda()
--b = b:cuda()
--c = c:cuda()
--c:mm(a,b) -- done on GPU

---- test with table
--classes = {key1='Hand', key2='Non-hand'}
--print(classes.key1)
--print(classes.key2)


classes = {}
classes[0] = 'Non-hand'
classes[1] = 'Hand'
print('Classes: 0-' .. classes[0] .. ' 1-' .. classes[1])

--[[loading hdf5 data of training samples]]
local train_file = hdf5.open(base_path .. 'subgestures/ASL_hand_train_tiny.hdf5', 'r')
local X_train = train_file:read('X_train'):all()
local y_train = train_file:read('y_train'):all()
train_file:close()
--[[loading hdf5 data of test examples]]
local test_file = hdf5.open(base_path .. 'subgestures/ASL_hand_test.hdf5', 'r')
local X_test = test_file:read('X_test'):all()
local y_test = test_file:read('y_test'):all()
test_file:close()

N_train = (#X_train)[1]
N_test = (#X_test)[1]
print('Number of training samples:' .. tostring(N_train))
print('Number of test samples:' .. tostring(N_test))


