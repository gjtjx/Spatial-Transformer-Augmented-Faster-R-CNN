local torch = require 'torch'
require 'hdf5'
require 'paths'
require 'image'

--[[
local function load(filename)
  local mode = 'binary'
  local referenced = true
  local file = torch.DiskFile(filename, 'r')
  file[mode](file)
  file:referenced(referenced)
  file:longSize(8)
  file:littleEndianEncoding()
  local object = file:readObject()
  file:close()
  return object
end
]]
base_path = '/media/wei/DATA/datasets/vlm/'

img_size = {}
img_size['H'] = 58
img_size['W'] = 58

print('Resize image to height: ' .. img_size['H'] .. ' width: ' .. img_size['W'])

--[[loading hdf5 data of training examples
dataset_size = 'tiny'
train_file = hdf5.open(base_path .. 'subgestures/ASL_hand_train_' .. dataset_size .. '.hdf5', 'r')
X_train = train_file:read('X_train'):all()
y_train = train_file:read('y_train'):all()
train_file:close()

N_train = X_train:size(1)
trainset = {};
-- resize to a good size for input to NN
trainset.data = torch.ByteTensor(N_train,3,img_size['H'],img_size['W']):zero()
trainset.label = torch.ByteTensor(N_train,1):zero()
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
function trainset:size() 
    return self.data:size(1) 
end
for i=1,trainset:size() do 
  -- trainset.data[i] = image.scale(X_train[i], img_size['H'], img_size['W']):byte()
  trainset.data[i] = X_train[i]
  trainset.label[i] = y_train[i] + 1
end
-- save prepceossed dataset into t7
out_path = base_path .. 'subgestures/' .. 'ASL_torch_hand_train_' .. dataset_size .. '.t7'
torch.save(out_path, trainset)
]]

--[[Test set
--loading hdf5 data of test examples
test_file = hdf5.open(base_path .. 'subgestures/ASL_hand_test.hdf5', 'r')
X_test = test_file:read('X_test'):all()
y_test = test_file:read('y_test'):all()
test_file:close()

N_test = X_test:size(1)
testset = {};
-- resize to a good size for input to NN
testset.data = torch.ByteTensor(N_test,3,img_size['H'],img_size['W']):zero()
testset.label = torch.ByteTensor(N_test,1):zero()
setmetatable(testset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
function testset:size() 
    return self.data:size(1) 
end
for i=1,testset:size() do 
  --testset.data[i] = image.scale(X_test[i], img_size['H'], img_size['W']):byte()
  testset.data[i] = X_test[i]
  testset.label[i] = y_test[i] + 1
end
-- save prepceossed dataset into t7
out_path = base_path .. 'subgestures/' .. 'ASL_torch_hand_test.t7'
torch.save(out_path, testset)
]]


--Validation set
--loading hdf5 data of validation examples
val_file = hdf5.open(base_path .. 'subgestures/ASL_hand_val.hdf5', 'r')
X_val = val_file:read('X_val'):all()
y_val = val_file:read('y_val'):all()
val_file:close()

N_val = X_val:size(1)

valset = {};
-- resize to a good size for input to NN
valset.data = torch.ByteTensor(N_val,3,img_size['H'],img_size['W']):zero()
valset.label = torch.ByteTensor(N_val,1):zero()
setmetatable(valset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
function valset:size() 
    return self.data:size(1) 
end
for i=1,valset:size() do 
  --valset.data[i] = image.scale(X_val[i], img_size['H'], img_size['W']):byte()
  valset.data[i] = X_val[i]
  valset.label[i] = y_val[i] + 1
end
-- save prepceossed dataset into t7
out_path = base_path .. 'subgestures/' .. 'ASL_torch_hand_val.t7'
torch.save(out_path, valset, 'ascii')

