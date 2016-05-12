local torch = require 'torch'
require 'hdf5'
require 'image'
require 'utilities'
require 'Rect' 
require 'lfs'
require 'LuaXML'
require 'csvigo'

ASL_BASE_DIR = '/media/wei/DATA/datasets/vlm/snaps/'
ASL_ANNO_DIR = '/media/wei/DATA/datasets/vlm/annotations/'

local img_size = {}
img_size['H'] = 58
img_size['W'] = 58

local ground_truth = {}
local background_folders = {}

-- In faster R-CNN network, it will automatically sample negative examples
local class_names = {}
class_names[1] = 'Hand'
local class_index = {}
class_index['Hand'] = 1

-- format of csv file:
-- image id, image path, hand type, left, top, width, height (left hand), left, top, width, height (right hand),
function import_file(dataset_name, name_table)
    local m = csvigo.load({path = ASL_ANNO_DIR..'det_GT_'..dataset_name..'.csv', mode = 'large'})
    local name = 'Hand'
    
    print(('%d rows loaded'):format(tostring(#m)))


    --format rows: convert coordinates from string to tensor
    local ROI = torch.IntTensor(#m, 8):zero()
--     print('First row:')
--     print(m[1]) -- get element

    for i=1, #m do
        local cnt = 1
        coord = string.sub(m[i][4], 2, -2) -- strip parenthesis
        for word in string.gmatch(coord, '([^,]+)') do

            ROI[i][cnt] = tonumber(word) -- is +1 necessary?
            cnt = cnt + 1
        end
        
        local l_rect = {}
        local r_rect = {}
        local rois = {}
        
        local type = tonumber(m[i][3])
        if type == 1 then  -- only left hand (rarely used)
            l_rect = Rect.new(ROI[i][1], ROI[i][2], ROI[i][1] + ROI[i][3], ROI[i][2] + ROI[i][4])
            table.insert(rois, {rect = l_rect, class_index = class_index[name], class_name = name})
        elseif type == 2 then -- only right hand
            r_rect = Rect.new(ROI[i][5], ROI[i][6], ROI[i][5] + ROI[i][7], ROI[i][6] + ROI[i][8])
            table.insert(rois, {rect = r_rect, class_index = class_index[name], class_name = name})
        elseif type == 3 then -- both hands
            l_rect = Rect.new(ROI[i][1], ROI[i][2], ROI[i][1] + ROI[i][3], ROI[i][2] + ROI[i][4])
            r_rect = Rect.new(ROI[i][5], ROI[i][6], ROI[i][5] + ROI[i][7], ROI[i][6] + ROI[i][8])
            table.insert(rois, {rect = l_rect, class_index = class_index[name], class_name = name})
            table.insert(rois, {rect = r_rect, class_index = class_index[name], class_name = name})
        elseif type == 0 then -- no/bad annotation
        else
            error('Hand type not supported!')
        end

        local image_path = m[i][2]
        table.insert(name_table, image_path)

        ground_truth[image_path] = { image_file_name = image_path, rois = rois }
    end
end



function create_ground_truth_file(dataset_name, background_folders, output_fn)
    --loading csv files of GT information from training set
    local trainset = {}
    local testset = {}

    import_file('train', trainset)
    import_file('test', testset)
    
    print('Ground truth number: '..tablelength(ground_truth))
    print('Training set number: '..#trainset)
    print('test set number: '..#testset)
    
    local file_names = keys(ground_truth)
    -- compile list of background images
    local background_files = {}

    print(string.format('Total images: %d; classes: %d; train_set: %d; validation_set: %d; (Background: %d)', 
    #file_names, #class_names, #trainset, #testset, #background_files
    ))

    save_obj(
        output_fn,
        {
          dataset_name = dataset_name,
          ground_truth = ground_truth,
          training_set = trainset,
          validation_set = testset,
          class_names = class_names,
          class_index = class_index,
          background_files = background_files
        }
    )
    print('Done.')
end


create_ground_truth_file(
  'ASL',
  background_folders,
  ASL_ANNO_DIR..'ASL_det.t7'
)