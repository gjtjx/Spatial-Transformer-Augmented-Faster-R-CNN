import os.path
import numpy as np
from sys import platform as _platform
from sys import getsizeof
import matplotlib.pyplot as plt
import scipy
import h5py
import csv
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from gen_samples import gen_pos_samples
from gen_samples import gen_neg_samples
from gen_samples import perturb_split_pos_samples
from gen_samples import perturb_split_neg_samples
from hdf5_writer import read_data
from hdf5_writer import write_hdf5
from hdf5_writer import downsample_dataset
from hdf5_reader import check_hdf5_file
from hdf5_reader import read_hdf5
from utility import foramt_mat_string_cell
from utility import gen_frame_num
debug = 1

if _platform == "linux" or _platform == "linux2":
  base_path = '/media/wei/DATA/datasets/vlm/'  # linux
elif _platform == "darwin":
  base_path = '/Users/wei/Data/vlm/'  # MAC OS X
elif _platform == "win32":
  base_path = 'H:\\datasets\\vlm'  # Windows

# groups = ['lb1113', 'tb1113']
groups = ['gb1113']


if len(groups) == 2:
  dataset_name = 'train'
else:
  dataset_name = 'test'


def draw_precision_recall_curve_test():
  """
  Loading csv file
  """
  input_fn = '/media/wei/DATA/LnxDropbox/Dropbox/research/gesture-sl-action/my_hand_detector/faster-rcnn.torch/debug/eval.txt'
  out_fn = '/media/wei/DATA/LnxDropbox/Dropbox/research/gesture-sl-action/my_hand_detector/faster-rcnn.torch/debug/precision_recall.png'
  with open(input_fn, 'rb') as csvfile:
    fieldnames = ['precision', 'recall', 'mAP']
    reader = csv.DictReader(csvfile, fieldnames=fieldnames)
    next(reader)  # skip header row

    # data = dict(enumerate(reader))

    precision = []
    recall = []
    for idx, row in enumerate(reader):
      print(float(row['precision']), float(row['recall']))
      precision.append(float(row['precision']))
      recall.append(float(row['recall']))
      if idx == 0:
        mAP = float(row['mAP'])

  print(precision)
  print(recall)

  """
    Draw and save curve
  """
  plt.plot(recall, precision, label='Precision-Recall curve')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.title('Precision-Recall: AUC={0:0.2f}'.format(mAP))
  plt.legend(loc="lower left")
  plt.show()
  plt.savefig(out_fn)


draw_precision_recall_curve_test()
# gen_det_GT_test()


def gen_det_GT_test():
  write_format = 'csv' # 'csv' | 'hdf5'
  paths, ROIs, types = {}, {}, {}
  cnt = 0

  for group in groups:
    """
    % 2. load 'GROUPNAME_annotation.mat'. you get ->
    %    GROUPNAME_annotation, by which you can access information for each
    %    subgestures and the directory, formatted as:
    %    (1) group name | (2) sign ID | (3) dir name | (4) scene name |
    %    (5) start frame Nbr | (6) ending frame Nbr | (7)duration | (8)lexicon
    % NOTE: it starts from 1, so you have to -1 for indexing in python
    """
    mat_path = os.path.join(base_path, 'annotations', group + '_annotation.mat')
    print 'loading ' + mat_path
    mat = scipy.io.loadmat(mat_path)
    mat_data = mat.get(group + '_annotation')
    sign_num, col_size = mat_data.shape

    for i in xrange(sign_num):
      """
      for every subgesture, load its annotation file of GROUPNAME_signID_lexicon.txt, with data formatted as:
      (1) frame Nbr | (2) sign type | (3-6) (left) x, y, width, height | (7-10) (right) x, y, width, height
      """
      anno_file_path = os.path.join(base_path, 'annotations', 'subgestures_hand',
                                    group + '_' + str(mat_data[i, 1][0][0]) + '_' + foramt_mat_string_cell(
                                      mat_data[i, 7]) + '.txt')
      anno = np.loadtxt(anno_file_path, dtype=np.dtype('uint32'))

      for offset, f_idx in enumerate(range(mat_data[i, 4], mat_data[i, 5] + 1)):
        print 'extracting groud truth from gesture %d, frame %s' % (i, gen_frame_num(f_idx))
        pic_path = os.path.join(base_path, 'snaps', group, foramt_mat_string_cell(mat_data[i, 2]),
                                foramt_mat_string_cell(mat_data[i, 3]),
                                gen_frame_num(f_idx) + '.jpg')
        paths[cnt] = pic_path
        types[cnt] = anno[offset, 1]

        ROIs[cnt] = {}
        if anno[offset, 1] == 0:  # no/bad annotation
          ROIs[cnt] = (0, 0, 0, 0, 0, 0, 0, 0)
        elif anno[offset, 1] == 1:  # only left hand (rarely used)
          ROIs[cnt] = (anno[offset, 2], anno[offset, 3], anno[offset, 4], anno[offset, 5], 0, 0, 0, 0)
        elif anno[offset, 1] == 2:  # only right hand
          ROIs[cnt] = (0, 0, 0, 0, anno[offset, 6], anno[offset, 7], anno[offset, 8], anno[offset, 9])
        elif anno[offset, 1] == 3:  # both hands
          ROIs[cnt] = (anno[offset, 2], anno[offset, 3], anno[offset, 4], anno[offset, 5],
                       anno[offset, 6], anno[offset, 7], anno[offset, 8], anno[offset, 9])
        else:
          raise ValueError('Hand gesture type not recognized!')

        cnt += 1

  if write_format == 'csv':
    with open('/media/wei/DATA/datasets/vlm/annotations/det_GT_' + dataset_name + '.csv', 'wb') as csvfile:
      writer = csv.writer(csvfile)
      for i in xrange(cnt):
        row_data = [i, paths[i], types[i], ROIs[i]]
        writer.writerow(row_data)


  elif write_format == 'hdf5':
    with h5py.File('/media/wei/DATA/datasets/vlm/annotations/det_GT_' + dataset_name + '.hdf5', 'w') as f:
      # dt = h5py.special_dtype(vlen=bytes)
      # dset = f.create_dataset('paths', (cnt, ), dtype=dt)
      # for k, v in paths.items():
      #   dset[k] = v
      f.create_dataset('meta', data=cnt)
      path_grp = f.create_group("paths")
      type_grp = f.create_group("types")
      roi_grp = f.create_group("ROIs")
      for k, v in types.items():
        type_grp.create_dataset(str(k), data=v)
      for k, v in paths.items():
        path_grp.create_dataset(str(k), data=v)
      for k, v in ROIs.items():
        roi_grp.create_dataset(str(k), data=v)
  else:
    pass


def gen_gb_dataset_test():
  # gen_neg_samples(base_path, groups)
  perturb_split_neg_samples(base_path)

  gb1113_pos_data = read_data(base_path, set_name='gb1113', data_type='ori_pos', verbose=True)
  gb1113_neg_data = read_data(base_path, set_name='gb1113', data_type='neg', verbose=True)

  # Concatenate and shuffle test positive, negative data
  # note: all X stored in dimentionality of (C, H, W)
  X_test = np.squeeze(np.vstack((gb1113_pos_data, gb1113_neg_data)))
  y_test = np.squeeze(np.vstack((np.ones((gb1113_pos_data.shape[0], 1)), np.zeros((gb1113_neg_data.shape[0], 1)))))
  shuffle_ind = np.arange(X_test.shape[0])
  np.random.shuffle(shuffle_ind)
  X_test = X_test[shuffle_ind]
  y_test = y_test[shuffle_ind]

  test_gb1113_h5_file_path = os.path.join(base_path, 'subgestures', 'ASL_hand_test_gb1113.hdf5')
  write_hdf5(test_gb1113_h5_file_path, datas=[X_test, y_test], data_names=['X_test', 'y_test'], verbose=True)


def write_ori_train_pos_hdf5_test():
  """
  Read, concatenate, shuffle, and write original (without augment) pos/neg train samples into hdf5
  """
  train_ori_pos_data = read_data(base_path, set_name='train', data_type='ori_pos', verbose=True)
  train_neg_data = read_data(base_path, set_name='train', data_type='neg', verbose=True)

  # Concatenate and shuffle test positive, negative data
  # note: all X stored in dimentionality of (C, H, W)
  X_train = np.squeeze(np.vstack((train_ori_pos_data, train_neg_data)))
  y_train = np.squeeze(np.vstack((np.ones((train_ori_pos_data.shape[0], 1)), np.zeros((train_neg_data.shape[0], 1)))))
  shuffle_ind = np.arange(X_train.shape[0])
  np.random.shuffle(shuffle_ind)
  X_train = X_train[shuffle_ind]
  y_train = y_train[shuffle_ind]


  ori_train_h5_file_path = os.path.join(base_path, 'subgestures', 'ASL_hand_train_ori.hdf5')
  write_hdf5(ori_train_h5_file_path, datas=[X_train, y_train], data_names=['X_train', 'y_train'], verbose=True)


def gen_pos_sample_without_augment_test():
  """
  Generate positive samples from video snaps
  """
  gen_pos_samples(base_path, groups, augment=False)

  """
  Split positive samples into training and test set
  """
  perturb_split_pos_samples(base_path, augment=False)


def gen_pos_samples_with_augment_test():
  """
  Generate positive samples from video snaps
  """
  gen_pos_samples(base_path, groups, augment=True)

  """
  Split positive samples into training and test set
  """
  perturb_split_pos_samples(base_path, augment=True)

def gen_neg_samples_test():
  """
  Generate negative samples from video snaps
  """
  gen_neg_samples(base_path, groups)

  """
  Split negative samples into training and test set
  """
  perturb_split_neg_samples(base_path)




def downsample_dataset_test():
  """
  Downsample dataset to 4X smaller
  """
  in_h5_path = os.path.join(base_path, 'subgestures', 'ASL_hand_train_huge.hdf5')

  downsampled_h5_path = os.path.join(base_path, 'subgestures', 'ASL_hand_train_tiny.hdf5')
  downsample_dataset(in_h5_path=in_h5_path, data_names={'X_train', 'y_train'},
                     out_h5_path=downsampled_h5_path, factor=64, verbose=True)

  downsampled_h5_path = os.path.join(base_path, 'subgestures', 'ASL_hand_train_medium.hdf5')
  downsample_dataset(in_h5_path=in_h5_path, data_names={'X_train', 'y_train'},
                     out_h5_path=downsampled_h5_path, factor=16, verbose=True)

  downsampled_h5_path = os.path.join(base_path, 'subgestures', 'ASL_hand_train_large.hdf5')
  downsample_dataset(in_h5_path=in_h5_path, data_names={'X_train', 'y_train'},
                     out_h5_path=downsampled_h5_path, factor=4, verbose=True)


def split_dataset_test():
  """
  Split test datasets into 'test' and 'validation' set
  """
  X_test_all = read_hdf5(os.path.join(base_path, 'subgestures', 'ASL_hand_test_all.hdf5'),
                      data_name='X_test', flag=None, verbose=True)
  y_test_all = read_hdf5(os.path.join(base_path, 'subgestures', 'ASL_hand_test_all.hdf5'),
                      data_name='y_test', flag=None, verbose=True)

  val_frac_wrt_total_test = 0.5 # i.e., N(val) / N(test) = 1
  N_test = X_test_all.shape[0]
  X_val = X_test_all[:(val_frac_wrt_total_test * N_test)]
  y_val = y_test_all[:(val_frac_wrt_total_test * N_test)]
  X_test = X_test_all[(val_frac_wrt_total_test * N_test):]
  y_test = y_test_all[(val_frac_wrt_total_test * N_test):]


  val_h5_file_path = os.path.join(base_path, 'subgestures', 'ASL_hand_val.hdf5')
  write_hdf5(val_h5_file_path, datas=[X_val, y_val], data_names=['X_val', 'y_val'], verbose=True)

  test_h5_file_path = os.path.join(base_path, 'subgestures', 'ASL_hand_test.hdf5')
  write_hdf5(test_h5_file_path, datas=[X_test, y_test], data_names=['X_test', 'y_test'], verbose=True)

  if debug:
    check_hdf5_file(test_h5_file_path, 'X_test', verbose=True)
    check_hdf5_file(val_h5_file_path, 'X_val', verbose=True)


def hdf5_test():
  """
  Read, concatenate, shuffle, and write pos/neg test samples into hdf5
  """
  test_pos_data = read_data(base_path, set_name='test', data_type='pos', verbose=True)
  test_neg_data = read_data(base_path, set_name='test', data_type='neg', verbose=True)

  # Concatenate and shuffle test positive, negative data
  # note: all X stored in dimentionality of (C, H, W)
  X_test = np.squeeze(np.vstack((test_pos_data, test_neg_data)))
  y_test = np.squeeze(np.vstack((np.ones((test_pos_data.shape[0], 1)), np.zeros((test_neg_data.shape[0], 1)))))
  shuffle_ind = np.arange(X_test.shape[0])
  np.random.shuffle(shuffle_ind)
  X_test = X_test[shuffle_ind]
  y_test = y_test[shuffle_ind]

  if debug:
  # debug
    print X_test.shape, y_test.shape
    print test_pos_data.shape
    idx = 1
    plt.imshow(1. / 255 * np.transpose(X_test[idx], (1, 2, 0))) # from (C, H, W) to (H, W, C)
    title_str = '%s' %  ('Hand' if y_test[idx] else 'Non-hand')
    plt.title(title_str)

    idx = 100
    plt.imshow(1. / 255 * np.transpose(X_test[idx], (1, 2, 0))) # from (C, H, W) tp (H, W, C)
    title_str = '%s' %  ('Hand' if y_test[idx] else 'Non-hand')
    plt.title(title_str)

  test_h5_file_path = os.path.join(base_path, 'subgestures', 'ASL_hand_test.hdf5')
  write_hdf5(test_h5_file_path, datas=[X_test, y_test], data_names=['X_test', 'y_test'], verbose=True)


  if debug:
    check_hdf5_file(test_h5_file_path, 'X_test', verbose=True)

  """
  Since positive training data is huge, we store them into hdf5 file first,
  and then read it into a preallocated array, in order to concatenate it with
  negative training data.
  """
  train_pos_data = read_data(base_path, set_name='train', data_type='pos', verbose=True)
  train_neg_data = read_data(base_path, set_name='train', data_type='neg', verbose=True)

  # write positive training data

  write_hdf5(os.path.join(base_path, 'subgestures', 'ASL_hand_train_pos.hdf5'), datas=[train_pos_data],
            data_names=['X_train_pos'], verbose=True)
  write_hdf5(os.path.join(base_path, 'subgestures', 'ASL_hand_train_neg.hdf5'), datas=[train_neg_data],
            data_names=['X_train_neg'], verbose=True)

  if debug:
    check_hdf5_file(os.path.join(base_path, 'subgestures', 'ASL_hand_train_pos.hdf5'), 'X_train_pos', verbose=True)
    check_hdf5_file(os.path.join(base_path, 'subgestures', 'ASL_hand_train_neg.hdf5'), 'X_train_neg', verbose=True)

  # TODO: free train_pos_data and X_train_neg for memory's sake,
  #       or we could simply set a breakpoint here and uncomment the above code
  del train_pos_data; del train_pos_data

  # obtain size of positive/negative training data
  with open(os.path.join(base_path, 'subgestures', 'train_pos_files.txt'), "r") as file:
      filenames = file.read().splitlines()
  X_train_pos_size = len(filenames)

  with open(os.path.join(base_path, 'subgestures', 'train_neg_files.txt'), "r") as file:
    filenames = file.read().splitlines()
  X_train_neg_size = len(filenames)

  # faster memory allocation with np.empty():
  X_train = np.empty((X_train_pos_size + X_train_neg_size, 3, 58, 58), dtype='uint8')

  # faster data loading using read_direct on preallocated numpy array
  X_train = read_hdf5(os.path.join(base_path, 'subgestures', 'ASL_hand_train_pos.hdf5'),
            data_name='X_train_pos', flag='read_directly', data=X_train,
            slice=[0, X_train_pos_size], verbose=True)
  X_train = read_hdf5(os.path.join(base_path, 'subgestures', 'ASL_hand_train_neg.hdf5'),
            data_name='X_train_neg', flag='read_directly', data=X_train,
            slice=[X_train_pos_size, X_train_pos_size + X_train_neg_size], verbose=True)
  # profile
  print 'memory usage of %d images:%.2f MB' % (X_train_pos_size + X_train_neg_size, getsizeof(X_train) / 1024 / 1024)

  """
  Concatenate positive and negative training labels
  Do not shuffle training samples due to unbalanced sample distribution
  NTOE: we may reduce the huge sample set into meidum/small ones for easier processing
  """
  # note: all X stored in dimentionality of (C, H, W)
  y_train = np.squeeze(np.vstack((np.ones((X_train_pos_size, 1)), np.zeros((X_train_neg_size, 1)))))
  #shuffle_ind = np.arange(X_train.shape[0])
  #np.random.shuffle(shuffle_ind)
  #X_train = X_train[shuffle_ind]
  #y_train = y_train[shuffle_ind]

  if debug:
    print X_train.shape, y_train.shape
    idx = 1
    plt.imshow(1. / 255 * np.transpose(X_train[idx], (1, 2, 0))) # from (C, H, W) to (H, W, C)
    title_str = '%s' %  ('Hand' if y_train[idx] else 'Non-hand')
    plt.title(title_str)

    idx = 100
    plt.imshow(1. / 255 * np.transpose(X_train[idx], (1, 2, 0))) # from (C, H, W) tp (H, W, C)
    title_str = '%s' %  ('Hand' if y_train[idx] else 'Non-hand')
    plt.title(title_str)

  train_h5_file_path = os.path.join(base_path, 'subgestures', 'ASL_hand_train.hdf5')
  write_hdf5(train_h5_file_path, datas=[X_train, y_train], data_names=['X_train', 'y_train'], verbose=True)

  if debug:
    check_hdf5_file(train_h5_file_path, 'X_train', verbose=True)



# Done!
