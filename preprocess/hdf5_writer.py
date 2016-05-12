import os
import matplotlib.pyplot as plt
from sys import getsizeof
import time
from PIL import Image
import numpy as np
import h5py
print h5py.version.info
import utility as util
from hdf5_reader import read_hdf5

debug = 0

"""
Store test data into hdf5 file
"""
def write_hdf5(h5_path, datas, data_names, verbose=True):
  if verbose:
    print 'writing test data into %s' % (h5_path)
  # dimentionality of data stored in hdf5 file : (N, C, H, W), corresponding to format required by torch
  time0 = time.clock()

  with h5py.File(h5_path, 'a') as f:  # for unknown reason, option 'family' is not supported in Windows, maybe try Linux?
    for idx, data in enumerate(datas):
      print 'saving as type %s' % (type(data))
      if isinstance(data, dict):
        for k, v in data.items():
          f.create_dataset(k, data=v, compression='gzip')
      else: # NOTE: you may not want save as 'uint8', which is the most efficient format for images
        f.create_dataset(data_names[idx], data=data, dtype='uint8', compression='gzip')
  if verbose:
    print 'for %d images(labels), saving to hdf5 file takes %.2fs' % (data.shape[0], time.clock() - time0)


def read_data(base_path, set_name, data_type, verbose=True):
  """
  Reading positive or negative test data into memory
  """
  list_path = os.path.join(base_path, 'subgestures', set_name + '_' + data_type + '_files.txt')
  time0 = time.clock()
  with open(list_path, "r") as file:
    filenames = file.read().splitlines()
  if verbose:
    print 'reading file list takes %.2fs' % (time.clock() - time0)

  data = np.zeros((len(filenames), 3, 58, 58), dtype='uint8')

  cnt = 0
  time0 = time.clock()
  for i in xrange(len(filenames)):
    file_path = os.path.join(base_path, 'subgestures', 'gb1113', data_type, filenames[i])
    # print file_path
    img = Image.open(file_path)
    # must be careful with dimentionality on convertion between PIL and ndarray:
    W, H = img.size  # img:(W, H, C)
    if W != 58 or H != 58:
      img = img.resize((58, 58))
    arr_img = util.PIL2array(img)  # arr_img:(H, W, C)
    arr_img = np.transpose(arr_img, (2, 0, 1))  # from (H, W, C) to (C, H, W)
    data[i] = arr_img
    if verbose:
      if i % 2000 == 0 or i == len(filenames) - 1:
        if cnt != 0:
          if i == len(filenames) - 1:
            cur = len(filenames) / 2000 + 1
          else:
            cur = i / 2000
          print 'reading +2000 images into memory takes %.2fs, %d/%d passed' \
                % (time.clock() - time0, cur, len(filenames) / 2000 + 1)
        cnt += 1

  if verbose:
    # Profile:
    print 'memory usage of %d %s images:%.2f MB' % (len(filenames), set_name, getsizeof(data) / 1024 / 1024)
    print 'reading data of size:'
    print data.shape
    print ', with dtype:'
    print data[0].dtype
    print 'on first sample: min-%f, max-%f' % (np.min(data[0][:]), np.max(data[0][:]))
    if debug: # debug
      plt.imshow(1. / 255 * np.transpose(data[0], (1, 2, 0)))  # from (C, H, W) tp (H, W, C)

  return data

def downsample_dataset(in_h5_path, data_names, out_h5_path, factor=4, verbose=True):
  if verbose:
    print 'Downsampling dataset %s...' % (in_h5_path)
  shuffled = False
  time0 = time.clock()
  for data_name in data_names:
    data = read_hdf5(in_h5_path, data_name=data_name, flag=None, verbose=True)
    in_len = data.shape[0]
    out_len = in_len / factor
    if not shuffled: # shuffle indices calculated for only once
      shuffle_ind = np.arange(in_len)
      np.random.shuffle(shuffle_ind)
      shuffle_ind = shuffle_ind[:out_len]
      shuffled = True
    else:
      pass
    out_data = data[shuffle_ind]
    write_hdf5(out_h5_path, datas=[out_data], data_names=[data_name], verbose=True)
  if verbose:
    print 'Downsampling takes %.2fs' % (time.clock() - time0)
