import os
import matplotlib.pyplot as plt
from sys import getsizeof
import time
import numpy as np
import h5py
print h5py.version.info

debug = 1

"""
Do a sanity check on hdf5 file
"""
def check_hdf5_file(h5_path, data_name, verbose=True):
  data = read_hdf5(h5_path, data_name, verbose=True)
  if verbose:
    print 'memory usage of %d images:%.2f MB' % (data.shape[0], getsizeof(data)/1024/1024)
  idx = 1
  if debug:
    plt.imshow(1. / 255 * np.transpose(data[idx], (1, 2, 0))) # from (C, H, W) tp (H, W, C)
    # title_str = '%s' % ('Hand' if data[idx] else 'Non-hand')
    # plt.title(title_str)

def read_hdf5(h5_path, data_name, flag=None, data=None, slice=None, verbose=True):
  """
  Note on the non-trivial use of h5py API read_direct when read huge dataset
  """
  if verbose:
    print 'reading data from %s' % (h5_path)
  time0 = time.clock()
  with h5py.File(h5_path, 'r') as f:
    if flag == 'read_directly':
      dset = f[data_name]
      if slice is None:
        dset.read_direct(data[...])
      else:
        dset.read_direct(data[slice[0]:slice[1]])
    else:
      data = f[data_name][...]
  if verbose:
    print 'reading data %s from hdf5 takes %.2fs' % (data_name, time.clock() - time0)

  return data

