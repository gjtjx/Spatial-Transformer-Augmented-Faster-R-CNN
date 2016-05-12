import numpy as np
from PIL import Image

def normalize(arr):
    arr = arr.astype('float32')
    if arr.max() > 1.0:
        arr /= 255.0
    return arr

def gen_frame_num(idx):
  return '%s' % ('%.4d' % (idx))

def foramt_mat_string_cell(cell):
  str_cell = str(cell)
  return str_cell[3:-2]

def PIL2array(img):
  return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
# return  numpy.asarray(img)

def array2PIL(arr):
  return Image.fromarray(np.uint8(255 * normalize(arr)))
