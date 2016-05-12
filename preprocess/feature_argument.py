from PIL import Image
import matplotlib.pyplot as plt
from skimage import transform
import utility as util
import numpy as np

def flip_image(arr_img):
  if arr_img.ndim == 3:
    return arr_img[:,::-1,:]
  else:
    return arr_img[:,::-1]


def augment_data(img):
  rotate_angles = [0, 45, 90, 135, 180, 225, 270, 315]
  scales = 4 # number of downsampling scales
  flip_flags = [True, False]
  cnt = 0
  output_imgs, output_filenames = {}, {}

  for f in flip_flags:
    if f:
      arr_img = util.PIL2array(img)
      # plt.imshow(arr_img)
      f_img = flip_image(arr_img)
      # plt.imshow(f_img)
      f_img = util.array2PIL(f_img)

      """
      # Optional: using affine transformation
      # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
      # So after that we rotate it another 180 degrees to get just the flip.
      shear = 180
      rotation = 180
      tform_augment = transform.AffineTransform(scale=(1, 1), rotation=np.deg2rad(rotation),
                                               shear=np.deg2rad(shear), translation=(0, 0))
      f_img = transform.warp(arr_img, tform_augment, mode='constant', order=3)
      plt.imshow(f_img)

      """
    else:
      f_img = img

    pyramid = tuple(transform.pyramid_gaussian(f_img, downscale=2))
    for p in xrange(scales):
      H, W, chs = pyramid[p].shape
      p_img = util.array2PIL(pyramid[p])
      #plt.imshow(pyramid[p])
      #p_img.show()
      for angle in rotate_angles:
        output = p_img.rotate(angle, expand=True)
        output = output.resize((58, 58))
        output_imgs[cnt] = output
        # output.show()
        """
        if f:
          output.save('samples/' + 'flipped'+ '_p' + str(p+1) + '_r' + str(angle) + '.jpg')
        else:
          output.save('samples/' + 'p' + str(p + 1) + '_r' + str(angle) + '.jpg')
        """
        if f:
          output_filenames[cnt] = 'flipped' + '_p' + str(p + 1) + '_r' + str(angle) + '.jpg'
        else:
          output_filenames[cnt] = 'p' + str(p + 1) + '_r' + str(angle) + '.jpg'

        cnt += 1
  return output_imgs, output_filenames