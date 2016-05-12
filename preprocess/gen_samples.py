from PIL import Image
import scipy.io
import numpy as np
import pylab
import matplotlib.pyplot as plt
import os.path
import time
import random
from skimage import exposure, img_as_float
from utility import gen_frame_num
from utility import foramt_mat_string_cell
from utility import array2PIL
from utility import PIL2array
from feature_argument import augment_data

def gen_output_path(out_path, group, mat_data, f_idx, filenames, type):
  if type == 0:
    pass
  elif type == 1:
    type_str = 'lhand'
  elif type == 2:
    type_str = 'rhand'
  else:
    pass
  if filenames == '.jpg':
    filenames = os.path.join(out_path,
                             group + '_' +
                             foramt_mat_string_cell(mat_data[2]) + '_' +
                             foramt_mat_string_cell(mat_data[3]) + '_' +
                             gen_frame_num(f_idx) + '_' + type_str + filenames)
  else:
    filenames = {idx: os.path.join(out_path,
                                   group + '_' +
                                   foramt_mat_string_cell(mat_data[2]) + '_' +
                                   foramt_mat_string_cell(mat_data[3]) + '_' +
                                   gen_frame_num(f_idx) + '_' +
                                   type_str + '_' + f)
                 for idx, f in filenames.iteritems()}
  return filenames

def write_images(imgs, output_path):
  if type(imgs) is list:
    for i in xrange(len(imgs)):
      imgs[i].save(output_path[i])
  else:
    imgs.save(output_path)


def gen_pos_samples(base_path, groups, augment=False):
  if augment:
    out_path = os.path.join(base_path, 'subgestures', 'all', 'aug_pos')
  else:
    out_path = os.path.join(base_path, 'subgestures', 'all', 'ori_pos')

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

    print 'start preprocess %d subgestures in group %s' % (sign_num, group)

    for i in xrange(sign_num):
      """
      for every subgesture, load its annotation file of GROUPNAME_signID_lexicon.txt, with data formatted as:
      (1) frame Nbr | (2) sign type | (3-6) (left) x, y, width, height | (7-10) (right) x, y, width, height
      """
      anno_file_path = os.path.join(base_path, 'annotations', 'subgestures_hand',
                                    group + '_' + str(mat_data[i, 1][0][0]) + '_' + foramt_mat_string_cell(mat_data[i, 7]) + '.txt')
      anno = np.loadtxt(anno_file_path, dtype=np.dtype('uint32'))

      for offset, f_idx in enumerate(range(mat_data[i, 4], mat_data[i, 5] + 1)):

        print 'sampling positive data on gesture %d, frame %s' % (i, gen_frame_num(f_idx))
        pic_path = os.path.join(base_path, 'snaps', group, foramt_mat_string_cell(mat_data[i, 2]),
                                  foramt_mat_string_cell(mat_data[i, 3]),
                                  gen_frame_num(f_idx) + '.jpg')
        img = Image.open(pic_path)

        if anno[offset, 1] == 0: # no/bad annotation
          pass
        elif anno[offset, 1] == 1:  # only left hand (rarely used)
          cropped = img.crop((anno[offset, 2], anno[offset, 3],
                              anno[offset, 2] + anno[offset, 4],
                              anno[offset, 3] + anno[offset, 5]))
          if augment:
            output_imgs, output_filenames = augment_data(cropped)
          else:
            output_imgs, output_filenames = cropped, '.jpg'
          output_filenames = gen_output_path(out_path, group, mat_data[i, :], f_idx, output_filenames, 1)
          write_images(output_imgs, output_filenames)

        elif anno[offset, 1] == 2: # only right hand
          cropped = img.crop((anno[offset,6], anno[offset,7],
                              anno[offset,6] + anno[offset,8],
                              anno[offset,7] + anno[offset,9]))
          if augment:
            output_imgs, output_filenames = augment_data(cropped)
          else:
            output_imgs, output_filenames = cropped, '.jpg'
          output_filenames = gen_output_path(out_path, group, mat_data[i, :], f_idx, output_filenames, 2)
          write_images(output_imgs, output_filenames)

        elif anno[offset, 1] == 3: # both hands
          cropped = img.crop((anno[offset, 2], anno[offset, 3],
                              anno[offset, 2] + anno[offset, 4],
                              anno[offset, 3] + anno[offset, 5])) # left
          if augment:
            output_imgs, output_filenames = augment_data(cropped)
          else:
            output_imgs, output_filenames = cropped, '.jpg'
          output_filenames = gen_output_path(out_path, group, mat_data[i, :], f_idx, output_filenames, 1)
          write_images(output_imgs, output_filenames)

          cropped = img.crop((anno[offset, 6], anno[offset, 7],
                              anno[offset, 6] + anno[offset, 8],
                              anno[offset, 7] + anno[offset, 9])) # right
          if augment:
            output_imgs, output_filenames = augment_data(cropped)
          else:
            output_imgs, output_filenames = cropped, '.jpg'

          output_filenames = gen_output_path(out_path, group, mat_data[i, :], f_idx, output_filenames, 2)
          write_images(output_imgs, output_filenames)

        else:
          raise ValueError('Hand gesture type not recognized!')


def extract_file_info(filename):
  info = {}
  metas = filename.split('_')
  info['group'] = metas[0]
  info['dir'] = '_'.join(metas[1:5])
  info['scene'] = metas[5]
  info['frame idx'] = metas[6]
  info['sign type'] = metas[7]
  if metas[8] == 'flipped':
    info['flipped'] = True
    info['p'] = int(metas[9][1])
    info['r'] = int(metas[10][1:-4])
  else:
    info['flipped'] = False
    info['p'] = int(metas[8][1])
    info['r'] = int(metas[9][1:-4])

  return info


def integrate_file_info(info):
  filename = ''
  filename += info['group'] + '_' + info['dir'] + '_' + info['scene'] + '_' + \
              info['frame idx'] + '_' + info['sign type'] + '_'
  if info['flipped'] == True:
    filename += 'flipped' + '_'
  filename += 'p' + str(info['p']) + '_'
  filename += 'r' + str(info['r'])
  filename += '.jpg'

  return filename


def perturb_split_pos_samples(base_path, augment=False):
  """
  Write all file names of augmented data to a list
  :param augment:
  """

  # # A fix:
  # filename_list_path = os.path.join(base_path, 'subgestures', 'test_pos_files.txt')
  # with open(filename_list_path, 'r') as file:
  #   test_pos_filenames = file.read().splitlines()
  #
  # filename_list_path = os.path.join(base_path, 'subgestures', 'ori_pos_files.txt')
  # with open(filename_list_path, 'r') as file:
  #   ori_pos_filenames = file.read().splitlines()
  #
  # filename_list_path = os.path.join(base_path, 'subgestures', 'train_ori_pos_files.txt')
  # with open(filename_list_path, 'w') as file:
  #   for i in xrange(len(ori_pos_filenames)):
  #     ori_filename = ori_pos_filenames[i]
  #     if not ori_filename in test_pos_filenames:
  #       new_ori_filename = ori_filename[:-10] + '.jpg'  # format data
  #       file.writelines("%s\n" % new_ori_filename)


  if augment:
    filename_list_path = os.path.join(base_path, 'subgestures', 'aug_pos_files.txt')
  else:
    filename_list_path = os.path.join(base_path, 'subgestures', 'ori_pos_files.txt')

  if os.path.isfile(filename_list_path):
    time0 = time.clock()
    with open(filename_list_path, "r") as file:
      pos_filenames = file.read().splitlines()
    print 'reading file list from raw text takes %.2fs' % (time.clock() - time0)
  else:
    if augment:
      data_path = os.path.join(base_path, 'subgestures', 'all', 'aug_pos')
    else:
      data_path = os.path.join(base_path, 'subgestures', 'all', 'ori_pos')
    time0 = time.clock()
    pos_filenames = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    print 'reading file list from direcotory takes %.2fs' % (time.clock() - time0)
    with open(filename_list_path, 'w') as file:
      file.writelines("%s\n" % item for item in pos_filenames)
    pos_filenames = map(lambda s: s.strip(), pos_filenames)

  """
  Write all file names of original data (before augmentation) to a list
  """
  filename_list_path = os.path.join(base_path, 'subgestures', 'ori_pos_files.txt')
  if os.path.isfile(filename_list_path):
    time0 = time.clock()
    with open(filename_list_path, "r") as file:
      ori_filenames = file.read().splitlines()
    print 'reading file list from raw text takes %.2fs' % (time.clock() - time0)
  else:
    ori_filenames = []
    with open(filename_list_path, 'w') as file:
      for i in xrange(len(pos_filenames)):
        info = extract_file_info(pos_filenames[i])
        if info['flipped'] == False and info['p'] == 1 and info['r'] == 0:
          file.writelines("%s\n" % pos_filenames[i])

  ori_filenames = np.array(ori_filenames)
  mask = range(len(ori_filenames))
  train_ind = sorted(random.sample(mask, int(0.8 * len(ori_filenames))))
  test_ind = list(set(mask) ^ set(train_ind))
  test_filenames = list(ori_filenames[test_ind])
  train_filenames = list(ori_filenames[train_ind])

  if augment:
    filename_list_path = os.path.join(base_path, 'subgestures', 'train_ori_pos_files.txt')
    with open(filename_list_path, 'w') as file:
      file.writelines("%s\n" % item for item in train_filenames)
  else:
    """
    Remove those samples which we think may not aid performance in training
    """
    filename_list_path = os.path.join(base_path, 'subgestures', 'train_aug_pos_files.txt')
    with open(filename_list_path, 'w') as file:
      for i in xrange(len(pos_filenames)):
        info = extract_file_info(pos_filenames[i])
        tmp = info.copy()
        tmp['p'], tmp['r'] = 1, 0
        ori_filename = integrate_file_info(tmp)
        if info['p'] != 4 and ori_filename in train_filenames: # data downsampled to 4th layer looks fuzzy, so remove it
          file.writelines("%s\n" % ori_filename[i])

  """
  Write file names of test set to a list
  """
  filename_list_path = os.path.join(base_path, 'subgestures', 'test_pos_files.txt')
  with open(filename_list_path, 'w') as file:
    file.writelines("%s\n" % item for item in test_filenames)


def gen_neg_samples(base_path, groups):

  out_path = os.path.join(base_path, 'subgestures', 'gb1113', 'neg')

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

    print 'start preprocess %d subgestures in group %s' % (sign_num, group)

    for i in xrange(sign_num):
      """
      for every subgesture, load its annotation file of GROUPNAME_signID_lexicon.txt, with data formatted as:
      (1) frame Nbr | (2) sign type | (3-6) (left) x, y, width, height | (7-10) (right) x, y, width, height
      """
      anno_file_path = os.path.join(base_path, 'annotations', 'subgestures_hand',
                                    group + '_' + str(mat_data[i, 1][0][0]) + '_' + foramt_mat_string_cell(mat_data[i, 7]) + '.txt')
      anno = np.loadtxt(anno_file_path, dtype=np.dtype('uint32'))

      for offset, f_idx in enumerate(range(mat_data[i, 4], mat_data[i, 5] + 1)):

        print 'sampling negative data on gesture %d, frame %s' % (i, gen_frame_num(f_idx))
        pic_path = os.path.join(base_path, 'snaps', group, foramt_mat_string_cell(mat_data[i, 2]),
                                foramt_mat_string_cell(mat_data[i, 3]),
                                gen_frame_num(f_idx) + '.jpg')
        img = Image.open(pic_path)
        W, H = img.size
        arr_img = PIL2array(img)
        mask_hands = np.zeros_like(arr_img)
        # need to check whether x indicates row here:
        if anno[offset, 1] == 0:  # no/bad annotation
          pass
        elif anno[offset, 1] == 1:  # only left hand (rarely used)
          mask_hands[anno[offset, 2]:(anno[offset, 2] + anno[offset, 4] - 1),
                     anno[offset, 3]:(anno[offset, 3] + anno[offset, 5] - 1)] = 1
          area_hands = anno[offset, 5] * anno[offset, 4]
        elif anno[offset, 1] == 2:  # only right hand
          mask_hands[anno[offset, 7]:(anno[offset, 7] + anno[offset, 9] - 1),
                     anno[offset, 6]:(anno[offset, 6] + anno[offset, 8] - 1)] = 1
          area_hands = anno[offset, 8] * anno[offset, 9]
        elif anno[offset, 1] == 3:  # both hands
          mask_hands[anno[offset, 2]:(anno[offset, 2] + anno[offset, 4] - 1),
                     anno[offset, 3]:(anno[offset, 3] + anno[offset, 5] - 1)] = 1
          mask_hands[anno[offset, 7]:(anno[offset, 7] + anno[offset, 9] - 1),
                     anno[offset, 6]:(anno[offset, 6] + anno[offset, 8] - 1)] = 1
          area_hands = min(anno[offset, 5] * anno[offset, 4], anno[offset, 8] * anno[offset, 9])
        else:
          raise ValueError('Type enumerate not recognized!')

        # for each snap, we randomly generate 5 negative samples
        for cnt in xrange(5):
          # extract a fixed [80, 80] region for tb, lb datasets, [65, 65] for gb dataset,
          # then resize it to [58, 58]
          left, top = random.randint(1, W - 57), random.randint(1, H - 57)
          if group == 'gb1113':
            right, bottom = min(W, left + 65 - 1), min(H, top + 65 - 1)
          else:
            right, bottom = min(W, left + 80 - 1), min(H, top + 80 - 1)

          mask_cropped = np.zeros_like(arr_img)
          mask_cropped[left:right, top:bottom] = 1
          mask = np.bitwise_and(mask_hands, mask_cropped)

          if group == 'gb1113':
            area_cropped = 65 * 65
          else:
            area_cropped = 80 * 80

          cropped = img.crop((left, top, right, bottom)).resize((58, 58))

          # tmp = exposure.histogram(img_as_float(cropped), nbins=8)
          # if exposure.is_low_contrast(cropped, fraction_threshold=0.02) == True:
          #   # without the line below, the figure won't show
          #   pylab.show(); plt.imshow(PIL2array(cropped))
          # else:
          #   pylab.show(); plt.imshow(PIL2array(cropped))

          """
          Remove those samples which we think may not aid performance in training
          """
          if group == 'gb1113':
            contrast_th = 0.1
          else:
            contrast_th = 0.1

          # for cropped image: if the ratio of overlapped region against hand regions has exceeded 0.1, or it has lower
          # contrast, then skip it:
          if 1.0 * np.count_nonzero(mask[:]) / min(area_cropped, area_hands) > 0.1 \
              or exposure.is_low_contrast(cropped, fraction_threshold=contrast_th) == True:
            # pylab.show()
            # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            # ax1.imshow(arr_img); ax1.set_title('arr_img')
            # ax2.imshow(mask_hands * 255); ax2.set_title('mask_hands')
            # ax3.imshow(mask_cropped * 255); ax3.set_title('mask_cropped')
            # ax4.imshow(mask * 255); ax4.set_title('mask')
            continue

          out = os.path.join(out_path, group + '_' +
                             foramt_mat_string_cell(mat_data[i, 2]) + '_' +
                             foramt_mat_string_cell(mat_data[i, 3]) + '_' +
                             gen_frame_num(f_idx) + '_' +
                             'rand_' + str(cnt+1) + '.jpg')
          cropped.save(out)


def perturb_split_neg_samples(base_path):
  """
  Write all file names of negative samples into a list
  """
  filename_list_path = os.path.join(base_path, 'subgestures', 'gb1113_neg_files.txt')
  if os.path.isfile(filename_list_path):
    time0 = time.clock()
    with open(filename_list_path, "r") as file:
      neg_filenames = file.read().splitlines()
    print 'reading file list from raw text takes %.2fs' % (time.clock() - time0)
  else:
    data_path = os.path.join(base_path, 'subgestures', 'gb1113', 'neg')
    time0 = time.clock()
    neg_filenames = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    print 'reading file list from direcotory takes %.2fs' % (time.clock() - time0)
    with open(filename_list_path, 'w') as file:
      file.writelines("%s\n" % item for item in neg_filenames)
    neg_filenames = map(lambda s: s.strip(), neg_filenames) # remove newline from all elements

  # ori_filenames = np.array(ori_filenames)
  mask = range(len(neg_filenames))
  train_ind = sorted(random.sample(mask, int(0.8 * len(neg_filenames))))
  test_ind = list(set(mask) ^ set(train_ind))
  neg_filenames = np.array(neg_filenames)
  test_filenames = list(neg_filenames[test_ind])
  train_filenames = list(neg_filenames[train_ind])

  """
    Split and write file names of negative samples for training and test sets
  """
  filename_list_path = os.path.join(base_path, 'subgestures', 'test_neg_files.txt')
  with open(filename_list_path, 'w') as file:
    file.writelines("%s\n" % item for item in test_filenames)
  filename_list_path = os.path.join(base_path, 'subgestures', 'train_neg_files.txt')
  with open(filename_list_path, 'w') as file:
    file.writelines("%s\n" % item for item in train_filenames)


