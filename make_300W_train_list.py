##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import os, sys
import os.path as osp
from scipy.io import loadmat
from utils.file_utils import load_list_from_folders

def load_box(mat_path, cdir):
  mat = loadmat(mat_path)
  mat = mat['bounding_boxes']
  mat = mat[0]
  assert len(mat) > 0, 'The length of this mat file should be greater than 0 vs {}'.format(len(mat))
  all_object = []
  for cobject in mat:
    name = cobject[0][0][0][0]
    bb_detector = cobject[0][0][1][0]
    bb_ground_t = cobject[0][0][2][0]
    image_path = osp.join(cdir, name)
    image_path = image_path[:-4]
    all_object.append( (image_path, bb_detector, bb_ground_t) )
  return all_object

def load_mats(lists):
  all_objects = []
  for dataset in lists:
    cobjects = load_box(dataset[0], dataset[1])
    all_objects = all_objects + cobjects
  return all_objects

def load_all_300w(root_dir):
  mat_dir = osp.join(root_dir, 'Bounding_Boxes')
  pairs = [(osp.join(mat_dir,  'bounding_boxes_lfpw_testset.mat'),   osp.join(root_dir, 'lfpw', 'testset')),
           (osp.join(mat_dir,  'bounding_boxes_lfpw_trainset.mat'),  osp.join(root_dir, 'lfpw', 'trainset')),
           (osp.join(mat_dir,  'bounding_boxes_ibug.mat'),           osp.join(root_dir, 'ibug')),
           (osp.join(mat_dir,  'bounding_boxes_afw.mat'),            osp.join(root_dir, 'afw')),
           (osp.join(mat_dir,  'bounding_boxes_helen_testset.mat'),  osp.join(root_dir, 'helen', 'testset')),
           (osp.join(mat_dir,  'bounding_boxes_helen_trainset.mat'), osp.join(root_dir, 'helen', 'trainset')),]

  all_datas = load_mats(pairs)
  data_dict = {}
  for _, cpair in enumerate(all_datas):
    image_path = cpair[0].replace(' ', '')
    data_dict[ image_path ] = (cpair[1], cpair[2])
  return data_dict

def return_box(image_path, pts_path, all_dict):
  image_path = image_path[:-4]
  assert image_path in all_dict, '{} not find'.format(image_path)
  np_boxes = all_dict[ image_path ]
  box_str = '{:.3f} {:.3f} {:.3f} {:.3f}'.format(np_boxes[1][0], np_boxes[1][1], np_boxes[1][2], np_boxes[1][3])
  return box_str

def make_300W_train_list(root, box_data, output_file_name):
  subsets = ['afw', 'helen', 'ibug', 'lfpw']
  dir_lists = [osp.join(root, subset) for subset in subsets]
  imagelist, _ = load_list_from_folders(dir_lists, ext_filter=['png', 'jpg', 'jpeg'], depth=3)

  with open(output_file_name, 'w') as txtfile:
    for image_path in imagelist:
      basename, _ = osp.splitext(image_path)
      anno_path = basename + '.pts'
      box_str = return_box(image_path, anno_path, box_data)
      txtfile.write('{} {} {}\n'.format(image_path, anno_path, box_str))

if __name__ == '__main__':

  if len(sys.argv) != 3:
    print('usage: make_300W_train_list.py 300W_DATASET_DIRECTORY OUTPUT_FILE_NAME')
    print('example: make_300W_train_list.py /root/datasets/300W 300W_train.txt')
    exit()
  path_300W = sys.argv[1]
  output_file_name = sys.argv[2]
  assert osp.exists(path_300W), '{:} does not exists'.format(path_300W)

  box_datas = load_all_300w(path_300W)
  make_300W_train_list(path_300W, box_datas, output_file_name)
