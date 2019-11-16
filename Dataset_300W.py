##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
from __future__ import print_function
from PIL import Image
import os
from os import path as osp
import numpy as np
import warnings
import math

from utils.file_utils import load_list_from_folders, load_txt_file
from utils.pts_utils import generate_label_map_laplacian
from utils.pts_utils import generate_label_map_gaussian
from utils.dataset_utils import pil_loader
from utils.dataset_utils import anno_parser
from point_meta import Point_Meta
import torch
import torch.utils.data as data


class Dataset_300W(data.Dataset):

  def __init__(self, num_pts, train_list, sigma, transform=None):

    self.NUM_PTS = num_pts
    self.train_list = train_list
    self.sigma = sigma
    self.transform = transform
    self.downsample = 4
    self.reset()
    self.load_list()


  def reset(self):
    self.datas = []
    self.labels = []
    self.face_sizes = []


  def append(self, data, label, box):
    self.datas.append( data )
    np_points, _ = anno_parser(label, self.NUM_PTS)
    meta = Point_Meta(self.NUM_PTS, np_points, box, data)
    self.labels.append( meta )


  def load_list(self):

    datas, labels, boxes = [], [], []
    for file_path in self.train_list:
      assert osp.isfile(file_path), 'The path : {} is not a file.'.format(file_path)
      listfile = open(file_path, 'r')
      listdata = listfile.read().splitlines()
      listfile.close()
      for data in listdata:
        alls = data.split(' ')
        if '' in alls: alls.remove('')
        datas.append( alls[0] )
        labels.append( alls[1] )
        box = np.array( [ float(alls[2]), float(alls[3]), float(alls[4]), float(alls[5]) ] )
        boxes.append( box )

    for idx, data in enumerate(datas):
      self.append(datas[idx], labels[idx], boxes[idx])


  def __len__(self):
    return len(self.datas)


  def __getitem__(self, index):
    image = pil_loader( self.datas[index] )
    xtarget = self.labels[index].copy()
    return self._process_(image, xtarget, index)


  def _process_(self, image, xtarget, index):

    # transform the image and points
    if self.transform is not None:
      image, xtarget = self.transform(image, xtarget)
        
    height, width = image.size(1), image.size(2)
    xtarget.apply_bound(width, height)
    Hpoint = xtarget.points.copy()

    target, mask = generate_label_map_gaussian(Hpoint, height//self.downsample, width//self.downsample, self.sigma, self.downsample) # H*W*C
      
    target = torch.from_numpy(target.transpose((2, 0, 1))).type(torch.FloatTensor)
    mask   = torch.from_numpy(mask.transpose((2, 0, 1))).type(torch.BoolTensor)

    return image, target, mask
