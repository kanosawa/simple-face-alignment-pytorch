##############################################################
### Copyright (c) 2018-present, Xuanyi Dong                ###
### Style Aggregated Network for Facial Landmark Detection ###
### Computer Vision and Pattern Recognition, 2018          ###
##############################################################
import argparse
import torch

class Options():
  def __init__(self, model_names):
    parser = argparse.ArgumentParser(description='Train Style Aggregated Network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)   
    parser.add_argument('--train_list',       type=str,   nargs='+', default=['madomagi_train.txt'], help='The list file path to the video training dataset.')
    parser.add_argument('--sigma',            type=float, default=4,      help='sigma distance for CPM.')
    parser.add_argument('--scale_min',        type=float, default=0.8,      help='argument scale : minimum scale factor.')
    parser.add_argument('--scale_max',        type=float, default=1.2,      help='argument scale : maximum scale factor.')
    parser.add_argument('--rotate_max',       type=int, default=30,       help='argument rotate : maximum rotate degree.')
    parser.add_argument('--pre_crop_expand',  type=float, default=0.2,    help='parameters for pre-crop expand ratio')
    parser.add_argument('--crop_height',      type=int, default=128,      help='argument crop : crop height.')
    parser.add_argument('--crop_width',       type=int, default=128,      help='argument crop : crop width.')
    parser.add_argument('--crop_perturb_max', type=int, default=10,       help='argument crop : center of maximum perturb distance.')
    parser.add_argument('--epochs',           type=int,   default=4000,    help='Number of epochs to train.')
    parser.add_argument('--batch_size',       type=int,   default=6,      help='Batch size for training.')
    parser.add_argument('--learning_rate',    type=float, default=1e-4,    help='The Learning Rate.')
    parser.add_argument('--momentum',         type=float, default=0.9,    help='Momentum.')
    parser.add_argument('--decay',            type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--num_pts',          type=int,   default=24,      help='num of landmarks.')
    # Checkpoints
    parser.add_argument('--print_freq',       type=int,   default=200,    metavar='N', help='print frequency (default: 200)')
    parser.add_argument('--resume',           type=str,   default='',     metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--start_epoch',      type=int,   default=0,      metavar='N', help='manual epoch number (useful on restarts)')
    # Acceleration
    parser.add_argument('--gpu_ids',          type=str, default='0',        help='empty for CPU, other for GPU-IDs')
    parser.add_argument('--workers',          type=int,   default=4,      help='number of data loading workers (default: 2)')
    # For Test
    parser.add_argument('--checkpoint_file',  type=str, default='checkpoints/2000.pth.tar', help='checkpoint filename')

    #parser.add_argument('--test_image',       type=str, default='test.jpg', help='test image filename')
    #parser.add_argument('--bbox',             type=int, nargs='+', default=[432, 819, 576, 972], help='bounding box for test image')
    #parser.add_argument('--bbox',             type=int, nargs='+', default=[321, 775, 604, 1011], help='bounding box for test image')
    #parser.add_argument('--bbox',             type=int, nargs='+', default=[264, 276, 538, 481], help='bounding box for test image')

    #parser.add_argument('--test_image',       type=str, default='test2.jpg', help='test image filename')
    #parser.add_argument('--bbox',             type=int, nargs='+', default=[58, 322, 342, 580], help='bounding box for test image')
    #parser.add_argument('--bbox',             type=int, nargs='+', default=[241, 153, 544, 453], help='bounding box for test image')

    #parser.add_argument('--test_image',       type=str, default='/root/datasets/300W/300W/01_Indoor/indoor_002.png', help='test image filename')
    #parser.add_argument('--bbox',             type=int, nargs='+', default=[0, 94, 530, 608], help='bounding box for test image')

    #parser.add_argument('--test_image',       type=str, default='/root/datasets/madomagi/madomagi/000001.jpg', help='test image filename')
    #parser.add_argument('--bbox',             type=int, nargs='+', default=[95, 137, 211, 240], help='bounding box for test image')

    #parser.add_argument('--test_image',       type=str, default='000001.png', help='test image filename')
    #parser.add_argument('--bbox',             type=int, nargs='+', default=[0, 0, 270, 272], help='bounding box for test image')

    #parser.add_argument('--test_image',       type=str, default='000002.png', help='test image filename')
    #parser.add_argument('--bbox',             type=int, nargs='+', default=[0, 0, 241, 253], help='bounding box for test image')

    #parser.add_argument('--test_image',       type=str, default='000003.png', help='test image filename')
    #parser.add_argument('--bbox',             type=int, nargs='+', default=[0, 0, 254, 250], help='bounding box for test image')

    parser.add_argument('--test_image',       type=str, default='000004.png', help='test image filename')
    parser.add_argument('--bbox',             type=int, nargs='+', default=[0, 0, 184, 182], help='bounding box for test image')

    self.opt = parser.parse_args()
    
