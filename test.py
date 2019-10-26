import numpy as np
import torch
from CFA import CFA
import cv2
import options
from utils.log_utils import make_inference_image

opt = options.Options(None)
args = opt.opt


if __name__ == '__main__':

    # model
    model = CFA(output_channel_num=69)
    model.cuda()

    # load weights
    snapshot = torch.load(args.checkpoint_file)
    model.load_state_dict(snapshot['state_dict'])

    # load image
    bbox = args.bbox
    img = cv2.imread(args.test_image)
    img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    img = cv2.resize(img, (args.crop_height, args.crop_width))
    img = img[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
    img = img.astype('float32') / 255.0
    img = torch.from_numpy(img)
    img = img.cuda()

    # inference
    outputs = model(img)

    # make inference image
    dammy_mask = torch.from_numpy(np.ones((1, 68, 1, 1)))
    inference_image = make_inference_image(img, outputs[-1], dammy_mask)

    # show inference image
    cv2.imshow('', inference_image)
    cv2.waitKey()
