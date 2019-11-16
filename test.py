import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageDraw
from CFA import CFA
import options
from utils.log_utils import make_inference_image


opt = options.Options(None)
args = opt.opt


if __name__ == '__main__':

    # model
    model = CFA(output_channel_num=args.num_pts + 1, checkpoint_name=args.checkpoint_file)
    model.cuda()

    # transform
    normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                       std=[0.5, 0.5, 0.5])
    train_transform = [transforms.ToTensor(), normalize]
    train_transform = transforms.Compose( train_transform )

    # load image
    img = Image.open(args.test_image)
    img = img.crop(tuple(args.bbox))
    img_tmp = img.resize((args.crop_width, args.crop_height), Image.BICUBIC)
    img = train_transform(img_tmp)
    img = img.unsqueeze(0).cuda()

    # inference
    heatmaps = model(img)
    heatmaps = heatmaps[-1].cpu().detach().numpy()[0]

    # draw landmarks
    draw = ImageDraw.Draw(img_tmp)
    for i in range(args.num_pts):
        heatmaps_tmp = cv2.resize(heatmaps[i], (args.crop_height, args.crop_width), interpolation=cv2.INTER_CUBIC)
        landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
        landmark_y = landmark[0]
        landmark_x = landmark[1]
        draw.ellipse((landmark_x - 2, landmark_y - 2, landmark_x + 2, landmark_y + 2), fill=(255, 0, 0))

    # show inference image
    img_tmp.show()
    input()
