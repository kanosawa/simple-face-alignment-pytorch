import numpy as np
import cv2
from PIL import Image, ImageDraw

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_inference_image(inputs, landmarks, mask):
    input_img = inputs.cpu().numpy()[0][0]*255
    input_img = input_img.astype(np.uint8)
    landmarks = landmarks.cpu().detach().numpy()[0]
    output_img = Image.fromarray(input_img)
    draw = ImageDraw.Draw(output_img)
    for i in range(landmarks.shape[0] - 1):
        if mask[0][i][0][0] == 1:
            landmark_ = cv2.resize(landmarks[i], (input_img.shape[:2]), interpolation=cv2.INTER_CUBIC)
            index = np.unravel_index(np.argmax(landmark_), landmark_.shape)
            draw.ellipse((index[1]-1, index[0]-1, index[1]+1, index[0]+1), fill=(0))
    output_img = np.asarray(output_img)
    return output_img