import os
import glob
import cv2

img_fname_list = glob.glob('/root/datasets/face_landmark/**/*.png', recursive=True)
with open('madomagi_train.txt', 'w') as file_w:
    for img_fname in img_fname_list:
        dirname, basename = os.path.split(img_fname)
        rootname, _ = os.path.splitext(basename)
        pts_fname = dirname + '/' + rootname + '.pts'
        img = cv2.imread(img_fname)
        file_w.write(img_fname + ' ' + pts_fname + ' 0 0 ' + \
            str(img.shape[1]) + ' ' + str(img.shape[0]) + '\n')

#print(file_list)

