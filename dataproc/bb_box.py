import scipy.io
import cv2
import numpy as np
from math import floor
import random


# mat = scipy.io.loadmat('puppet_mask/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0/puppet_mask.mat')

# array = np.array(mat["part_mask"][:,:,0])

# contours, _ = cv2.findContours(array.copy() ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # not copying here will throw
# an error
# rect = cv2.boundingRect(contours[0]) # basically you can feed this rect into your classifier x,y,w,
# h = rect # a - angle print(rect)

def get_bb_box(path):
    mat = scipy.io.loadmat(path)
    array = np.array(mat["part_mask"])
    out = []
    for i in range(array.shape[2]):
        contours, _ = cv2.findContours(array[:, :, i].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.boundingRect(contours[0])
        out.append([rect[2] * rect[3], rect])
    # print(w,h)
    return max(out)[1]


def randomShade(x1, y1, x2, y2, n=4):
    # x,y - coordinate of top-let corner

    width = x2 - x1
    height = y2 - y1
    area = width * height / n

    min_w = floor(area / height)
    min_h = floor(area / width)

    new_w = random.randint(min_w, width - 1)
    new_h = floor(area / new_w) if area / new_w < height else height

    x1_ = random.randint(x1, x1 + (width - new_w))
    x2_ = x1_ + new_w

    y1_ = random.randint(y1, y1 + (height - new_h))
    y2_ = y1_ + new_h

    return (x1_, y1_, x2_, y2_), (x1_, y1_, new_w, new_h)


def get_bb_from_gt(gttubes, n):
    # There is only frame list in each video
    occlusion = {}
    for x in gttubes:
        frame = list(gttubes[x].values())[0][0]
        area = np.zeros(frame.shape[0])
        area = (frame[:, 3] - frame[:, 1]) * (frame[:, 4] - frame[:, 2])

        # get the frame with max area
        max_frame = np.argmax(area)
        x1, y1, x2, y2 = frame[max_frame][1:5].astype("int64")

        occlusion[x] = (randomShade(x1, y1, x2, y2, n), (x1, y1, x2, y2))
    return occlusion
