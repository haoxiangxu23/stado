import cv2
import numpy as np
import random
from math import floor


# def randomShade(x, y, w, h, n=2):
#     # x,y - coordinate of top-let corner

#     x1 = x
#     x2 = x + w
#     y1 = y
#     y2 = y + h

#     width = x2 - x1
#     height = y2 - y1
#     area = width*height/n

#     min_w = floor(area/height)
#     min_h = floor(area/width)

#     new_w = random.randint(min_w, width-1)
#     new_h = floor(area/new_w)

#     x1_ = random.randint(x1, x1+(width-new_w))
#     x2_ = x1_ + new_w

#     y1_ = random.randint(y1, y1+(height-new_h))
#     y2_ = y1_ + new_h

#     return x1_,y1_,x2_,y2_

def randomShade(rect, n=2):
    # x,y - coordinate of top-let corner

    x1 = rect[0]
    x2 = x1 + rect[2]
    y1 = rect[1]
    y2 = y1 + rect[3]

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


def randomShade(x1, y1, x2, y2, n=2):
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
