import cv2
import numpy as np
import random
from math import floor
from get_file import get_files

def randomShade(x1, y1, x2, y2, n=4):
    width = x2 - x1
    height = y2 - y1
    area = width*height/n

    min_w = floor(area/height)
    min_h = floor(area/width)

    new_w = random.randint(min_w, width-1)
    new_h = floor(area/new_w)

    x1_ = random.randint(x1, x1+(width-new_w))
    x2_ = x1_ + new_w

    y1_ = random.randint(y1, y1+(height-new_h))
    y2_ = y1_ + new_h

    return x1_,y1_,x2_,y2_

def editVideo(path):
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (frame_width,frame_height))

cap = cv2.VideoCapture("hmdb51_org/drink/21_drink_u_nm_np1_fr_goo_9.avi")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('out1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (frame_width,frame_height))

with open("bb_file/HMDB51/21_drink_u_nm_np1_fr_goo_9.bb", "r") as f:
    bb = f.read().splitlines()

for i in range(len(bb)):
    bb[i] = bb[i].split(" ")
    for j in range(len(bb[i])):
        bb[i][j] = float(bb[i][j])

i = 0 #record processed frame number
flag = 0 # flag for the first frame with the object
ratio = 1/8
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        if len(bb[i]) > 1:
            if flag == 0:
                start = (round(bb[i][1]), round(bb[i][2]))
                end = (round(bb[i][3]), round(bb[i][4]))

                x1,y1,x2,y2 = randomShade(start[0], start[1], end[0], end[1])
                print(x1,y1,x2,y2)
                frame = cv2.rectangle(frame, (x1, y1),
                    (x2, y2), (255,0,0), -1)
                flag = 1
            else: 
                frame = cv2.rectangle(frame, (x1, y1),
                    (x2, y2), (255,0,0), -1)
        out.write(frame)
        cv2.imshow('frame', frame)
        i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()


files = get_files("hmdb51_org")
print(files.keys())