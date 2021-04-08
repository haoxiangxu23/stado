from get_file import *
from bb_box import *
from add_occlusion import *
from overlay import *
from get_file_nums import *
import numpy as np
import random
from math import floor
from pathlib import Path
import os
import scipy.io

## for JHMDB dataset


hmdb_path = "hmdb51_org"
jhmdb_bb_path = "MOC-Detector/data/JHMDB/puppet_mask"
jhmdb_rgb_path = "MOC-Detector/data/JHMDB/Frames"
jhmdb_rgb_path_new = "MOC-Detector/data/JHMDB_new/Frames"
jhmdb_flow_path = "MOC-Detector/data/JHMDB/FlowBrox04"
jhmdb_flow_path_new = "MOC-Detector/data/JHMDB_new/FlowBrox04"
# jhmdb_bb_path = "../data/JHMDB/puppet_mask"
# jhmdb_rgb_path = "../data/JHMDB/Frames"
# jhmdb_rgb_path_new = "../data/JHMDB_new/Frames"
# jhmdb_flow_path = "../data/JHMDB/FlowBrox04"
# jhmdb_flow_path_new = "../data/JHMDB_new/FlowBrox04"

if not os.path.exists("MOC-Detector/data/JHMDB_new"):
    os.mkdir("MOC-Detector/data/JHMDB_new")
# if not os.path.exists("../data/JHMDB_new"):
#     os.mkdir("../data/JHMDB_new")
if not os.path.exists(jhmdb_rgb_path_new):
    os.mkdir(jhmdb_rgb_path_new)
if not os.path.exists(jhmdb_flow_path_new):
    os.mkdir(jhmdb_flow_path_new)

cat_files = get_catagory_and_files(jhmdb_rgb_path)

all_paths = []
## get puppet_mask
for cat in cat_files:
    if not os.path.exists(os.path.join(jhmdb_rgb_path_new,cat)):
        os.mkdir(os.path.join(jhmdb_rgb_path_new,cat))
    if not os.path.exists(os.path.join(jhmdb_flow_path_new,cat)):
        os.mkdir(os.path.join(jhmdb_flow_path_new,cat))
    
    files = cat_files[cat]
    for f in files:
        path = os.path.join(jhmdb_bb_path, cat, f, "puppet_mask.mat")
        if os.path.isfile(path):
            all_paths.append([cat, f, path])

## get bbox
for i, (cat, f, bb_path) in enumerate(all_paths):
    print(i)
    rgb_path = os.path.join(jhmdb_rgb_path, cat, f)
    rgb_path_new = os.path.join(jhmdb_rgb_path_new, cat, f)
    if not os.path.exists(rgb_path_new):
        os.mkdir(rgb_path_new)
    flow_path = os.path.join(jhmdb_flow_path, cat, f)
    flow_path_new = os.path.join(jhmdb_flow_path_new, cat, f)
    if not os.path.exists(flow_path_new):
        os.mkdir(flow_path_new)
    bb = get_bb_box(bb_path)

    try:
        rect, loc = randomShade(bb,4)
    except:
        print(bb,rgb_path)
        break

    occ = str(random.randint(1,25)).zfill(5)
    occ_path = os.path.join("MOC-Detector/data/low_res_occ", occ+".png")
    # occ_path = os.path.join("../data/low_res_occ", occ+".png")

    num = get_number(rgb_path)

    flag = 0

    for i in range(1,num+1):
        img_name = str(i).zfill(5)
        rgb_img_path = os.path.join(rgb_path, img_name+".png")
        rgb_img_path_new = os.path.join(rgb_path_new, img_name+".png")
        flow_img_path = os.path.join(flow_path, img_name+".jpg")
        flow_img_path_new = os.path.join(flow_path_new, img_name+".jpg")

        rgb_img = cv2.imread(rgb_img_path)
        flow_img = cv2.imread(flow_img_path)
        overlay =  cv2.imread(occ_path,-1)

        

        try:
            cv2.imwrite(rgb_img_path_new, overlay_transparent(rgb_img, overlay, loc[0], loc[1], (loc[2],loc[3])))
        except:
            print(rgb_img_path, occ_path,loc[0], loc[1], (loc[2],loc[3]))
            flag = 1
            break
        try:
            cv2.imwrite(flow_img_path_new, overlay_flow(flow_img, overlay, loc[0], loc[1], (loc[2],loc[3])))
        except:
            print(flow_img_path, occ_path,loc[0], loc[1], (loc[2],loc[3]))
            flag = 1
            break
    if flag == 1:
        break
        
    
# print(os.path.join(flow_path,"00001.jpg"))
# img1 = cv2.imread(os.path.join(flow_path,"00001.jpg"))
# rect, loc = randomShade(bb)
# img2 = cv2.imread("MOC-Detector/data/low_res_occ/00022.png",-1)
# # img2 = cv2.resize(img2, (loc[2],loc[3]), interpolation = cv2.INTER_AREA) 

# # cv2.imshow('image',overlay_transparent(img1, img2, loc[0], loc[1], (loc[2],loc[3])))
# # cv2.waitKey(0)
# # cv2.rectangle(img, (rect[0],rect[1]), (rect[2],rect[3]), (0,255,0), 2)

# # overlay = cv2.add(img[loc[1]:loc[1]+loc[3], loc[0]:loc[0]+loc[2]],overlay)
# # img[loc[1]:loc[1]+loc[3], loc[0]:loc[0]+loc[2]] = overlay
# # print(img1)
# overlay_flow(img1, img2, loc[0], loc[1], (loc[2],loc[3]))

# cv2.imshow("image", img1) 
# cv2.waitKey(0)
# print(randomShade(bb[1]))