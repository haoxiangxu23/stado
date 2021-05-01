from get_file import *
from bb_box import *
from add_occlusion import *
from overlay import *
from get_file_nums import *
import numpy as np
import random
import pickle
from math import floor
from pathlib import Path
import os
import scipy.io
import sys

# jhmdb_gt_path = "MOC-Detector/data/JHMDB/puppet_mask"
# jhmdb_rgb_path = "MOC-Detector/data/JHMDB/Frames"
# jhmdb_rgb_path_new = "MOC-Detector/data/JHMDB_new/Frames"
# jhmdb_flow_path = "MOC-Detector/data/JHMDB/FlowBrox04"
# jhmdb_flow_path_new = "MOC-Detector/data/JHMDB_new/FlowBrox04"
def run(n):
    # jhmdb_gt_path = "../data/JHMDB/JHMDB-GT.pkl"
    # jhmdb_rgb_path = "../data/JHMDB/Frames"
    # jhmdb_rgb_path_new = "../data/JHMDB_new/Frames"
    # jhmdb_flow_path = "../data/JHMDB/FlowBrox04"
    # jhmdb_flow_path_new = "../data/JHMDB_new/FlowBrox04"

    if n == 2:
        suffix = 50
    elif n == 3:
        suffix = 33
    elif n == 4:
        suffix = 25
    else:
        suffix = "new"

    jhmdb_gt_path = "../data/JHMDB_00/JHMDB-GT.pkl"
    jhmdb_data_path = "../data/JHMDB_{}".format(suffix)
    jhmdb_rgb_path = "../data/JHMDB_00/Frames"
    jhmdb_rgb_path_new = "../data/JHMDB_{}/Frames".format(suffix)
    jhmdb_flow_path = "../data/JHMDB_00/FlowBrox04"
    jhmdb_flow_path_new = "../data/JHMDB_{}/FlowBrox04".format(suffix)

    # if not os.path.exists("MOC-Detector/data/JHMDB_new"):
    #     os.mkdir("MOC-Detector/data/JHMDB_new")
    if not os.path.exists(jhmdb_data_path):
        os.mkdir(jhmdb_data_path)
    if not os.path.exists(jhmdb_rgb_path_new):
        os.mkdir(jhmdb_rgb_path_new)
    if not os.path.exists(jhmdb_flow_path_new):
        os.mkdir(jhmdb_flow_path_new)

    with open(jhmdb_gt_path, 'rb') as fid:
        gt_cache = pickle.load(fid, encoding='iso-8859-1')

    video_names = list(gt_cache["nframes"])
    cats = list(gt_cache["labels"])
    gt_cache["gt_occ"] = {}

    # create directory for each category
    for cat in cats:
        if not os.path.exists(os.path.join(jhmdb_rgb_path_new,cat)):    
            os.mkdir(os.path.join(jhmdb_rgb_path_new,cat))
        if not os.path.exists(os.path.join(jhmdb_flow_path_new,cat)):
            os.mkdir(os.path.join(jhmdb_flow_path_new,cat)) 

    # generate occlusion bbox    
    occlusion = get_bb_from_gt(gt_cache["gttubes"], n)

    # create directory for each video 
    for i, path in enumerate(video_names):
        print(i)
        rgb_path = os.path.join(jhmdb_rgb_path, path)
        rgb_path_new = os.path.join(jhmdb_rgb_path_new, path)
        if not os.path.exists(rgb_path_new):
            os.mkdir(rgb_path_new)
        flow_path = os.path.join(jhmdb_flow_path, path)
        flow_path_new = os.path.join(jhmdb_flow_path_new, path)
        if not os.path.exists(flow_path_new):
            os.mkdir(flow_path_new)

        (rect, loc), bbox = occlusion[path]

        occ = str(random.randint(1,25)).zfill(5)
        # occ_path = os.path.join("MOC-Detector/data/low_res_occ", occ+".png")
        occ_path = os.path.join("../data/low_res_occ", occ+".png")
        
        for i in range(1, gt_cache["nframes"][path]+1):
            img_name = str(i).zfill(5)
            rgb_img_path = os.path.join(rgb_path, img_name+".png")
            rgb_img_path_new = os.path.join(rgb_path_new, img_name+".png")
            flow_img_path = os.path.join(flow_path, img_name+".jpg")
            flow_img_path_new = os.path.join(flow_path_new, img_name+".jpg")

            rgb_img = cv2.imread(rgb_img_path)
            flow_img = cv2.imread(flow_img_path)
            overlay =  cv2.imread(occ_path,-1)

            img, mask = overlay_transparent(rgb_img, overlay, bbox, loc[0], loc[1], (loc[2],loc[3]))
            mask = mask[:,:,0]
            cv2.imwrite(rgb_img_path_new, img)
            cv2.imwrite(flow_img_path_new, overlay_flow(flow_img, overlay, loc[0], loc[1], (loc[2],loc[3])))
        gt_cache["gt_occ"][path] = mask
    pickle.dump(gt_cache, open("../data/JHMDB_{}/JHMDB-GT.pkl".format(suffix), "wb"))

def main():
    if len(sys.argv) == 1:
        print("creating default 25% occ ratio")
        run(4)
        print("creating default 33% occ ratio")
        run(3)
        print("creating default 25% occ ratio")
        run(2)
    else:
        print("creating default 1/{} occ ratio".format(sys.argv[1]))
        run(int(sys.argv[1]))


if __name__ == "__main__":
    main()
