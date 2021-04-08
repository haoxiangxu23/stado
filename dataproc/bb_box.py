import scipy.io
import cv2
import numpy as np

# mat = scipy.io.loadmat('puppet_mask/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0/puppet_mask.mat')

# array = np.array(mat["part_mask"][:,:,0])

# contours,_ = cv2.findContours(array.copy() ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # not copying here will throw an error
# rect = cv2.boundingRect(contours[0]) # basically you can feed this rect into your classifier
# x,y,w,h = rect # a - angle
# print(rect)

def get_bb_box(path):
    mat = scipy.io.loadmat(path)
    array = np.array(mat["part_mask"])
    out = []
    for i in range(array.shape[2]):
        contours,_ = cv2.findContours(array[:,:,i].copy() ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.boundingRect(contours[0])
        out.append([rect[2]*rect[3],rect])
    # print(w,h)
    return max(out)[1]

# print(get_bb_box("MOC-Detector/data/JHMDB/puppet_mask/jump/Sam_Cooksey_Goalkeeper_Training_jump_f_cm_np1_ri_bad_1/puppet_mask.mat"))