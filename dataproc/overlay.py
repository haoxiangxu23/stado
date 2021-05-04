import cv2
import numpy as np

def overlay_transparent(background_img, img_to_overlay_t, bbox, x, y, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()
    
    # ground truth background
    b_h, b_w, _ = bg_img.shape
    gt = np.full((b_h, b_w, 3), 0, dtype = np.uint8) # black
    gt_with_bbox = np.full((b_h, b_w, 3), 0, dtype = np.uint8)
    gt_with_bbox[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB 
    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))

    # Apply some simple filtering to remove edge noise
    ret, mask = cv2.threshold(img_to_overlay_t[:, :, 3], 0, 255, cv2.THRESH_BINARY)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]

    # this is for ground truth
    occ_white = np.full((h, w, 3), 255, dtype = np.uint8)

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

    # ground truth
    occ_mask = cv2.bitwise_and(occ_white.copy(),occ_white.copy(),mask = cv2.bitwise_not(mask))
    gt[y:y+h, x:x+w] = occ_mask

    # Update the original image with our new ROI
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

    gt_with_bbox[y:y+h, x:x+w] = occ_mask

    # cv2.imshow("image", bg_img) 
    # cv2.imshow("image1", gt) 
    # cv2.imshow("image2", gt_with_bbox) 
    # cv2.waitKey(0)

    return bg_img, gt_with_bbox


def overlay_flow(background_img, img_to_overlay_t, x, y, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()
    flow_img = np.full((bg_img.shape[0], bg_img.shape[1], 3), (137, 131, 136), np.uint8)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB 
    b,g,r,a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b,g,r))

    # Apply some simple filtering to remove edge noise
    ret, mask = cv2.threshold(img_to_overlay_t[:, :, 3], 0, 255, cv2.THRESH_BINARY)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]
    flow_roi = flow_img[y:y+h, x:x+w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(flow_roi.copy(), flow_roi.copy(), mask=mask)

    # # Update the original image with our new ROI
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
    
    return bg_img
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    # print(mask)
    # print(cv2.bitwise_not(mask))
