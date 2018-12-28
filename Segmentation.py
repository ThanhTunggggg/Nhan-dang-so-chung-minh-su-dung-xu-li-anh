import numpy as np
import cv2
import imutils
from OCR import *

def check_contours(boundaries, img_orig, img_preproc, _check) :

    """
    Tim contours
    """
    (_, cntrs, _) = cv2.findContours(img_preproc.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_width = boundaries[0]
    upper_width = boundaries[1]
    lower_height = boundaries[2]
    upper_height = boundaries[3]

    if _check is True :
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:5]
    else :
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :

        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :

            x_cntr_list.append(intX)
            target_contours.append(cntr)

            
            if _check is True :
                img_res = img_orig[intY:intY+intHeight, intX:intX+intWidth, :]

            
            if _check is False :

                char_copy = np.ones((44, 24))
                char = img_orig[intY:intY+intHeight, intX:intX+intWidth]
                char = cv2.resize(char, (20, 40))

                
                char = cv2.subtract(255, char)

                # Resize the image to 24x44 with black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0

                img_res.append(char_copy)

    if _check is not True:
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        target_contours_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])
            target_contours_copy.append(target_contours[idx])
        img_res = img_res_copy
        target_contours = target_contours_copy

    return target_contours, img_res

 
def segment_characters(image, cmnd) :
    """
    Tach cac so
    """
    # Step 1: Aligned anh
    img = cv2.resize(image, (150, 75))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('crop_b', img_binary)

    #Dao bit va open anh
    img_binary = cv2.bitwise_not(img_binary)
    rows, cols = img_gray_lp.shape
    img_binary_lp = cv2.erode(img_binary, (2, 2))
    img_binary_lp = cv2.dilate(img_binary_lp, (2, 2))
    
    horizontal = img_binary_lp
    
    #masked_img = cv2.bitwise_and(img_gray, img_gray, mask=img_binary_lp)
    #masked_img_inv = cv2.bitwise_not(masked_img)
    #cv2.imshow("masked img", masked_img_inv)
    #cv2.imwrite("result2.jpg", masked_img_inv)
    
    horizontalsize = int(cols / 30)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1),iterations=5)
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1),iterations=5)
    #cv2.imshow("horizontal", horizontal)

    #mask = cv2.morphologyEx(img_binary_lp, cv2.MORPH_CLOSE, (-1, -1), iterations=10)

    #mask = cv2.dilate(img_binary_lp, None, iterations=5)
    #mask = cv2.erode(mask, None, iterations=5)
    #cv2.imshow('mask', horizontal)

    cnts = cv2.findContours(horizontal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
        cnt_max = max(cnts, key=cv2.contourArea)
        #x, y, wm, hm = cv2.boundingRect(cnt_max)
        center, (w, h), angle = rotrect = cv2.minAreaRect(cnt_max)
        #box = cv2.boxPoints(rotrect)
        #box = np.int0(box)
    
        #warp = four_point_transform(image, box)#np.array([(x-int(w/2), y - int(h/2)), (x-int(w/2), y+int(h/2)), (x+int(w/2), y -int(h/2)), (x+int(w/2), y + int(h/2))]))
        #cv2.imshow("rr0", warp)
        #text = read_ID(warp)


        print("angle :", angle)
        if w >= h:
            img_binary = imutils.rotate(img_binary, angle)
        else:
            img_binary = imutils.rotate(img_binary, angle+90)

    if cmnd:
        vertical = img_binary
        verticalsize = int(rows / 30)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (verticalsize, 1))
        vertical = cv2.erode(vertical, verticalStructure, (-1, -1), iterations=1)
        vertical = cv2.dilate(vertical, verticalStructure, (-1, -1), iterations=1)
        img_binary = vertical
        #cv2.imshow("vertical", vertical)


    #Step 2: Tach tung so
    LP_WIDTH = img_binary.shape[0]
    LP_HEIGHT = img_binary.shape[1]
    #cv2.imshow('crop_a', img_binary)

    # Tao duong bao trang
    #img_binary[0:3,:] = 255
    #img_binary[:,0:3] = 255
    #img_binary[72:75,:] = 255
    #img_binary[:,147:150] = 255

    boundaries_crop = [LP_WIDTH/15,
                       LP_WIDTH/3,
                       LP_HEIGHT/3,
                       LP_HEIGHT]

    img_binary = cv2.bitwise_not(img_binary)
    char_contours, char_list = check_contours(boundaries_crop, img_binary, img_binary, False)
    print(len(char_contours))

    """
    if len(char_contours) == 0 :
            invert_img = np.invert(img_binary_lp)
            char_contours, char_list = check_contours(boundaries_crop, img_binary, invert_img, False)
    """

    full_character = []
    if len(char_contours) == 12 or True:
        full_character = char_list

    return full_character