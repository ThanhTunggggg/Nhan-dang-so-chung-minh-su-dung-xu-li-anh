#!/usr/bin/env python

# Python 2/3 compatibility
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv
#import pyocr
from PIL import Image
from glob import glob
from Segmentation import *

def angle_cos(p0, p1, p2):
    """
    Tim cosin giua 2 vector
    Input: 3 diem co toa do (x,y)
    Output: cosin
    """
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
def dim_rate(p0, p1, p2):
    """
    Tim ty le chieu dai/chieu rong
    """
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return np.dot(d1,d1)/np.dot(d2,d2)

def find_squares(img):
    #img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
    #for gray in cv.split(img):
    if True:
        for thrs in xrange(1, 255, 255):
            if thrs != 0:
                bin = cv.Canny(gray, 50, 255, apertureSize=5)
                cv.imshow('bin', bin)
                bin = cv.dilate(bin, None)
            
            #cv.imwrite("bin.jpg", bin)
            contours = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]
            #print(contours)
            #cv.drawContours(img, contours, 0, (255,0,0), 2)
            #cv.drawContours(img, contours, -1, (0, 255, 0), 3 )
            #cv.imshow("imgs", img)
            for cnt in contours:
                if (cv.contourArea(cnt) < 5000 or cv.contourArea(cnt) > 800000):
                    #print("TTTT")
                    continue
                cnt = cv.convexHull(cnt)
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.01*cnt_len, True)
                if len(cnt) == 4 and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    max_rate = np.max([dim_rate(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    #print(max_rate)
                    if  max_cos < 0.2 and max_rate < 4:
                        #print(max_cos)
                        squares.append(cnt)
                        #print(cnt)
    return squares

def order_points(pts):
    """
    Sap xet toa do 4 goc theo thu tu: top-left, top-right, bottom-right, bottom-left
	"""
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    """
    Bien doi phoi canh tu 4 diem
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def is_cmnd(warp):
    """
    Kiem tra la CMMN hay CCCD
    """
    h, w, c = img.shape
    print("shape of wrap", (h, w, c))
    color = ('b','g','r')
    print("rate", np.mean(img[:,int(0.95*w),1])/np.mean(img[:,int(0.95*w),0]))
    if np.mean(img[:,int(0.95*w),1])/np.mean(img[:,int(0.95*w),0]) > 1:
        return True
    return False


def crop_ID(warped, cmnd):
    """
    Crop ROI
    """
    h, w, _ = warp.shape
    #(dX, dY) = (int(w * 0.5), int(h * 0.45))
    #for CMND
    if cmnd: 
        crop = warp[int(h*0.22):int(h*0.33), int(w*0.49):int(w*0.86)]
    #for CCCD    
    else: 
        crop = warp[int(h*0.28):int(h*0.37), int(w*0.45):int(w*0.88)]
    return crop, h, w


def read_ID(crop):
    """
    Nhan dang so CMT Nhan dang so dung tesseract
    """

    tools = pyocr.get_available_tools()[0]
    text = tools.image_to_string(Image.fromarray(crop), builder=pyocr.builders.DigitBuilder())
    print("\nID Number: ", text)
    return text


if __name__ == '__main__':

    for fn in glob('datatest/*.jpg'):
        img = cv.imread(fn)
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        #cv.namedWindow('gray', cv.WINDOW_NORMAL)
        #cv.namedWindow('bin', cv.WINDOW_NORMAL)
        cv.namedWindow('warp', cv.WINDOW_NORMAL)
        cv.namedWindow('crop', cv.WINDOW_NORMAL)
        squares = find_squares(img)
        cmnd = False

        if len(squares):
        	#Take min square
            squares.sort(key = cv.contourArea)
            square = squares[0] 
            print("Squares: ",squares)
            print("\nSquare: ",square)

            #f len(squares) > 1 and cv.contourArea(squares[0])/cv.contourArea(squares[1]) > 0.7:
        	#   is_cmnd = True
            
            #Crop vung anh chua so cmnd/cccd
            warp = four_point_transform(img, square)

            #Check loai the
            cmnd = is_cmnd(warp)
            print("Loai the ", "CMMD" if cmnd else "CCCD")

            #cv.imwrite("rcc.jpg", warp)
            #warp_ = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
            #cmnd = True
            crop, h, w = crop_ID(warp, cmnd)
            #hist = cv.calcHist([crop],[0],None,[256],[0,256])

            characters = segment_characters(crop, cmnd)
            for i in range(len(characters)):
                #print(len(characters))
                #char = read_ID(characters[i])
                print("so thu", i, " :", char)
                cv.imshow('characters', characters[i])
                ch = cv.waitKey(0)
                if ch == 27:
                    break
                if ch == 32:
                   continue

            #Nhan dang
            #text = read_ID(crop)                     
            
            if cmnd:
            	cv.putText(warp, "".join(text), (int(w*0.49),int(h*0.2)), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                 fontScale=0.7, color=(255, 0, 0), thickness = 2)
            else:
            	cv.putText(warp, "".join(text), (int(w*0.46),int(h*0.25)), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                 fontScale=0.7, color=(255, 0, 0), thickness = 2)
            
            cv.drawContours(img, [square], -1, (0, 255, 0), 3)
            cv.circle(img, tuple(square[0]), 15, (255,0,0), -1)
            cv.circle(img, tuple(square[1]), 15, (255,0,0), -1)
            cv.circle(img, tuple(square[2]), 15, (255,0,0), -1)
            cv.circle(img, tuple(square[3]), 15, (255,0,0), -1)
            cv.imshow("warp", warp)
            cv.imshow("crop", crop)
            #cv.imwrite("crm.jpg", crop)

        #cv.drawContours( img, squares, -1, (0, 255, 0), 3 )
        cv.imshow('image', img)
        #cv.imshow('gray', gray)
        #cv.imwrite("warp.jpg", warp)
        #cv.imwrite("image.jpg", img)
        ch = cv.waitKey(0)
        if ch == 27:
            break
        if ch == 32:
            continue
                        
    cv.destroyAllWindows()
