import cv2
import imutils
import numpy as np

input_size = 48

def find_board(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bfilter= cv2.bilateralFilter(gray,13,20,20)
    edged= cv2.Canny(bfilter,30,180)
    keypoints= cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours= imutils.grab_contours(keypoints)
    contours= sorted(contours,key=cv2.contourArea,reverse=True)[:15]
    location=None

    for contour in contours:
        approx= cv2.approxPolyDP(contour,15,True)
        if len(approx)==4:
            location=approx
            break
    result= get_perspective(img, location)
    return result,location

def get_perspective(img,location,height=900,width=900):
    pts1=np.float32([location[0],location[3],location[1],location[2]])
    pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix= cv2.getPerspectiveTransform(pts1,pts2)
    result= cv2.warpPerspective(img,matrix,(width,height))
    return result

def split_boxes(board):
    rows=np.vsplit(board,9)
    boxes=[]
    for r in rows:
        cols=np.hsplit(r,9)
        for box in cols:
            box= cv2.resize(box,(input_size,input_size))/255.0
            boxes.append(box)
    return boxes