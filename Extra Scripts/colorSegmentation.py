import numpy as np
import cv2
import imutils
import time
from PIL import Image
import glob

startTimeFrameDifferencing= time.time()

filenames = glob.glob("Dataset/*.jpg")
filenames.sort()
frameList = [cv2.imread(frame) for frame in filenames]

for i in range(7):
    # cv2.imshow("Frame {}".format(i),frameList[i])
    currFrame= frameList[i+1]
    currFrameFiltered= cv2.medianBlur(currFrame,7)
    currFrameHSV=cv2.cvtColor(currFrameFiltered,cv2.COLOR_BGR2HSV)
    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)
    mask_white = cv2.inRange(currFrameHSV, light_white, dark_white)
    currFrameWhite= cv2.bitwise_and(currFrameFiltered,currFrameFiltered,mask=mask_white)

    cv2.imshow('White parts',currFrameWhite)
    cv2.waitKey(0)

