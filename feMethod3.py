import numpy as np
import cv2
import imutils
import time
from PIL import Image
import glob
import math
from matplotlib import pyplot as plt

startTimeFrameDifferencing= time.time()

filenames = glob.glob("Dataset/*.jpg")
filenames.sort()
frameList = [cv2.imread(frame) for frame in filenames]

for i in range(len(frameList)-2):
	# cv2.imshow("Frame {}".format(i),frameList[i])
	previousFrame= frameList[i]
	currFrame= frameList[i+1]
	nextFrame= frameList[i+2]
	previousFrameFiltered= cv2.medianBlur(previousFrame,7)

	currFrameFiltered= cv2.medianBlur(currFrame,7)
	nextFrameFiltered= cv2.medianBlur(nextFrame,7)

	previousFrameGray= cv2.cvtColor(previousFrameFiltered,cv2.COLOR_BGR2GRAY)
	currFrameGray= cv2.cvtColor(currFrameFiltered,cv2.COLOR_BGR2GRAY)
	nextFrameGray= cv2.cvtColor(nextFrameFiltered,cv2.COLOR_BGR2GRAY)

	delta_plus = cv2.absdiff(currFrameGray, previousFrameGray)
	delta_0 = cv2.absdiff(nextFrameGray, previousFrameGray)
	delta_minus = cv2.absdiff(currFrameGray,nextFrameGray)

	dbp = cv2.threshold(delta_plus, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	dbm = cv2.threshold(delta_minus, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	db0 = cv2.threshold(delta_0, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	detect = cv2.bitwise_and(cv2.bitwise_and(dbp, dbm), cv2.bitwise_not(db0))

	# nd = cv2.bitwise_not(detect)

	num, labels, stats, centroids = cv2.connectedComponentsWithStats(detect, ltype=cv2.CV_16U)
	font = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 0.6
	fontColor = 127
	lineType = 1

	min_area = 50


	d = detect.copy()
	candidates = list()
	for stat in stats:
		area = stat[cv2.CC_STAT_AREA]
		if area < min_area:
			continue # Skip small objects (noise)
			
		lt = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
		rb = (lt[0] + stat[cv2.CC_STAT_WIDTH], lt[1] + stat[cv2.CC_STAT_HEIGHT])                                                
		bottomLeftCornerOfText = (lt[0], lt[1] - 15)

		candidates.append((lt, rb, area))
		cv2.rectangle(d, lt, rb, fontColor, lineType)

		cv2.putText(d, "{}: {:.0f}".format(len(candidates), stat[cv2.CC_STAT_AREA]),bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

	cv2.imshow('image',d)
	print("Found {} components".format(num - 1))
	k = cv2.waitKey(0)
	if k==27:
		    break
	else: 
		    continue
