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

for i in range(len(frameList)-1):
	# cv2.imshow("Frame {}".format(i),frameList[i])
	previousFrame= frameList[i]
	currFrame= frameList[i+1]

	previousFrameFiltered= cv2.medianBlur(previousFrame,7)
	currFrameFiltered= cv2.medianBlur(currFrame,7)

	previousFrameGray= cv2.cvtColor(previousFrameFiltered,cv2.COLOR_BGR2GRAY)
	currFrameGray= cv2.cvtColor(currFrameFiltered,cv2.COLOR_BGR2GRAY)

	frame_diff_curr_previous= cv2.absdiff(previousFrameGray,currFrameGray)

	bitwiseAndFramDiff= cv2.bitwise_and(frame_diff_curr_previous,previousFrameGray)

# startTimeImageDifferencing= time.time()
# (score, image_diff) =compare_ssim(image3Gray,image4Gray,full=True)
# image_diff= (image_diff*255).astype("uint8")
# print("--- %s seconds ---" % (time.time() - startTimeImageDifferencing)) 

	threshFrameDifferencing = cv2.threshold(bitwiseAndFramDiff, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# filteredThreshFrameDifferencing= cv2.medianBlur(threshFrameDifferencing,7)

# kernelDilation = np.ones((7,7), np.uint8)
# kernelErosion = np.ones((3,3), np.uint8)
# img_dilation = cv2.dilate(filteredThreshFrameDifferencing, kernelDilation, iterations=1) 
# img_erosion = cv2.erode(img_dilation, kernelErosion, iterations=2) 

# threshImageDifferencing = cv2.threshold(image_diff, 0, 255,
# 	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# cnts = cv2.findContours(threshFrameDifferencing.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# for c in cnts:
# 	# compute the bounding box of the contour and then draw the
# 	# bounding box on both input images to represent where the two
# 	# images differ
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	cv2.rectangle(threshFrameDifferencing, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# cv2.rectangle(image4, (x, y), (x + w, y + h), (0, 0, 255), 2)

# cv2.imshow('Frame 1',image3)
# cv2.imshow('Frame 2',image4)
# cv2.imshow('Frame Differencing',frame_diff)
# cv2.imshow('Image Differencing',image_diff)
	print("--- %s seconds ---" % (time.time() - startTimeFrameDifferencing)) 

	cv2.imshow('Thresholded image Frame differencing',threshFrameDifferencing)
# cv2.imshow('Filtered Thresholded Frame Differencing',filteredThreshFrameDifferencing)
# cv2.imshow('Erosion',img_erosion)
# cv2.imshow('Dilation',img_dilation)
# cv2.imshow('Thresholded image Image differencing',threshImageDifferencing)

	cv2.waitKey(0)