import numpy as np
import cv2
import imutils
import time
from PIL import Image
import glob
import re

# Function to sort a list in natural manner
def natural_sort(l): 
	'Function to sort a list in natural manner'
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

startTimeReadingFrames= time.time()

# Location of dataset
filenames = glob.glob("Dataset/*.jpg")
# filenames = glob.glob("Testing/*.jpg")

# Reading each frame and storing it in a list
frameList = [cv2.imread(frame) for frame in natural_sort(filenames)]
endTimeReadingFrames= time.time()
print("Reading Frames--- %s seconds ---" % (endTimeReadingFrames - startTimeReadingFrames))

startTimeForeGroundExtraction= time.time()
# Parsing through the frames
i=0
while i < (len(frameList)-2):
	# cv2.imshow("Frame {}".format(i),frameList[i])
	
	# Storing three frames
	previousFrame= frameList[i]
	currFrame= frameList[i+1]
	nextFrame= frameList[i+2]

	# startTimeFilteringFrames=time.time()
	# # Filtering the frames with a Median filter
	# previousFrameFiltered= cv2.medianBlur(previousFrame,7)
	# currFrameFiltered= cv2.medianBlur(currFrame,7)
	# nextFrameFiltered= cv2.medianBlur(nextFrame,7)
	# endTimeFilteringFrames=time.time()
	# print("Filtering Frames--- %s seconds ---" % (endTimeFilteringFrames - startTimeFilteringFrames))

	startTimeConvertingFrames=time.time()
	# Converting the frames to grayscale(for thresholding)
	previousFrameGray= cv2.cvtColor(previousFrame,cv2.COLOR_BGR2GRAY)
	currFrameGray= cv2.cvtColor(currFrame,cv2.COLOR_BGR2GRAY)
	nextFrameGray= cv2.cvtColor(nextFrame,cv2.COLOR_BGR2GRAY)
	endTimeConvertingFrames=time.time()
	print("Converting to Grayscale--- %s seconds ---" % (endTimeConvertingFrames - startTimeConvertingFrames))
	# cv2.imshow('GreyScale versions',np.hstack([image3Gray,image4Gray]))

	startTimeFilteringFrames=time.time()
	# Filtering the frames with a Median filter
	previousFrameGray= cv2.medianBlur(previousFrameGray,7)
	currFrameGray= cv2.medianBlur(currFrameGray,7)
	nextFrameGray= cv2.medianBlur(nextFrameGray,7)
	endTimeFilteringFrames=time.time()
	print("Filtering Frames--- %s seconds ---" % (endTimeFilteringFrames - startTimeFilteringFrames))

	startTimeFrameDifferencing=time.time()
	# Performing frame differencing
	frame_diff_curr_previous= cv2.absdiff(previousFrameGray,currFrameGray)
	frame_diff_next_curr= cv2.absdiff(currFrameGray,nextFrameGray)
	endTimeFrameDifferencing=time.time()
	print("Frame Differencing--- %s seconds ---" % (endTimeFrameDifferencing - startTimeFrameDifferencing))

	startTimeAndFrames=time.time()
	# Combining the differential images using an AND operation
	bitwiseAndFramDiff= cv2.bitwise_and(frame_diff_curr_previous,frame_diff_next_curr)
	endTimeAndFrames=time.time()
	print("AND operation--- %s seconds ---" % (endTimeAndFrames - startTimeAndFrames))

	startTimeThresholding=time.time()
	# Thresholding the combined image using Otsu thresholding
	threshFrameDifferencing = cv2.threshold(bitwiseAndFramDiff, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	endTimeThresholding=time.time()
	print("Thresholding--- %s seconds ---" % (endTimeThresholding - startTimeThresholding))

	# Defining 7x7 kernels for Dilation & Erosion
	kernelDilation = np.ones((7,7), np.uint8)
	kernelErosion = np.ones((3,3), np.uint8)

	startTimeDilation=time.time()
	# Performing morphological dilation to join disconnected components in the binary image
	img_dilation = cv2.dilate(threshFrameDifferencing, kernelDilation, iterations=3)
	endTimeDilation=time.time()
	print("Dilation(iter=2)--- %s seconds ---" % (endTimeDilation - startTimeDilation))

	startTimeErosion=time.time()
	# Performing morphological erosion to reduce thickness of objects and remove small white noise if present
	img_erosion = cv2.erode(img_dilation, kernelErosion, iterations=4)
	endTimeErosion=time.time()
	print("Erosion(iter=2)--- %s seconds ---" % (endTimeErosion - startTimeErosion))

	startTimeBlurringBinary=time.time()
	# Blurring the binary image to get smooth shapes of objects
	final_image = cv2.medianBlur(img_erosion,7)
	endTimeBlurringBinary=time.time()
	print("Final Blur--- %s seconds ---" % (endTimeBlurringBinary - startTimeBlurringBinary))

	endTimeForegroundExtrction=time.time()
	print("Foreground Extraction--- %s seconds ---" % (endTimeForegroundExtrction - startTimeForeGroundExtraction))

	num, labels, stats, centroids = cv2.connectedComponentsWithStats(final_image, ltype=cv2.CV_16U)

	print("Number of Components: ",num,end='\n')
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

	# print("--- %s seconds ---" % (time.time() - startTimeFrameDifferencing))

	cv2.imshow('Thresholded image Frame differencing',threshFrameDifferencing)
	# cv2.imshow('eroded',img_erosion)
	# cv2.imshow('dilated',img_dilation)
	cv2.imshow('Filtered final image',final_image)
	# print("Found {} components".format(num - 1))

# cv2.imshow('Erosion',img_erosion)
# cv2.imshow('Dilation',img_dilation)
# cv2.imshow('Thresholded image Image differencing',threshImageDifferencing)

	i+=1 # increments the loop

	# Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
	k = cv2.waitKey(0)
	if k==27:
		break
	elif k==32:
		i-=2
	else:
		continue