import numpy as np
import cv2
# import imutils
import time
import re
# from matplotlib import pyplot as plt

# Function to sort a list in natural manner
def natural_sort(l): 
	'Function to sort a list in natural manner'
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

def readyFrame(previousFrame,currFrame,nextFrame):
	'Converts the frame to Gray Scale and filters it'
	startTimeReadyingFrame = time.time()
	# Converting the frames to grayscale(for thresholding)
	previousFrameGray= cv2.cvtColor(previousFrame,cv2.COLOR_BGR2GRAY)
	currFrameGray= cv2.cvtColor(currFrame,cv2.COLOR_BGR2GRAY)
	nextFrameGray= cv2.cvtColor(nextFrame,cv2.COLOR_BGR2GRAY)
	# Filtering the frames with a Median filter
	# previousFrameGray= cv2.medianBlur(previousFrameGray,7)
	# currFrameGray= cv2.medianBlur(currFrameGray,7)
	# nextFrameGray= cv2.medianBlur(nextFrameGray,7)
	previousFrameGray=cv2.GaussianBlur(previousFrameGray,(7,7),0)
	currFrameGray=cv2.GaussianBlur(currFrameGray,(7,7),0)
	nextFrameGray=cv2.GaussianBlur(nextFrameGray,(7,7),0)
	endTimeReadyingFrame= time.time()
	print("Frame Ready--- %s seconds ---" % (endTimeReadyingFrame - startTimeReadyingFrame))
	return previousFrameGray,currFrameGray,nextFrameGray

def frameDifferencing(previousFrameGray,currFrameGray,nextFrameGray):
	'Performs Frame Differencing bw pre and curr & curr and next, combines with AND operation and thresholds the difference image'
	startTimeFrameDifferencing=time.time()
	# Performing frame differencing
	frame_diff_curr_previous= cv2.absdiff(previousFrameGray,currFrameGray)
	frame_diff_next_curr= cv2.absdiff(currFrameGray,nextFrameGray)
	# Combining the differential images using an AND operation
	bitwiseAndFramDiff= cv2.bitwise_and(frame_diff_curr_previous,frame_diff_next_curr)
	# plt.hist(bitwiseAndFramDiff.ravel(),256,[0,256])
	# plt.show()
	# Thresholding the combined image using Otsu thresholding
	threshFrameDifferencing = cv2.threshold(bitwiseAndFramDiff, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	endTimeFrameDifferencing=time.time()
	print("Frame Differencing--- %s seconds ---" % (endTimeFrameDifferencing - startTimeFrameDifferencing))
	return threshFrameDifferencing

def morphologicalOperations(threshFrameDifferencing, dilationIterations, erosionIterations):
	'Performs Dilation Followed by Erosion using a 7x7 Kernel'
	startTimeMorphologicalOperations=time.time()
	# Defining 7x7 kernels for Dilation & Erosion
	# kernelDilation =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
	# kernelErosion=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	kernelDilation = np.ones((7,7), np.uint8)
	kernelErosion = np.ones((3,3), np.uint8)
	# Performing morphological dilation to join disconnected components in the binary image
	img_dilation = cv2.dilate(threshFrameDifferencing, kernelDilation, iterations=dilationIterations)
	# Performing morphological erosion to reduce thickness of objects and remove small white noise if present
	img_erosion = cv2.erode(img_dilation, kernelErosion, iterations=erosionIterations)
	endTimeMorphologicalOperations=time.time()
	print("Morphological Operations--- %s seconds ---" % (endTimeMorphologicalOperations - startTimeMorphologicalOperations))
	return img_erosion