import glob
import time
import cv2
from foregroundExtraction import readyFrame,frameDifferencing,morphologicalOperations,natural_sort
from ballDetection import filterSize,drawRectangle

startTimeReadingFrames= time.time()
# Location of dataset
filenames = glob.glob("Dataset1/*.jpg")
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

    # Readying the frames
	previousFrameGray, currFrameGray, nextFrameGray = readyFrame(previousFrame,currFrame,nextFrame)

	# Performing frame differencing
	threshFrameDifferencing = frameDifferencing(previousFrameGray,currFrameGray,nextFrameGray)

	# Performing morphological operations
	img_erosion = morphologicalOperations(threshFrameDifferencing,3,4)
	
	startTimeBlurringBinary=time.time()
	# Blurring the binary image to get smooth shapes of objects
	final_image = cv2.medianBlur(img_erosion,7)
	endTimeBlurringBinary=time.time()
	print("Final Blur--- %s seconds ---" % (endTimeBlurringBinary - startTimeBlurringBinary))

	endTimeForegroundExtrction=time.time()
	print("Foreground Extraction--- %s seconds ---" % (endTimeForegroundExtrction - startTimeForeGroundExtraction))

	cv2.imshow('Thresholded image Frame differencing',threshFrameDifferencing)
	# cv2.imshow('eroded',img_erosion)
	# cv2.imshow('dilated',img_dilation)
	cv2.imshow('Filtered final image',final_image)

	i+=1 # increments the loop

	# Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
	k = cv2.waitKey(0)
	if k==27:
		break
	elif k==32:
		i-=2
	else:
		continue