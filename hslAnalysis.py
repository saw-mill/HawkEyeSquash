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
	# previousFrame= frameList[i]
	currFrame= frameList[i+1]
	# nextFrame= frameList[i+2]

    # Readying the frames
	# previousFrameGray, currFrameGray, nextFrameGray = readyFrame(previousFrame,currFrame,nextFrame)
	currHsl= cv2.cvtColor(currFrame,cv2.COLOR_BGR2HLS)
	currHsv= cv2.cvtColor(currFrame,cv2.COLOR_BGR2HSV)
	Lchannel = currHsl[:,:,1]
	Vchannel = currHsv[:,:,1]
	Hchannel = currHsv[1,1,:]
	Schannel = currHsv[:,1,:]
	mask = cv2.inRange(Vchannel, 0, 250)
	res = cv2.bitwise_and(currFrame,currFrame, mask= mask)
	cv2.imshow('asd',currFrame)
	cv2.imshow('HSL image',Vchannel)
	# cv2.imshow('HSV image',Vchannel)

	i+=1 # increments the loop

	# Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
	k = cv2.waitKey(0)
	if k==27:
		break
	elif k==32:
		i-=2
	else:
		continue