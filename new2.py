import glob
import time
import cv2
import math
from foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort
from ballDetection import findContours, sizeDetection, playerProximityDetection, regionDetection

startTimeReadingFrames = time.time()
# Location of dataset
filenames = glob.glob("Dataset1/*.jpg")

# Reading each frame and storing it in a list
frameList = [cv2.imread(frame) for frame in natural_sort(filenames)]
endTimeReadingFrames = time.time()
print("Reading Frames--- %s seconds ---" %
      (endTimeReadingFrames - startTimeReadingFrames))

startTimeForeGroundExtraction = time.time()
# Parsing through the frames
ballCandidatesPreviousFrame = list()
i = 0
while i < (len(frameList)-2):
    # cv2.imshow("Frame {}".format(i),frameList[i])

    # Storing three frames
    previousFrame = frameList[i]
    currFrame = frameList[i+1]
    nextFrame = frameList[i+2]

    # Readying the frames
    previousFrameGray, currFrameGray, nextFrameGray = readyFrame(
        previousFrame, currFrame, nextFrame)

    # Performing frame differencing
    threshFrameDifferencing = frameDifferencing(
        previousFrameGray, currFrameGray, nextFrameGray)

    # Performing morphological operations
    img_erosion = morphologicalOperations(threshFrameDifferencing, 4, 4)

    startTimeBlurringBinary = time.time()
    # Blurring the binary image to get smooth shapes of objects
    final_image = cv2.medianBlur(img_erosion, 7)
    endTimeBlurringBinary = time.time()
    print("Final Blur--- %s seconds ---" %
          (endTimeBlurringBinary - startTimeBlurringBinary))

    endTimeForegroundExtrction=time.time()
    print("Foreground Extraction--- %s seconds ---" % (endTimeForegroundExtrction - startTimeForeGroundExtraction))

    final_image_copy = final_image.copy()

    contours, hier = findContours(final_image_copy)

    ballCandidates, playerCadidates, incompletePlayerCandidates = sizeDetection(contours, currFrame)
    
    ballCandidatesFiltered = playerProximityDetection(ballCandidates, playerCadidates, incompletePlayerCandidates, currFrame)

    ballCandidatesFilteredProximity, ballCandidatesPreviousFrame =regionDetection(ballCandidatesFiltered,ballCandidatesPreviousFrame,currFrame)
    
    for cand in ballCandidatesFilteredProximity:
        if not cand:
            cv2.imshow('Candidate image', currFrame)
        else:
            cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
            cv2.imshow('Candidate image', currFrame)

    i += 1  # increments the loop

    # Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
    k = cv2.waitKey(0)
    if k == 27:
        break
    elif k == 32:
        i -= 2
    else:
        continue
