import glob
import time
import cv2
import math
from Modules.foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort
from Modules.ballDetection import findContours, sizeDetection, playerProximityDetection, regionDetection, courtBoundaryDetection

startTimeReadingFrames = time.time()
datasetName= "Dataset2"
# Location of dataset
filenames = glob.glob(datasetName+"/*.jpg")

# Reading each frame and storing it in a list
frameList = [cv2.imread(frame) for frame in natural_sort(filenames)]
endTimeReadingFrames = time.time()
print("Reading Frames--- %s seconds ---" %
      (endTimeReadingFrames - startTimeReadingFrames))

# Parsing through the frames
ballCandidatesPreviousFrame = list()
i = 0
while i < (len(frameList)-2):
    # cv2.imshow("Frame {}".format(i),frameList[i])

    # Storing three frames
    previousFrame = frameList[i]
    currFrame = frameList[i+1]
    nextFrame = frameList[i+2]

    #
    # 
    # FOREGROUND EXTRACTION
    #
    #
    startTimeForeGroundExtraction = time.time()

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

    endTimeForegroundExtraction=time.time()
    print("Foreground Extraction--- %s seconds ---" % (endTimeForegroundExtraction - startTimeForeGroundExtraction))

    #
    #
    # BALL DETECTION
    #
    #
    startTimeBallDetection =time.time()

    # Making a copy of pre-processed image frame
    final_image_copy = final_image.copy()

    # Finding contours in the frame
    contours, hier = findContours(final_image_copy)

    # Separating candidates based on size
    ballCandidates, playerCadidates, incompletePlayerCandidates = sizeDetection(contours, currFrame,i)

    ballCandidatesFilteredBoundary = list()
    playerCadidatesFilteredBoundary = list()
    incompletePlayerCandidatesFilteredBoundary = list()
    for cand in ballCandidates:
            if (cand[0] < 145 or cand[0] > 1085):
                    continue
            else:
                    ballCandidatesFilteredBoundary.append(cand)
    for playercand in playerCadidates:
            if (playercand[0] < 145 or playercand[0] > 1085):
                    continue
            else:
                    playerCadidatesFilteredBoundary.append(playercand)
    for incompletecand in incompletePlayerCandidates:
            if (incompletecand[0] < 145 or incompletecand[0] > 1085):
                    continue
            else:
                    incompletePlayerCandidatesFilteredBoundary.append(incompletecand)
    
    print("Player Candidates: %d" % len(playerCadidatesFilteredBoundary))
    print("Incomplete Player Candidate: %d" % len(incompletePlayerCandidatesFilteredBoundary))

    for cand in ballCandidatesFilteredBoundary:
        if not cand:
            cv2.imshow('Candidate image', currFrame)
        else:
            cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
            cv2.putText(currFrame, str(cand[0])+","+str(cand[1]),(cand[0]+1, cand[1]+1),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
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