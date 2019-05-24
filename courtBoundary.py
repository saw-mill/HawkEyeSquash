import glob
import time
import cv2
import math
from Modules.foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort, convert480p
from Modules.ballDetection import findContours, sizeDetection, playerProximityDetection, regionDetection, courtBoundaryDetection

startTimeReadingFrames = time.time()
datasetName = "Dataset2"
if (datasetName == "Dataset1"):
    startFrameDataset = 65
    endFrameDataset = 560
elif (datasetName == "Dataset2"):
    startFrameDataset = 35
    endFrameDataset = 215
elif (datasetName == "Dataset3"):
    startFrameDataset = 10
    endFrameDataset = 140
dictFrameNumberscX = {}
dictFrameNumberscY = {}
ballCandidatesPreviousFrame = list()

# Creating Video Object
cap = cv2.VideoCapture('DatasetVideos/'+datasetName+'.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, startFrameDataset)
endTimeReadingFrames = time.time()
print("Reading Frames--- %s seconds ---" %
      (endTimeReadingFrames - startTimeReadingFrames))

i = 0
while (cap.isOpened()):
    print("######Start of Frame#####")
    if(i == 0): # If first frame read 3 frames
        ret1, previousFrame = cap.read()
        ret2, currFrame = cap.read()
        ret3, nextFrame = cap.read()
    else: # Read just the next frame from the 2nd frame onwards
        previousFrame = currFrame
        currFrame = nextFrame
        ret, nextFrame = cap.read()
    print("Frame Number {}".format(i + 1))

    # Changing from 720p to 480p
    previousFrame = convert480p(previousFrame)
    currFrame = convert480p(currFrame)
    nextFrame = convert480p(nextFrame)

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

    if(len(ballCandidatesFilteredBoundary)>0):
        for cand in ballCandidatesFilteredBoundary:
            cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
            cv2.putText(currFrame, str(cand[0])+","+str(cand[1]),(cand[0]+1, cand[1]+1),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow('Candidate image', currFrame)
    else:
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