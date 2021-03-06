import glob
import time
import cv2
import math
from Modules.foregroundExtractionD5 import readyFrame, frameDifferencing, morphologicalOperations, natural_sort, convert480p
from Modules.ballDetectionRes import findContours, sizeDetection, playerProximityDetection, courtBoundaryDetection

startTimeReadingFrames = time.time()
datasetName = "Dataset5"
if (datasetName == "Dataset1"):
    startFrameDataset = 65
    endFrameDataset = 560
elif (datasetName == "Dataset2"):
    startFrameDataset = 35
    endFrameDataset = 215
elif (datasetName == "Dataset3"):
    startFrameDataset = 10
    endFrameDataset = 140
elif (datasetName == "Dataset4"):
    startFrameDataset = 1
    endFrameDataset = 330
elif (datasetName == "Dataset5"):
    startFrameDataset = 1
    endFrameDataset = 200
dictFrameNumberscX = {}
dictFrameNumberscY = {}
ballCandidatesPreviousFrame = list()
# Creating Video Object
cap = cv2.VideoCapture('DatasetVideos/'+datasetName+'.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, startFrameDataset)
endTimeReadingFrames = time.time()
print("Reading Frames--- %s seconds ---" %
      (endTimeReadingFrames - startTimeReadingFrames))

startTimeForeGroundExtraction = time.time()
# Parsing through the frames

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

    # Readying the frames
    previousFrameGray, currFrameGray, nextFrameGray = readyFrame(
        previousFrame, currFrame, nextFrame)

    # Performing frame differencing
    threshFrameDifferencing = frameDifferencing(
        previousFrameGray, currFrameGray, nextFrameGray)

    # Performing morphological operations
    final_image = morphologicalOperations(threshFrameDifferencing, 6, 4)

    startTimeBlurringBinary = time.time()
    # Blurring the binary image to get smooth shapes of objects
    # final_image = cv2.medianBlur(img_erosion, 7)
    # endTimeBlurringBinary = time.time()
    # print("Final Blur--- %s seconds ---" %
    #       (endTimeBlurringBinary - startTimeBlurringBinary))

    endTimeForegroundExtrction=time.time()
    print("Foreground Extraction--- %s seconds ---" % (endTimeForegroundExtrction - startTimeForeGroundExtraction))

    # final_image_copy = final_image.copy()

    contours, hier = findContours(final_image)

    ballCandidates, playerCadidates, incompletePlayerCandidates = sizeDetection(contours, currFrame,i)
    
    ballCandidates, playerCadidates, incompletePlayerCandidates = courtBoundaryDetection(datasetName,ballCandidates,playerCadidates,incompletePlayerCandidates,currFrame)

    ballCandidatesFiltered = playerProximityDetection(ballCandidates, playerCadidates, incompletePlayerCandidates, currFrame)

    if (len(ballCandidatesFiltered) > 0):
        for cand in ballCandidatesFiltered:
            cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
            cv2.putText(currFrame, str(cand[0])+","+str(cand[1]), (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow('Candidate image', currFrame)
    else:
        cv2.imshow('Candidate image', currFrame)
        

    # for cand in ballCandidatesFiltered:
    #     if cand is None:
    #         cv2.imshow('Candidate image', currFrame)
    #     else:
    #         cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
    #         cv2.imshow('Candidate image', currFrame)
    ballCandidatesFilteredProximity = list()

    # if len(ballCandidatesPreviousFrame) > 0:
    #     for cand in ballCandidatesFiltered:
    #         ballCandFlag = False
    #         for prevCand in ballCandidatesPreviousFrame:
    #             dist = math.sqrt(math.pow((cand[0] - prevCand[0]), 2) + math.pow((cand[1] - prevCand[1]), 2))
    #             if dist > 2 and dist < 70:
    #                 ballCandFlag = True
    #             else:
    #                 continue
    #         if ballCandFlag is True:
    #             ballCandidatesFilteredProximity.append(cand)
    #             cv2.putText(currFrame, "Ball Candidate", (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #             cv2.drawContours(currFrame, [cand[3]], -1, (0, 255, 0), 2)
    #         else:
    #             cv2.putText(currFrame, "Not Ball Candidate", (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #             cv2.drawContours(currFrame, [cand[3]], -1, (0, 255, 0), 2)
    #     ballCandidatesPreviousFrame = ballCandidatesFilteredProximity.copy()
    #     cv2.imshow('Candidate image', currFrame)
    # else:
    #     for cand in ballCandidatesFiltered:
    #         cv2.drawContours(currFrame, [cand[3]], -1, (0, 255, 0), 2)
    #         cv2.putText(currFrame, "Ball Candidate", (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #     ballCandidatesPreviousFrame = ballCandidatesFiltered.copy()
    #     cv2.imshow('Candidate image', currFrame)
    
    i += 1  # increments the loop

    # Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
    k = cv2.waitKey(0)
    if k == 27:
        break
    elif k == 32:
        i -= 2
    else:
        continue
