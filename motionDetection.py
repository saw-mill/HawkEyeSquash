import glob
import time
import cv2
import math
from Modules.foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort
from Modules.ballDetection import findContours, sizeDetection, playerProximityDetection

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

    ballCandidates, playerCadidates, incompletePlayerCandidates = sizeDetection(contours, currFrame,i)
    
    ballCandidatesFiltered = playerProximityDetection(ballCandidates, playerCadidates, incompletePlayerCandidates, currFrame)

    # for cand in ballCandidatesFiltered:
    #     if cand is None:
    #         cv2.imshow('Candidate image', currFrame)
    #     else:
    #         cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
    #         cv2.imshow('Candidate image', currFrame)
    ballCandidatesFilteredProximity = list()

    if len(ballCandidatesPreviousFrame) > 0:
        for cand in ballCandidatesFiltered:
            ballCandFlag = False
            for prevCand in ballCandidatesPreviousFrame:
                dist = math.sqrt(math.pow((cand[0] - prevCand[0]), 2) + math.pow((cand[1] - prevCand[1]), 2))
                if dist > 2 and dist < 70:
                    ballCandFlag = True
                else:
                    continue
            if ballCandFlag is True:
                ballCandidatesFilteredProximity.append(cand)
                cv2.putText(currFrame, "Ball Candidate", (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.drawContours(currFrame, [cand[3]], -1, (0, 255, 0), 2)
            else:
                cv2.putText(currFrame, "Not Ball Candidate", (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.drawContours(currFrame, [cand[3]], -1, (0, 255, 0), 2)
        ballCandidatesPreviousFrame = ballCandidatesFilteredProximity.copy()
        cv2.imshow('Candidate image', currFrame)
    else:
        for cand in ballCandidatesFiltered:
            cv2.drawContours(currFrame, [cand[3]], -1, (0, 255, 0), 2)
            cv2.putText(currFrame, "Ball Candidate", (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        ballCandidatesPreviousFrame = ballCandidatesFiltered.copy()
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
