import glob
import time
import cv2
import math
import numpy as np
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
meas=[]
pred = []
mp = np.array((2, 1), np.float32)  # measurement
tp = np.zeros((2, 1), np.float32)  # tracked / prediction
kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.09
kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00003
i = 0
while i < (len(frameList)-2):
    # cv2.imshow("Frame {}".format(i),frameList[i])

    # Storing three frames
    previousFrame = frameList[i]
    currFrame = frameList[i+1]
    nextFrame = frameList[i + 2]
    
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

    # Removing candidates outside the Court Boundary in Dataset2 
    if (datasetName == 'Dataset2'):
        ballCandidates, playerCadidates, incompletePlayerCandidates = courtBoundaryDetection(ballCandidates,playerCadidates,incompletePlayerCandidates,currFrame)
    
    # Removing Candidates that are close to the Players
    ballCandidatesFiltered = playerProximityDetection(ballCandidates, playerCadidates, incompletePlayerCandidates, currFrame)

    # Removing candidates that are not in their expected region after motion
    ballCandidatesFilteredProximity, ballCandidatesPreviousFrame =regionDetection(ballCandidatesFiltered,ballCandidatesPreviousFrame,currFrame)
    
    endTimeBallDetection = time.time()
    print("Ball Detection--- %s seconds ---" % (endTimeBallDetection - startTimeBallDetection))

    if (i + 1 == 1):
        x = ballCandidatesFilteredProximity[0][0]
        y = ballCandidatesFilteredProximity[0][1]
        mp = np.array([[np.float32(x)], [np.float32(y)]])
        initstate = [mp[0], mp[1]]
        tp[0] = initstate[0]
        tp[1] = initstate[1]
        pred.append((int(tp[0]), int(tp[1])))
        cv2.circle(currFrame, (tp[0], tp[1]), 10, (0, 0, 255), -1)
    else:
        tp = kalman.predict()
        tp[0] = tp[0] + initstate[0]
        tp[1] = tp[1] + initstate[1]
        pred.append((int(tp[0]), int(tp[1])))
        print("prediction: ")
        print(tp)
        cv2.circle(currFrame,(tp[0],tp[1]), 10, (0,0,255), -1)
        cv2.putText(currFrame, str(tp[0]) + "," + str(tp[1]), (tp[0], tp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if (len(ballCandidatesFilteredProximity) == 1):
            for cand in ballCandidatesFilteredProximity:
                x = cand[0]
                y = cand[1]
                x = x - initstate[0]
                y = y - initstate[1]
                mp = np.array([[np.float32(x)], [np.float32(y)]])
                print("measurement: ")
                print(mp)
                meas.append((x, y))
                kalman.correct(mp)
                print("correction: ")
                print(kalman.correct(mp))
                cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
                cv2.putText(currFrame, str(cand[0]) + "," + str(cand[1]), (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow('Candidate image', currFrame)

        elif(len(ballCandidatesFilteredProximity) > 1):
            print(meas)
            for cand in ballCandidatesFilteredProximity:
                cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
                cv2.putText(currFrame, str(cand[0]) + "," + str(cand[1]), (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow('Candidate image', currFrame)
        else:
            print(meas)
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