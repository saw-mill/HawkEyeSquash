import glob
import time
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Modules.foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort
from Modules.ballDetection import findContours, sizeDetection, playerProximityDetection, regionDetection, courtBoundaryDetection

startTimeReadingFrames = time.time()
datasetName = "Dataset1"
# Location of dataset
filenames = glob.glob(datasetName + "/*.jpg")
totalFramesDataset2 = 194
totalFramesDataset1 = 560

# Reading each frame and storing it in a list
frameList = [cv2.imread(frame) for frame in natural_sort(filenames)]
endTimeReadingFrames = time.time()
print("Reading Frames--- %s seconds ---" %
      (endTimeReadingFrames - startTimeReadingFrames))

# Parsing through the frames
levelEstimateXcoord = []
trendEstimateXcoord = []
levelEstimateYcoord = []
trendEstimateYcoord = []

predXcoord = []
predYcoord = []

alphaXcoord = 0.95
betaXcoord = 0.01

alphaYcoord = 0.95
betaYcoord = 0.05

dictFrameNumberscX = {}
dictFrameNumberscY = {}
ballCandidatesPreviousFrame = list()
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
    final_image = morphologicalOperations(threshFrameDifferencing, 4, 4)

    # startTimeBlurringBinary = time.time()
    # # Blurring the binary image to get smooth shapes of objects
    # final_image = cv2.medianBlur(img_erosion, 7)
    # endTimeBlurringBinary = time.time()
    # print("Final Blur--- %s seconds ---" %
    #       (endTimeBlurringBinary - startTimeBlurringBinary))

    endTimeForegroundExtraction = time.time()
    print("Foreground Extraction--- %s seconds ---" %
          (endTimeForegroundExtraction - startTimeForeGroundExtraction))

    #
    #
    # BALL DETECTION
    #
    #
    startTimeBallDetection = time.time()

    # Making a copy of pre-processed image frame
    # final_image_copy = final_image.copy()

    # Finding contours in the frame
    contours, hier = findContours(final_image)

    # Separating candidates based on size
    ballCandidates, playerCadidates, incompletePlayerCandidates = sizeDetection(
        contours, currFrame, i)

    # Removing candidates outside the Court Boundary in Dataset2
    if (datasetName == 'Dataset2'):
        ballCandidates, playerCadidates, incompletePlayerCandidates = courtBoundaryDetection(
            ballCandidates, playerCadidates, incompletePlayerCandidates, currFrame)

    # Removing Candidates that are close to the Players
    ballCandidatesFiltered = playerProximityDetection(
        ballCandidates, playerCadidates, incompletePlayerCandidates, currFrame)

    # Removing candidates that are not in their expected region after motion
    ballCandidatesFilteredProximity, ballCandidatesPreviousFrame = regionDetection(
        ballCandidatesFiltered, ballCandidatesPreviousFrame, currFrame)

    endTimeBallDetection = time.time()
    print("Ball Detection--- %s seconds ---" %
          (endTimeBallDetection - startTimeBallDetection))

    startTimeExponentialPred = time.time()

    height, width, channels = currFrame.shape
    imageCenter = [width / 2, height / 2]

    if (i + 1 == 1):
        # No prediction
        if not ballCandidatesFilteredProximity:
            initstate = imageCenter
        else:
            if (len(ballCandidatesFilteredProximity) == 1):
                initstate = [ballCandidatesFilteredProximity[0][0], ballCandidatesFilteredProximity[0][1]]
            else:
                minDistInitCand = 10000
                for cand in ballCandidatesFilteredProximity:
                    distCenter = math.sqrt(math.pow(
                        (cand[0] - imageCenter[0]), 2) + math.pow((cand[1] - imageCenter[1]), 2))
                    if (distCenter < minDistInitCand):
                        initstate = [cand[0], cand[1]]
                        minDistInitCand = distCenter
        levelEstimateXcoord.append(initstate[0])
        trendEstimateXcoord.append(betaXcoord*levelEstimateXcoord[i])
        levelEstimateYcoord.append(initstate[1])
        trendEstimateYcoord.append(betaYcoord*levelEstimateYcoord[i])
        # print(levelEstimateXcoord)
        # print(levelEstimateYcoord)
        # print(trendEstimateXcoord)
        # print(trendEstimateYcoord)
        # print(predXcoord)
        # print(predYcoord)
        cv2.drawContours(
            currFrame, [ballCandidatesFilteredProximity[0][3]], -1, (255, 0,), 2)
        # cv2.imshow('Candidate image', currFrame)

    else:
        # if one candidate, do estimation
        if (len(ballCandidatesFilteredProximity) == 1):
            predXcoord.append(
            round(levelEstimateXcoord[i-1] + trendEstimateXcoord[i-1], 2))
            predYcoord.append(
            round(levelEstimateYcoord[i-1] + trendEstimateYcoord[i-1], 2))

            levelEstimateXcoord.append(alphaXcoord * ballCandidatesFilteredProximity[0][0] + (
                1 - alphaXcoord) * (levelEstimateXcoord[i-1] + trendEstimateXcoord[i-1]))
            trendEstimateXcoord.append(
                betaXcoord * (levelEstimateXcoord[i] - levelEstimateXcoord[i-1]) + (1 - betaXcoord) * trendEstimateXcoord[i-1])

            levelEstimateYcoord.append(alphaYcoord * ballCandidatesFilteredProximity[0][1] + (
                1 - alphaYcoord) * (levelEstimateYcoord[i-1] + trendEstimateYcoord[i-1]))
            trendEstimateYcoord.append(
                betaYcoord*(levelEstimateYcoord[i] - levelEstimateYcoord[i - 1]) + (1 - betaYcoord)*trendEstimateYcoord[i - 1])
                
            dictFrameNumberscX[i + 1] = predXcoord[i - 1]
            dictFrameNumberscY[i + 1] = predYcoord[i - 1]

            # print(predXcoord)
            # print(predYcoord)
            # print(levelEstimateXcoord)
            # print(levelEstimateYcoord)
            # print(trendEstimateXcoord)
            # print(trendEstimateYcoord)

            cv2.circle(currFrame, (int(predXcoord[i-1]), int(
                predYcoord[i-1])), 10, (0, 255, 0), -1)
            cv2.putText(currFrame, str(predXcoord[i-1]) + "," + str(predYcoord[i-1]), (int(
                predXcoord[i-1]) + 1, int(predYcoord[i-1]) + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.drawContours(
                currFrame, [ballCandidatesFilteredProximity[0][3]], -1, (255, 0,), 2)
            # cv2.imshow('Candidate image', currFrame)

        elif (len(ballCandidatesFilteredProximity) > 1):
            predXcoord.append(
            round(levelEstimateXcoord[i-1] + trendEstimateXcoord[i-1], 2))
            predYcoord.append(
            round(levelEstimateYcoord[i-1] + trendEstimateYcoord[i-1], 2))

            minDistObject = 1000
            minDistXcoord = 0
            minDistYcoord = 0
            for cand in ballCandidatesFilteredProximity:
                distncePredAct = math.sqrt(
                    math.pow((cand[0] - predXcoord[i-1]), 2) + math.pow((cand[1] - predYcoord[i-1]), 2))
                # #drawing a line
                # cv2.line(currFrame, (int(cand[0]), int(cand[1])), (int(
                # tp[0]), int(tp[1])), (255, 0, 0), 2)
                # xmidPointPlayer = (cand[0]+tp[0])*0.5
                # ymidPointPlayer = (cand[1]+tp[1])*0.5
                # cv2.putText(currFrame, str(round(distncePredAct,2)), (int(xmidPointPlayer), int(
                # ymidPointPlayer)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # print("Distance predact {}".format(distncePredAct))

                if (distncePredAct < 50):
                    if (distncePredAct < minDistObject):
                        minDistObject = distncePredAct
                        minDistXcoord = cand[0]
                        minDistYcoord = cand[1]

            if (minDistObject == 1000):
                cv2.circle(currFrame, (int(predXcoord[i - 1]), int(predXcoord[i - 1])), 10, (0, 0, 255), -1)
                x = predXcoord[-1]
                y = predYcoord[-1]

                levelEstimateXcoord.append(alphaXcoord * x + (
                1 - alphaXcoord) * (levelEstimateXcoord[i-1] + trendEstimateXcoord[i-1]))
                trendEstimateXcoord.append(
                betaXcoord * (levelEstimateXcoord[i] - levelEstimateXcoord[i-1]) + (1 - betaXcoord) * trendEstimateXcoord[i-1])

                levelEstimateYcoord.append(alphaYcoord * y + (
                    1 - alphaYcoord) * (levelEstimateYcoord[i-1] + trendEstimateYcoord[i-1]))
                trendEstimateYcoord.append(
                    betaYcoord*(levelEstimateYcoord[i] - levelEstimateYcoord[i - 1]) + (1 - betaYcoord)*trendEstimateYcoord[i - 1])

                dictFrameNumberscX[i + 1] = predXcoord[i - 1]
                dictFrameNumberscY[i + 1] = predYcoord[i - 1]
                # print(predXcoord)
                # print(predYcoord)
                # print(levelEstimateXcoord)
                # print(levelEstimateYcoord)
                # print(trendEstimateXcoord)
                # print(trendEstimateYcoord)
            else:
                x = minDistXcoord
                y = minDistYcoord
                levelEstimateXcoord.append(alphaXcoord * x + (
                1 - alphaXcoord) * (levelEstimateXcoord[i-1] + trendEstimateXcoord[i-1]))
                trendEstimateXcoord.append(
                betaXcoord * (levelEstimateXcoord[i] - levelEstimateXcoord[i-1]) + (1 - betaXcoord) * trendEstimateXcoord[i-1])

                levelEstimateYcoord.append(alphaYcoord * y + (
                    1 - alphaYcoord) * (levelEstimateYcoord[i-1] + trendEstimateYcoord[i-1]))
                trendEstimateYcoord.append(
                    betaYcoord*(levelEstimateYcoord[i] - levelEstimateYcoord[i - 1]) + (1 - betaYcoord)*trendEstimateYcoord[i - 1])
                
                dictFrameNumberscX[i + 1] = predXcoord[i - 1]
                dictFrameNumberscY[i + 1] = predYcoord[i - 1]
                # print(predXcoord)
                # print(predYcoord)
                # print(levelEstimateXcoord)
                # print(levelEstimateYcoord)
                # print(trendEstimateXcoord)
                # print(trendEstimateYcoord)

                cv2.circle(
                        currFrame, (int(predXcoord[i - 1]), int(predYcoord[i - 1])), 10, (0, 255, 0), -1)
                cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
                cv2.putText(currFrame, str(cand[0]) + "," + str(
                        cand[1]), (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # cv2.imshow('Candidate image', currFrame)
        
        else:
            lastIndexPredEst = len(predXcoord) - 1
            m = i- lastIndexPredEst -1
            # print("m={}".format(m))
            predXcoord.append(
            round(levelEstimateXcoord[i-1] + m*trendEstimateXcoord[i-1], 2))
            predYcoord.append(
            round(levelEstimateYcoord[i - 1] + m*trendEstimateYcoord[i - 1], 2))

            x = predXcoord[-1]
            y = predYcoord[-1]

            levelEstimateXcoord.append(alphaXcoord * x + (
            1 - alphaXcoord) * (levelEstimateXcoord[i-1] + trendEstimateXcoord[i-1]))
            trendEstimateXcoord.append(
            betaXcoord * (levelEstimateXcoord[i] - levelEstimateXcoord[i-1]) + (1 - betaXcoord) * trendEstimateXcoord[i-1])

            levelEstimateYcoord.append(alphaYcoord * y + (
                1 - alphaYcoord) * (levelEstimateYcoord[i-1] + trendEstimateYcoord[i-1]))
            trendEstimateYcoord.append(
                betaYcoord*(levelEstimateYcoord[i] - levelEstimateYcoord[i - 1]) + (1 - betaYcoord)*trendEstimateYcoord[i - 1])

            dictFrameNumberscX[i + 1] = predXcoord[i - 1]
            dictFrameNumberscY[i + 1] = predYcoord[i - 1]
            # print(predXcoord)
            # print(predYcoord)
            # print(levelEstimateXcoord)
            # print(levelEstimateYcoord)
            # print(trendEstimateXcoord)
            # print(trendEstimateYcoord)

            cv2.circle(currFrame, (int(predXcoord[i-1]), int(predYcoord[i-1])), 10, (0, 0, 255), -1)
            dictFrameNumberscX[i + 1] = predXcoord[i-1]
            dictFrameNumberscY[i + 1] = predYcoord[i-1]
            # cv2.imshow('Candidate image', currFrame)

    endTimeExponentialPred = time.time()

    print("Ball Tracking in --- %s seconds ---" % (endTimeExponentialPred-startTimeExponentialPred))


    if (((i + 1) % totalFramesDataset1) == 0):
        print(dictFrameNumberscX)
        keys = list(dictFrameNumberscX.keys())
        xvalues = list(dictFrameNumberscX.values())
        yvalues = list(dictFrameNumberscY.values())
        plt.xlabel('Frame Number')
        plt.ylabel('Candidate Kalman X-Coordinate')
        plt.title('CFI with Kalman X Prediction')
        plt.plot(keys, xvalues, 'r--', linewidth=2)
        plt.show()

        plt.xlabel('Frame Number')
        plt.ylabel('Candidate Kalman Y-Coordinate')
        plt.title('CFI with Kalman Y Prediction')
        plt.plot(keys, yvalues, 'g--', linewidth=2)
        plt.show()

    i += 1  # increments the loop

    # Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
    # k = cv2.waitKey(0)
    # if k == 27:
    #     break
    # elif k == 32:
    #     i -= 2
    # else:
    #     continue
