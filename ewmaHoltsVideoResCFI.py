import time
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from Modules.foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort, convert480p
from Modules.ballDetectionRes import findContours, sizeDetection, playerProximityDetection, regionDetection, courtBoundaryDetection

# Initializing
datasetName = "Dataset10"
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
elif (datasetName == "Dataset6"):
    startFrameDataset = 0
    endFrameDataset = 180
elif (datasetName == "Dataset7"):
    startFrameDataset = 0
    endFrameDataset = 220
elif (datasetName == "Dataset8"):
    startFrameDataset = 0
    endFrameDataset = 240
elif (datasetName == "Dataset9"):
    startFrameDataset = 0
    endFrameDataset = 200
elif (datasetName == "Dataset10"):
    startFrameDataset = 0
    endFrameDataset = 230
dictFrameNumberscX = {}
dictFrameNumberscY = {}
ballCandidatesPreviousFrame = list()
#Profiling Structures
trackingTime = list()
detectionTime = list()
feTime = list()
processTime = list()

# Reading frames
startTimeReadingFrames = time.time()
# Creating Video Object
cap = cv2.VideoCapture('DatasetVideos/'+datasetName+'.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, startFrameDataset)
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

i = 0
while (cap.isOpened()):
    print("######Start of Frame{}#####".format(i + 1))
    startTimeProcess = time.time()
    if(i == 0):  # If first frame read 3 frames
        ret1, previousFrame = cap.read()
        ret2, currFrame = cap.read()
        ret3, nextFrame = cap.read()
    else:  # Read just the next frame from the 2nd frame onwards
        previousFrame = currFrame
        currFrame = nextFrame
        ret, nextFrame = cap.read()
    print("Frame Number {}".format(i + 1))

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
    final_image = morphologicalOperations(threshFrameDifferencing, 4, 4)

    

    # startTimeBlurringBinary = time.time()
    # # Blurring the binary image to get smooth shapes of objects
    # final_image = cv2.medianBlur(final_image, 7)
    # endTimeBlurringBinary = time.time()
    # print("Final Blur--- %s seconds ---" %
    #       (endTimeBlurringBinary - startTimeBlurringBinary))

    # cv2.imshow('final_image',final_image)
    endTimeForegroundExtraction = time.time()
    print("Foreground Extraction--- %s seconds ---" %
          (endTimeForegroundExtraction - startTimeForeGroundExtraction))
    feTime.append(endTimeForegroundExtraction - startTimeForeGroundExtraction) #Profiling
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
    ballCandidates, playerCadidates, incompletePlayerCandidates = courtBoundaryDetection(datasetName,
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
    detectionTime.append(endTimeBallDetection - startTimeBallDetection) #Profiling
    #
    #
    # BALL TRACKING
    #
    #

    startTimeExponentialPred = time.time()

    # Calculating the centre of the image frame for initstate
    height, width, channels = currFrame.shape
    imageCenter = [width / 2, height / 2]

    # First Frame, No Prediction
    if (i + 1 == 1):
        # If no candidate detected, use image centre as initial state
        if not ballCandidatesFilteredProximity:
            initstate = imageCenter
        # If Candidates detected
        else:
            # If a single candidate detected, use it for the initial state
            if (len(ballCandidatesFilteredProximity) == 1):
                initstate = [ballCandidatesFilteredProximity[0]
                             [0], ballCandidatesFilteredProximity[0][1]]
            # If multiple candidates, calculate candidate closest to the image centre for initial state
            else:
                minDistInitCand = 10000
                for cand in ballCandidatesFilteredProximity:
                    distCenter = math.sqrt(math.pow(
                        (cand[0] - imageCenter[0]), 2) + math.pow((cand[1] - imageCenter[1]), 2))
                    if (distCenter < minDistInitCand):
                        initstate = [cand[0], cand[1]]
                        minDistInitCand = distCenter
        # Update Estimation Parameters
        levelEstimateXcoord.append(initstate[0])
        trendEstimateXcoord.append(betaXcoord*levelEstimateXcoord[i])
        levelEstimateYcoord.append(initstate[1])
        trendEstimateYcoord.append(betaYcoord*levelEstimateYcoord[i])
        if(__debug__):
            cv2.imshow('Candidate image', currFrame)

    # If not the first frame
    else:
        # if one candidate, do prediction then estimation
        if (len(ballCandidatesFilteredProximity) == 1):
            # Prediction using estimation parameters from the previous frame
            predXcoord.append(
                round(levelEstimateXcoord[i-1] + trendEstimateXcoord[i-1], 2))
            predYcoord.append(
                round(levelEstimateYcoord[i - 1] + trendEstimateYcoord[i - 1], 2))

            # Estimation using the detection in current frame
            levelEstimateXcoord.append(alphaXcoord * ballCandidatesFilteredProximity[0][0] + (
                1 - alphaXcoord) * (levelEstimateXcoord[i-1] + trendEstimateXcoord[i-1]))
            trendEstimateXcoord.append(
                betaXcoord * (levelEstimateXcoord[i] - levelEstimateXcoord[i-1]) + (1 - betaXcoord) * trendEstimateXcoord[i-1])

            levelEstimateYcoord.append(alphaYcoord * ballCandidatesFilteredProximity[0][1] + (
                1 - alphaYcoord) * (levelEstimateYcoord[i-1] + trendEstimateYcoord[i-1]))
            trendEstimateYcoord.append(
                betaYcoord*(levelEstimateYcoord[i] - levelEstimateYcoord[i - 1]) + (1 - betaYcoord)*trendEstimateYcoord[i - 1])

            # Appending prediction for plotting
            dictFrameNumberscX[i + 1] = predXcoord[i - 1]
            dictFrameNumberscY[i + 1] = predYcoord[i - 1]

            cv2.circle(currFrame, (int(predXcoord[i-1]), int(
                predYcoord[i-1])), 10, (0, 255, 0), -1)
            cv2.putText(currFrame, str(predXcoord[i-1]) + "," + str(predYcoord[i-1]), (int(
                predXcoord[i-1]) + 1, int(predYcoord[i-1]) + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.drawContours(
                currFrame, [ballCandidatesFilteredProximity[0][3]], -1, (255, 0,), 2)
            if(__debug__):
                cv2.imshow('Candidate image', currFrame)

        # If multiple candidates
        elif (len(ballCandidatesFilteredProximity) > 1):
            # Prediction using previous estimates
            predXcoord.append(
                round(levelEstimateXcoord[i-1] + trendEstimateXcoord[i-1], 2))
            predYcoord.append(
                round(levelEstimateYcoord[i-1] + trendEstimateYcoord[i-1], 2))

            # Calculate candidate closest to the prediction
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

            # If no candidate is close to the prediction, use the prediction and update estimate with the prediction value
            if (minDistObject == 1000):
                cv2.circle(
                    currFrame, (int(predXcoord[i - 1]), int(predXcoord[i - 1])), 10, (0, 0, 255), -1)
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

            # If a candidate close to the prediction, use that for estimation
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

                cv2.circle(
                    currFrame, (int(predXcoord[i - 1]), int(predYcoord[i - 1])), 10, (0, 255, 0), -1)
                cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
                cv2.putText(currFrame, str(cand[0]) + "," + str(
                    cand[1]), (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if(__debug__):
                    cv2.imshow('Candidate image', currFrame)

        # If no candidate detected
        else:
            # Predict and use the prediction for estimation
            lastIndexPredEst = len(predXcoord) - 1
            m = i - lastIndexPredEst - 1
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

            cv2.circle(
                currFrame, (int(predXcoord[i-1]), int(predYcoord[i-1])), 10, (0, 0, 255), -1)
            dictFrameNumberscX[i + 1] = predXcoord[i-1]
            dictFrameNumberscY[i + 1] = predYcoord[i-1]
            if(__debug__):
                cv2.imshow('Candidate image', currFrame)

    endTimeExponentialPred = time.time()
    trackingTime.append(endTimeExponentialPred-startTimeExponentialPred)

    print("Ball Tracking in --- %s seconds ---" %
          (endTimeExponentialPred-startTimeExponentialPred))

    endTimeProcess = time.time()
    processTime.append(endTimeProcess - startTimeProcess)  #Profiling
    
    print("Total Process in --- %s seconds ---" %
          (endTimeProcess-startTimeProcess))
    # Print Ball Trajectory 2D Feature Image
    if (((i + 1) % endFrameDataset) == 0):
        print("Average FE Time: {}".format(
            sum(feTime)/(endFrameDataset-startFrameDataset)))
        print("Average Detection Time: {}".format(
            sum(detectionTime)/(endFrameDataset-startFrameDataset)))
        print("Average Tracking Time: {}".format(
            sum(trackingTime) / (endFrameDataset - startFrameDataset)))
        print("Average Total Process Time: {}".format(
            sum(processTime) / (endFrameDataset - startFrameDataset)))
        # print(dictFrameNumberscX)
        keys = list(dictFrameNumberscX.keys())
        xvalues = list(dictFrameNumberscX.values())
        yvalues = list(dictFrameNumberscY.values())

        plt.xlabel('Frame Number')
        plt.ylabel('Candidate Y-Coordinate Double Exponential')
        plt.title('CFI with Double Exponential Y-Prediction')
        plt.plot(keys, yvalues, 'g--', linewidth=2)
        # plt.axis([-10,250,-5,500])
        plt.show()
        break
    print("######End of Frame#####")
    i += 1  # increments the loop

    # Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
    if(__debug__):
        k = cv2.waitKey(40)
        if k == 27:
            break
        elif k == 32:
            i -= 2
        else:
            continue
