import glob
import time
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from Modules.foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort, convert480p
from Modules.ballDetection import findContours, sizeDetection, playerProximityDetection, regionDetection, courtBoundaryDetection

startTimeReadingFrames = time.time()
datasetName = "Dataset1"
# Location of dataset
filenames = glob.glob(datasetName + "/*.jpg")

if (datasetName == "Dataset1"):
    totalFramesDataset = 560
elif (datasetName == "Dataset2"):
    totalFramesDataset = 194

# Reading each frame and storing it in a list
frameList = [cv2.imread(frame) for frame in natural_sort(filenames)]
endTimeReadingFrames = time.time()
print("Reading Frames--- %s seconds ---" %
      (endTimeReadingFrames - startTimeReadingFrames))

# Parsing through the frames
dictFrameNumberscX = {}
dictFrameNumberscY = {}
ballCandidatesPreviousFrame = list()

startKalmanInitTime = time.time()

mp = np.array((2, 1), np.float32)  # measurement
tp = np.zeros((2, 1), np.float32)  # tracked / prediction
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.009
kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.00003

endKalmanInitTime = time.time()

trackingTime = list()
i = 0
while i < (len(frameList)-2):
    # cv2.imshow("Frame {}".format(i),frameList[i])

    # Storing three frames
    previousFrame = frameList[i]
    currFrame = frameList[i+1]
    nextFrame = frameList[i + 2]

    # # Changing from 720p to 480p
    # previousFrame = convert480p(previousFrame)
    # currFrame = convert480p(currFrame)
    # nextFrame = convert480p(nextFrame)

    print("Frame Number {}".format(i+1))
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

    # final_image = cv2.medianBlur(final_image,7)

    # cv2.imshow('final image', final_image)
    endTimeForegroundExtraction = time.time()
    print("Foreground Extraction--- %s seconds ---" %
          (endTimeForegroundExtraction - startTimeForeGroundExtraction))

    #
    #
    # BALL DETECTION
    #
    #
    startTimeBallDetection = time.time()

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

    #
    #
    # BALL TRACKING
    #
    #

    startKalmanPredTime = time.time()

    # Calculating the centre of the image frame for initstate
    height, width, channels = currFrame.shape
    imageCenter = [width/2, height/2]

    # First Frame
    if (i + 1 == 1):
        # If no candidate detected, use image centre as initial state
        if not ballCandidatesFilteredProximity:
            initstate = imageCenter
        # If Candidates detected
        else:
            # If a single candidate detected, use it for the initial state
            if (len(ballCandidatesFilteredProximity) == 1):
                x = ballCandidatesFilteredProximity[0][0]
                y = ballCandidatesFilteredProximity[0][1]
                mp = np.array([[np.float32(x)], [np.float32(y)]])
                initstate = [mp[0], mp[1]]
            # If multiple candidates, calculate candidate closest to the image centre for initial state
            else:
                minDistInitCand = 10000
                for cand in ballCandidatesFilteredProximity:
                    distCenter = math.sqrt(math.pow(
                        (cand[0] - imageCenter[0]), 2) + math.pow((cand[1] - imageCenter[1]), 2))
                    if (distCenter < minDistInitCand):
                        initstate = [cand[0], cand[1]]
                        minDistInitCand = distCenter
        # Using Initstate for First Prediction
        tp[0] = initstate[0]
        tp[1] = initstate[1]
        cv2.circle(currFrame, (tp[0], tp[1]), 10, (0, 0, 255), -1)
        dictFrameNumberscX[i + 1] = tp[0]
        dictFrameNumberscY[i + 1] = tp[1]
        if(__debug__):
            cv2.imshow('Candidate image', currFrame)
    # If not the first frame
    else:
        # Do Prediction
        tp = kalman.predict()
        tp[0] = tp[0] + initstate[0]
        tp[1] = tp[1] + initstate[1]

        # If one candidate, measure and correct
        if (len(ballCandidatesFilteredProximity) == 1):
            for cand in ballCandidatesFilteredProximity:
                # distncePredAct = math.sqrt(
                #     math.pow((cand[0] - tp[0]), 2) + math.pow((cand[1] - tp[1]), 2))
                x = cand[0]
                y = cand[1]
                x = x - initstate[0]
                y = y - initstate[1]
                mp = np.array([[np.float32(x)], [np.float32(y)]])
                corrected = kalman.correct(mp)
                corrected[0] = corrected[0] + initstate[0]
                corrected[1] = corrected[1] + initstate[1]
                cv2.circle(
                    currFrame, (corrected[0], corrected[1]), 10, (0, 255, 0), -1)
                dictFrameNumberscX[i + 1] = corrected[0]
                dictFrameNumberscY[i + 1] = corrected[1]
            # cv2.circle(currFrame, (tp[0], tp[1]),
                #            10, (0, 0, 255), -1)  # pred

                # #drawing a line
                # cv2.line(currFrame, (int(cand[0]), int(cand[1])), (int(
                # tp[0]), int(tp[1])), (255, 0, 0), 2)
                # xmidPointPlayer = (cand[0]+tp[0])*0.5
                # ymidPointPlayer = (cand[1]+tp[1])*0.5
                # cv2.putText(currFrame, str(round(distncePredAct,2)), (int(xmidPointPlayer), int(
                # ymidPointPlayer)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # print("Distance predact {}".format(distncePredAct))

                cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
                cv2.putText(currFrame, str(
                    cand[2]), (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if(__debug__):
                cv2.imshow('Candidate image', currFrame)

        # If multiple candidates,
        elif(len(ballCandidatesFilteredProximity) > 1):
            minDistObject = 1000
            minDistXcoord = 0
            minDistYcoord = 0
            # Calculate candidate closest to the prediction
            for cand in ballCandidatesFilteredProximity:
                distncePredAct = math.sqrt(
                    math.pow((cand[0] - tp[0]), 2) + math.pow((cand[1] - tp[1]), 2))
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
            # If no candidate is close to the prediction, predict only
            if (minDistObject == 1000):
                cv2.circle(currFrame, (tp[0], tp[1]), 10, (0, 0, 255), -1)
                dictFrameNumberscX[i + 1] = tp[0]
                dictFrameNumberscY[i + 1] = tp[1]
            # If a candidate close to the prediction, use it for measurement and correction
            else:
                x = minDistXcoord
                y = minDistYcoord
                x = x - initstate[0]
                y = y - initstate[1]
                mp = np.array([[np.float32(x)], [np.float32(y)]])
                corrected = kalman.correct(mp)
                corrected[0] = corrected[0] + initstate[0]
                corrected[1] = corrected[1] + initstate[1]
                cv2.circle(
                    currFrame, (corrected[0], corrected[1]), 10, (0, 255, 0), -1)
                dictFrameNumberscX[i + 1] = corrected[0]
                dictFrameNumberscY[i+1] = corrected[1]

                cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
                cv2.putText(currFrame, str(
                    cand[2]), (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if(__debug__):
                cv2.imshow('Candidate image', currFrame)
        # If no candidate detected, predict only
        else:
            cv2.circle(currFrame, (tp[0], tp[1]), 10, (0, 0, 255), -1)
            dictFrameNumberscX[i + 1] = tp[0]
            dictFrameNumberscY[i+1] = tp[1]
            if(__debug__):
                cv2.imshow('Candidate image', currFrame)

    endKalmanPredTime = time.time()

    trackingTime.append((endKalmanPredTime -
                         startKalmanPredTime)+(endKalmanInitTime-startKalmanInitTime))

    print("Ball Tracking in --- %s seconds ---" % ((endKalmanPredTime -
                                                    startKalmanPredTime)+(endKalmanInitTime-startKalmanInitTime)))

    # Print Ball Trajectory 2D Feature Image
    if (((i + 1) % totalFramesDataset) == 0):
        print("Average Tracking Time: {}".format(
            sum(trackingTime)/totalFramesDataset))
        # print(dictFrameNumberscX)
        keys = list(dictFrameNumberscX.keys())
        xvalues = list(dictFrameNumberscX.values())
        yvalues = list(dictFrameNumberscY.values())
        plt.xlabel('Frame Number')
        plt.ylabel('Candidate Kalman X-Coordinate')
        plt.title('CFI with Kalman X Prediction')
        plt.plot(keys, xvalues, 'r--', linewidth=2)
        # plt.axis([-20, 600, 0, 1300])
        # plt.axis([-20,210,100,1200])
        plt.show()

        plt.xlabel('Frame Number')
        plt.ylabel('Candidate Kalman Y-Coordinate')
        plt.title('CFI with Kalman Y Prediction')
        plt.plot(keys, yvalues, 'g--', linewidth=2)
        # plt.axis([-20, 600, 25, 1000])
        # plt.axis([-20,210,50,900])
        plt.show()

        # scatter plot

        # print(dictFrameNumberscY)
        # for data_dict in dictFrameNumberscX.items():
        #     print(data_dict)
        #     x = data_dict[0]
        #     values = data_dict[1]
        #     for value in values:
        #         # plt.subplot(1, 2, 1)
        #         plt.scatter(x,value)
        #         plt.xlabel('Frame Number')
        #         plt.ylabel('Candidate X-Coordinate')
        #         plt.title("Candidate Feature Image X-coordinate")
        # dictFrameNumberscX.clear()
        # plt.show()

        # plt.xlabel('Frame Number')
        # plt.ylabel('Candidate Kalman Y-Coordinate')
        # plt.title('CFI with Kalman Y Prediction')
        # plt.plot(keys, yvalues, 'g--', linewidth=2)
        # plt.show()

    i += 1  # increments the loop

    # Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
    if(__debug__):
        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == 32:
            i -= 2
        else:
            continue
