import glob
import time
import cv2
import math
from Modules.foregroundExtractionD5 import readyFrame, frameDifferencing, morphologicalOperations, natural_sort, convert480p

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
    startFrameDataset = 0
    endFrameDataset = 550
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
    endTimeBlurringBinary = time.time()
    print("Final Blur--- %s seconds ---" %
          (endTimeBlurringBinary - startTimeBlurringBinary))

    endTimeForegroundExtrction=time.time()
    print("Foreground Extraction--- %s seconds ---" % (endTimeForegroundExtrction - startTimeForeGroundExtraction))

    final_image_copy = final_image.copy()
    contours, hier = cv2.findContours(
        final_image_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_BallArea = 500
    max_BallArea = 1500
    min_PlayerArea = 4000
    min_IncompletePlayerArea = 1500
    min_BallDistance = 95
    
    ballCandidates = list()
    playerCadidates = list()
    incompletePlayerCandidates = list()
    print(len(contours))
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            continue
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > min_PlayerArea:
            playerCadidates.append([cX, cY, area, perimeter])
        elif area > min_IncompletePlayerArea and area < min_PlayerArea:
            incompletePlayerCandidates.append([cX, cY, area, perimeter])
        elif area < max_BallArea and area > min_BallArea:
            ballCandidates.append([cX, cY, area, perimeter])
            cv2.drawContours(currFrame, [cnt], -1, (0, 0, 255), 1)
            cv2.putText(currFrame, str(cX)+","+str(cY), (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        elif area < min_BallArea:
            continue
        # cv2.imshow('Candidate image', currFrame)
    print("Ball Candidates: %d" % len(ballCandidates))
    print("Player Candidates: %d" % len(playerCadidates))
    print("Incomplete Player Candidate: %d" % len(incompletePlayerCandidates))
    
    ballCandidatesFiltered= list()

    if not ballCandidates:
        print("No ball Candidates")
        cv2.imshow('Candidate image', currFrame)
    else:
        minDist = 99999999
        minDistPoint = []
        for cand in ballCandidates:
            minDist = 99999999
            minDistPoint = []
            if len(playerCadidates) > 1:
                for player in playerCadidates:
                    dist = math.sqrt(math.pow((cand[0]-player[0]),2)+math.pow((cand[1]-player[1]),2))
                    if dist < minDist:
                        minDist = dist
                        minDistPoint = [player[0],player[1]]
            elif len(playerCadidates) == 1:
                distFromPlayer = math.sqrt(math.pow((cand[0] - playerCadidates[0][0]), 2) + math.pow((cand[1] - playerCadidates[0][1]), 2))
                if distFromPlayer < minDist:
                    minDist = distFromPlayer
                    minDistPoint = [playerCadidates[0][0],playerCadidates[0][1]]
                for part in incompletePlayerCandidates:
                    dist = math.sqrt(math.pow((cand[0] - part[0]), 2) + math.pow((cand[1] - part[1]), 2))
                    if dist < minDist:
                        minDist = dist
                        minDistPoint = [part[0],part[1]]
            elif len(incompletePlayerCandidates) > 0:
                for part in incompletePlayerCandidates:
                    dist = math.sqrt(math.pow((cand[0] - part[0]), 2) + math.pow((cand[1] - part[1]), 2))
                    if dist < minDist:
                        minDist = dist
                        minDistPoint = [part[0], part[1]]
            else:
                continue
            minDist = round(minDist, 2)
            cv2.line(currFrame, (int(cand[0]), int(cand[1])), (int(
            minDistPoint[0]), int(minDistPoint[1])), (255, 0, 0), 2)
            xmidPointPlayer = (cand[0]+minDistPoint[0])*0.5
            ymidPointPlayer = (cand[1]+minDistPoint[1])*0.5
            cv2.putText(currFrame, str(minDist), (int(xmidPointPlayer), int(
            ymidPointPlayer)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Candidate image', currFrame)
            # if (minDist >= min_BallDistance):
            #     cand.append(minDist)
            #     cand.append(minDistPoint)
            #     ballCandidatesFiltered.append(cand)
            #     cv2.line(currFrame, (int(cand[0]), int(cand[1])), (int(
            #     minDistPoint[0]), int(minDistPoint[1])), (255, 0, 0), 2)
            #     xmidPointPlayer = (cand[0]+minDistPoint[0])*0.5
            #     ymidPointPlayer = (cand[1]+minDistPoint[1])*0.5
            #     cv2.putText(currFrame, str(minDist), (int(xmidPointPlayer), int(
            #     ymidPointPlayer)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
