import glob
import time
import cv2
from Modules.foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort,convert480p

startTimeReadingFrames = time.time()
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
    startFrameDataset = 0
    endFrameDataset = 150
elif (datasetName == "Dataset6"):
    startFrameDataset = 1
    endFrameDataset = 180
elif (datasetName == "Dataset7"):
    startFrameDataset = 0
    endFrameDataset = 220
elif (datasetName == "Dataset8"):
    startFrameDataset = 1
    endFrameDataset = 240
elif (datasetName == "Dataset9"):
    startFrameDataset = 1
    endFrameDataset = 246
elif (datasetName == "Dataset10"):
    startFrameDataset = 1
    endFrameDataset = 233
elif (datasetName == "Dataset11"):
    startFrameDataset = 1
    endFrameDataset = 220
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

    previousFrameGray, currFrameGray, nextFrameGray = readyFrame(
        previousFrame, currFrame, nextFrame)

    # Performing frame differencing
    threshFrameDifferencing = frameDifferencing(
        previousFrameGray, currFrameGray, nextFrameGray)

    # Performing morphological operations
    final_image = morphologicalOperations(threshFrameDifferencing, 4, 4)
    # cv2.imshow('Final Image',final_image)
    endTimeForegroundExtrction=time.time()
    print("Foreground Extraction--- %s seconds ---" % (endTimeForegroundExtrction - startTimeForeGroundExtraction))

    final_image_copy = final_image.copy()
    contours, hier = cv2.findContours(
        final_image_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_BallArea = 355
    max_BallArea = 1200
    min_PlayerArea = 4000
    min_IncompletePlayerArea = 1200
    
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
        cv2.drawContours(currFrame, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(currFrame, str(area), (cX, cY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # if area > min_PlayerArea:
        #     playerCadidates.append((cX, cY, area, perimeter))
        # elif area > min_IncompletePlayerArea and area < min_PlayerArea:
        #     incompletePlayerCandidates.append((cX, cY, area, perimeter))
        # elif area < max_BallArea and area > min_BallArea:
        #     ballCandidates.append((cX, cY, area, perimeter))
        #     cv2.drawContours(currFrame, [cnt], -1, (0, 255, 0), 1)
        #     cv2.putText(currFrame, str(area), (cX, cY),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # elif area < min_BallArea:
        #     continue
        cv2.imshow('Candidate image', currFrame)
    print("Ball Canidates: %d" % len(ballCandidates))
    print("Player Candidates: %d" % len(playerCadidates))
    print("Incomplete Player Candidate: %d" %len(incompletePlayerCandidates))

    if (((i + 1) % endFrameDataset) == 0):
        break

    i += 1  # increments the loop

    # Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
    k = cv2.waitKey(0)
    if k == 27:
        break
    elif k == 32:
        i -= 2
    else:
        continue
