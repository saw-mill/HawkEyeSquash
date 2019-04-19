import glob
import time
import cv2
from Modules.foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort

startTimeReadingFrames = time.time()
# Location of dataset
filenames = glob.glob("Dataset2/*.jpg")
# filenames = glob.glob("Testing/*.jpg")

# Reading each frame and storing it in a list
frameList = [cv2.imread(frame) for frame in natural_sort(filenames)]
endTimeReadingFrames = time.time()
print("Reading Frames--- %s seconds ---" %
      (endTimeReadingFrames - startTimeReadingFrames))

startTimeForeGroundExtraction = time.time()
# Parsing through the frames

i = 0
while i < (len(frameList)-2):
    # cv2.imshow("Frame {}".format(i),frameList[i])

    # Storing three frames
    previousFrame = frameList[i]
    currFrame = frameList[i+1]
    nextFrame = frameList[i + 2]
    
    print("Frame Number {}".format(i+1))

    # Readying the frames
    previousFrameGray, currFrameGray, nextFrameGray = readyFrame(
        previousFrame, currFrame, nextFrame)

    # Performing frame differencing
    threshFrameDifferencing = frameDifferencing(
        previousFrameGray, currFrameGray, nextFrameGray)

    # Performing morphological operations
    final_image = morphologicalOperations(threshFrameDifferencing, 4, 4)

    endTimeForegroundExtrction=time.time()
    print("Foreground Extraction--- %s seconds ---" % (endTimeForegroundExtrction - startTimeForeGroundExtraction))

    final_image_copy = final_image.copy()
    contours, hier = cv2.findContours(
        final_image_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_BallArea = 300
    max_BallArea = 1500
    min_PlayerArea = 10000
    min_IncompletePlayerArea = 1800
    
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

    i += 1  # increments the loop

    # Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
    k = cv2.waitKey(0)
    if k == 27:
        break
    elif k == 32:
        i -= 2
    else:
        continue
