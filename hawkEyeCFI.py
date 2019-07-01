import time
import cv2
import matplotlib.pyplot as plt
from Modules.foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort,convert480p
from Modules.ballDetectionRes import findContours, sizeDetection, playerProximityDetection, regionDetection, courtBoundaryDetection

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
    # if (datasetName == 'Dataset2'):
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

    # Adding candidates in containers for CFI plotting
    dictFrameNumberscX[i+1] = []
    dictFrameNumberscY[i+1] = []
    for cand in ballCandidatesFilteredProximity:
        dictFrameNumberscX.get(i+1).append(cand[0])
        dictFrameNumberscY.get(i+1).append(cand[1])

    # Drawing and Displaying contours around the candidates
    if (len(ballCandidatesFilteredProximity) > 0):
        for cand in ballCandidatesFilteredProximity:
            cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
            cv2.putText(currFrame, "A:"+str(
                    cand[2])+" MD:"+str(cand[5]), (cand[0] + 1, cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if(__debug__):
                cv2.imshow('Candidate image', currFrame)
    else:
        if(__debug__):
            cv2.imshow('Candidate image', currFrame)

    if (((i + 1) % endFrameDataset) == 0):
        print(dictFrameNumberscX)

        for data_dict in dictFrameNumberscY.items():
            print(data_dict)
            x = data_dict[0]
            values = data_dict[1]
            for value in values:
                # plt.subplot(1, 2, 2)
                plt.scatter(x, value)
                plt.xlabel('Frame Number')
                plt.ylabel('Candidate Y-Coordinate')
                plt.title("Candidate Feature Image Y-coordinate")
                plt.axis([-10,250,-5,500])
        plt.show()

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
