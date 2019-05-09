import glob
import time
import cv2
import matplotlib.pyplot as plt
from Modules.foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort
from Modules.ballDetection import findContours, sizeDetection, playerProximityDetection, regionDetection, courtBoundaryDetection

startTimeReadingFrames = time.time()
datasetName = "Dataset2"
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
i = 0
while i < (len(frameList)-2):
    # cv2.imshow("Frame {}".format(i),frameList[i])

    # Storing three frames
    previousFrame = frameList[i]
    currFrame = frameList[i+1]
    nextFrame = frameList[i + 2]

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

    # Adding candidates in containers for CFI plotting
    dictFrameNumberscX[i+1] = []
    dictFrameNumberscY[i+1] = []
    for cand in ballCandidatesFilteredProximity:
        dictFrameNumberscX.get(i+1).append(cand[0])
        dictFrameNumberscY.get(i+1).append(cand[1])

    # Drawing and Displaying contours around the candidates
    for cand in ballCandidatesFilteredProximity:
        if not cand:
            cv2.imshow('Candidate image', currFrame)
        else:
            cv2.drawContours(currFrame, [cand[3]], -1, (255, 0,), 2)
            cv2.putText(currFrame, str(cand[0]) + "," + str(cand[1]), (cand[0] + 1,
                                                                       cand[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if(__debug__):
                cv2.imshow('Candidate image', currFrame)

    if (((i + 1) % totalFramesDataset) == 0):
        print(dictFrameNumberscX)

        for data_dict in dictFrameNumberscX.items():
            print(data_dict)
            x = data_dict[0]
            values = data_dict[1]
            for value in values:
                # plt.subplot(1, 2, 1)
                plt.scatter(x, value)
                plt.xlabel('Frame Number')
                plt.ylabel('Candidate X-Coordinate')
                plt.title("Candidate Feature Image X-coordinate")
                # plt.axis([-20, 600, 0, 1300])
                plt.axis([-20, 210, 100, 1200])
        plt.show()

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
                # plt.axis([-20, 600, 25, 1000])
                plt.axis([-20, 210, 50, 900])
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
