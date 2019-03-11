import glob
import time
import cv2
from scipy.spatial import distance
import math
from foregroundExtraction import readyFrame, frameDifferencing, morphologicalOperations, natural_sort
from ballDetection import filterSize, drawRectangle

startTimeReadingFrames = time.time()
# Location of dataset
filenames = glob.glob("Dataset1/*.jpg")
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

    # endTimeForegroundExtrction=time.time()
    # print("Foreground Extraction--- %s seconds ---" % (endTimeForegroundExtrction - startTimeForeGroundExtraction))
    final_image_copy = final_image.copy()

    contours, hier = cv2.findContours(
        final_image_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 200
    max_area = 1500

    # max_area=330
    # min_area=1

    candidates = list()
    print(len(contours))
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            continue
        area = cv2.contourArea(cnt)
        if area > max_area or area < min_area:
            continue
        x,y,w,h=cv2.boundingRect(cnt)
        rectArea=w*h
        degOfCompactness=float(area)/rectArea
        perimeter = cv2.arcLength(cnt, True)
        (x, y), (MA, ma), angle = cv2.fitEllipseDirect(cnt)
        a = ma/2
        b = MA / 2
        ratioOfAxis=float(ma)/MA
        ellipseArea=math.pi * a * b
        eccentricity = math.sqrt(pow(a, 2)-pow(b, 2))
        eccentricity = round(eccentricity / a, 2)
        formFactor = pow(perimeter, 2) / 4 * math.pi * area
        roundness = 4 * math.pi * area/pow(perimeter, 2)
        formFactor = round(formFactor, 2)
        roundness = round(roundness, 2)
        areaRatio = float(area) / ellipseArea
        candidates.append((cX, cY, area, perimeter))
        # cv2.ellipse(currFrame,ellipse,(0,255,0),2)
        cv2.drawContours(currFrame, [cnt], -1, (0, 0, 255), 1)
        cv2.putText(currFrame, str(ratioOfAxis), (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('Candidate image', currFrame)
        # cv2.imshow('Final image {}'.format(i),threshFrameDifferencing)
    print(len(candidates))



    i += 1  # increments the loop

    # Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
    k = cv2.waitKey(0)
    if k == 27:
        break
    elif k == 32:
        i -= 2
    else:
        continue