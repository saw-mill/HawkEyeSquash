import glob
import time
import cv2
import numpy as np
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

frameNumber = 0
while frameNumber < (len(frameList)-2):
    # cv2.imshow("Frame {}".format(i),frameList[i])

    # Storing three frames
    previousFrame = frameList[frameNumber]
    currFrame = frameList[frameNumber+1]
    nextFrame = frameList[frameNumber+2]

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

    min_area = 300
    max_area = 1500

    startTimeSizeFilter = time.time()
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        final_image, ltype=cv2.CV_16U)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    fontColor = 127
    lineType = 1

    d = final_image.copy()
    candidates = list()
    for stat in stats:
        area = stat[cv2.CC_STAT_AREA]
        if area > max_area or area < min_area:
            # Skip big objects (players) and skip small objects (noisy candidates)
            continue

        lt = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
        rb = (lt[0] + stat[cv2.CC_STAT_WIDTH],
              lt[1] + stat[cv2.CC_STAT_HEIGHT])
        bottomLeftCornerOfText = (lt[0], lt[1] - 15)

        candidates.append((lt, rb, area))
        cv2.rectangle(d, lt, rb, fontColor, lineType)

        cv2.putText(d, "{}: {:.0f}".format(len(candidates), stat[cv2.CC_STAT_AREA]),
                    bottomLeftCornerOfText,
                    font, fontScale, fontColor, lineType)
    endTimeSizeFilter = time.time()
    print("Size Filter--- %s seconds ---" %
          (endTimeSizeFilter - startTimeSizeFilter))

    cv2.imshow('Final image', currFrame)

    # # print(len(candidates))
    cv2.imshow('rectangles', d)

    # Thinning threshold.
    psi = 1

    # Area matching threshold.
    gamma = 0.3

    fom_detected = False
    start = time.time()

    sub_regions = list()
    for i, candidate in enumerate(candidates):
        # The first two elements of each `candidate` tuple are
        # the opposing corners of the bounding box.
        x1, y1 = candidate[0]
        x2, y2 = candidate[1]

        # We had placed the candidate's area in the third element of the tuple.
        actual_area = candidate[2]

        # For each candidate, estimate the "radius" using a distance transform.
        # The transform is computed on the (small) bounding rectangle.
        cand = d[y1:y2, x1:x2]
        dt = cv2.distanceTransform(
            cand, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
        radius = np.amax(dt)

        # "Thinning" of pixels "close to the center" to estimate a
        # potential FOM path.
        ret, Pt = cv2.threshold(dt, psi * radius, 255, cv2.THRESH_BINARY)

        # TODO: compute actual path lenght, using best-fit straight line
        #   along the "thinned" path.
        # For now, we estimate it as the max possible lenght in the bounding box, its diagonal.
        w = x2 - x1
        h = y2 - y1
        path_len = math.sqrt(w * w + h * h)
        expected_area = radius * (2 * path_len + math.pi * radius)

        area_ratio = abs(actual_area / expected_area - 1)
        is_fom = area_ratio < gamma
        if is_fom and fom_detected:
            print("[WARN] More than one FOM detected")
        elif is_fom:
            print("FOM DETECTED")
            fom_detected = True
            fom = i
        print("{}: radius: {:.2f}, len: {:.2f}, (A / A^ -1): {:.2f}".format(
            i, radius, path_len, area_ratio))

    frameNumber += 1  # increments the loop

    # Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
    k = cv2.waitKey(0)
    if k == 27:
        break
    elif k == 32:
        i -= 2
    else:
        continue
