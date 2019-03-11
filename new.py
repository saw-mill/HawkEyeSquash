import glob
import time
import cv2
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
        # x,y,w,h=cv2.boundingRect(cnt)
        # rectArea=w*h
        # extent=float(area)/rectArea
        perimeter = cv2.arcLength(cnt, True)
        candidates.append((cX, cY, area, perimeter))
        cv2.drawContours(currFrame, [cnt], -1, (0, 255, 0), 1)
        cv2.putText(currFrame, str(area), (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('Candidate image', currFrame)
        # cv2.imshow('Final image {}'.format(i),threshFrameDifferencing)
    print(len(candidates))

    # startTimeSizeFilter=time.time()
    # num, labels, stats, centroids = cv2.connectedComponentsWithStats(final_image, ltype=cv2.CV_16U)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale = 0.6
    # fontColor = 127
    # lineType = 1

    # min_area = 100
    # max_area = 1500

    # d= final_image.copy()
    # candidates = list()
    # for stat in stats:
    # 	area = stat[cv2.CC_STAT_AREA]
    # 	if area > max_area or area < min_area:
    # 		continue # Skip big objects (players) and skip small objects (noisy candidates)

    # 	lt = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
    # 	rb = (lt[0] + stat[cv2.CC_STAT_WIDTH], lt[1] + stat[cv2.CC_STAT_HEIGHT])
    # 	bottomLeftCornerOfText = (lt[0], lt[1] - 15)

    # 	candidates.append((lt, rb, area))
    # 	cv2.rectangle(d, lt, rb, fontColor, lineType)

    # 	cv2.putText(d, "{}: {:.0f}".format(len(candidates), stat[cv2.CC_STAT_AREA]),
    # 				bottomLeftCornerOfText,
    # 				font, fontScale, fontColor, lineType)
    # endTimeSizeFilter=time.time()
    # print("Size Filter--- %s seconds ---" % (endTimeSizeFilter - startTimeSizeFilter))

    # cv2.imshow('thresh', threshFrameDifferencing)
    # cv2.imshow('erosion', img_erosion)

    # cv2.imshow('Final image', currFrame)
    
    # print(len(candidates))
    # cv2.imshow('rectangles',d)

    i += 1  # increments the loop

    # Exits the loop when Esc is pressed, goes to previous frame when space pressed and goes to next frame when any other key is pressed
    k = cv2.waitKey(0)
    if k == 27:
        break
    elif k == 32:
        i -= 2
    else:
        continue
