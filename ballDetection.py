import numpy as np
import cv2
import imutils
import time
from PIL import Image
import re

def filterSize(inputFrame):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(inputFrame, ltype=cv2.CV_16U)
    candidates = list()
    for stat in stats:
        area = stat[cv2.CC_STAT_AREA]
        lt = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
        rb = (lt[0] + stat[cv2.CC_STAT_WIDTH], lt[1] + stat[cv2.CC_STAT_HEIGHT])

        candidates.append((lt, rb, area))

        return candidates

def drawRectangle(d,candidates):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    fontColor = 127
    lineType = 1

    for candidate in candidates:
        bottomLeftCornerOfText = (candidate[0][0],candidate[0][1]-15)

        cv2.rectangle(d,candidate[0],candidate[1],fontColor,lineType)

        cv2.putText(d,"{}: {:.0f}".format(candidates.index(candidate), candidate[2]),
                bottomLeftCornerOfText, 
                font, fontScale, fontColor, lineType)

    return d
