import numpy as np
import argparse
import cv2

videoCapture = cv2.VideoCapture('SquashRally2_EDIT.mp4')

totalFrames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
print(totalFrames)

myframe = 60
count =0

videoCapture.set(cv2.CAP_PROP_POS_FRAMES,myframe)

while(videoCapture.isOpened()):
    retValue, frame = videoCapture.read()
    if retValue == True:
        # cv2.imshow('Frame',frame) 
        cv2.imwrite("Dataset/%d.jpg" % count,frame)
        count=count+1

        if cv2.waitKey(1000) & 0xFF == ord('x'):
            break

    else:
        break

videoCapture.release()

cv2.destroyAllWindows()