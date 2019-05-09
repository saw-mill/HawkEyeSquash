import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('DatasetVideos/SRally3.mp4')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
  ret1, frame1 = cap.read()
  ret2, frame2 = cap.read()
  ret3, frame3 = cap.read()
  if ret1 == True and ret2 == True and ret3 == True:
 
    # Display the resulting frame
    cv2.imshow('Frame1', frame1)
    cv2.imshow('Frame2', frame2)
    cv2.imshow('Frame3', frame3)
 
    # Press Q on keyboard to  exit
    k = cv2.waitKey(0)
    if k == 27:
        break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()