import numpy as np
import cv2
import imutils
import time

image1= cv2.imread("Dataset/frame7.jpg")
image2= cv2.imread("Dataset/frame9.jpg")

image3= cv2.medianBlur(image1,7)
image4= cv2.medianBlur(image2,7)

image3Gray= cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)
image4Gray= cv2.cvtColor(image4,cv2.COLOR_BGR2GRAY)
# cv2.imshow('GreyScale versions',np.hstack([image3Gray,image4Gray]))
startTimeFrameDifferencing= time.time()
frame_diff=cv2.absdiff(image3Gray,image4Gray)

# startTimeImageDifferencing= time.time()
# (score, image_diff) =compare_ssim(image3Gray,image4Gray,full=True)
# image_diff= (image_diff*255).astype("uint8")
# print("--- %s seconds ---" % (time.time() - startTimeImageDifferencing)) 

threshFrameDifferencing = cv2.threshold(frame_diff, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

filteredThreshFrameDifferencing= cv2.medianBlur(threshFrameDifferencing,7)

kernelDilation = np.ones((7,7), np.uint8)
kernelErosion = np.ones((3,3), np.uint8)
img_dilation = cv2.dilate(filteredThreshFrameDifferencing, kernelDilation, iterations=1) 
img_erosion = cv2.erode(img_dilation, kernelErosion, iterations=2) 

# threshImageDifferencing = cv2.threshold(image_diff, 0, 255,
# 	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# cnts = cv2.findContours(threshFrameDifferencing.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# for c in cnts:
# 	# compute the bounding box of the contour and then draw the
# 	# bounding box on both input images to represent where the two
# 	# images differ
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	cv2.rectangle(threshFrameDifferencing, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# cv2.rectangle(image4, (x, y), (x + w, y + h), (0, 0, 255), 2)

# cv2.imshow('Frame 1',image3)
# cv2.imshow('Frame 2',image4)
# cv2.imshow('Frame Differencing',frame_diff)
# cv2.imshow('Image Differencing',image_diff)
cv2.imshow('Thresholded image Frame differencing',threshFrameDifferencing)
cv2.imshow('Filtered Thresholded Frame Differencing',filteredThreshFrameDifferencing)
cv2.imshow('Erosion',img_erosion)
cv2.imshow('Dilation',img_dilation)
# cv2.imshow('Thresholded image Image differencing',threshImageDifferencing)
print("--- %s seconds ---" % (time.time() - startTimeFrameDifferencing)) 

cv2.waitKey(0)