import cv2           
import cvzone       
import numpy as np  


# imgShapes = cvzone.downloadImageFromUrl(url='https://github.com/cvzone/cvzone/blob/master/Results/shapes.png?raw=true')
imgShapes = cv2.imread("Computer_Vision/freeCodeCamp Projects/shapes.png")

imgCanny = cv2.Canny(imgShapes, 50, 150)
imgDilated = cv2.dilate(imgCanny, np.ones((5, 5), np.uint8), iterations=1)
imgContours, conFound = cvzone.findContours(imgShapes, imgDilated)

imgContours, conFound = cvzone.findContours(
    imgShapes, imgDilated, minArea=1000,maxArea=100000, sort=True,
    filter=None, drawCon=True, c=(255, 0, 0), ct=(255, 0, 255),
    retrType=cv2.RETR_EXTERNAL, approxType=cv2.CHAIN_APPROX_NONE)

imgContoursFiltered, conFoundFiltered = cvzone.findContours(
    imgShapes, imgDilated, minArea=1000, maxArea=100000, sort=True,
    filter=[8], drawCon=True, c=(255, 0, 0), ct=(255, 0, 255),
    retrType=cv2.RETR_EXTERNAL, approxType=cv2.CHAIN_APPROX_NONE)

cv2.imshow("imgContours", imgContours)

cv2.imshow("imgContoursFiltered", imgContoursFiltered)

cv2.waitKey(0)
