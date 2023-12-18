import cvzone
import cv2
from cvzone.ColorModule import ColorFinder

myColorFinder = ColorFinder(trackBar=False)

cap = cv2.VideoCapture(0)

cap.set(3, 200)
cap.set(4, 100)

hsvVals = {'hmin': 4, 'smin': 0, 'vmin': 234, 'hmax': 116, 'smax': 255, 'vmax': 255}

while True:
    success, img = cap.read()

    imgOrange, mask = myColorFinder.update(img, hsvVals)

    imgContours, conFound = cvzone.findContours(img, mask)

    imgStack = cvzone.stackImages([img, imgOrange, mask, imgContours], 2, 1)

    cv2.imshow("Image Stack", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
