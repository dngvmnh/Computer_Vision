import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cv2

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

segmentor = SelfiSegmentation(model = 1)  #   0 or 1

while True:
    success, img = cap.read()

    imgOut = segmentor.removeBG(img, imgBg=(0, 255, 0), cutThreshold=0.3)

    imgStacked = cvzone.stackImages([img, imgOut], cols=2, scale=1)

    cv2.imshow("Image", imgStacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
