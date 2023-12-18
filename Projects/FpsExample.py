import cvzone
import cv2
from cvzone.FPS import FPS

fpsReader = FPS()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60) 

while True:
    success, img = cap.read()

    fps, img = fpsReader.update(img, pos=(20, 50),
                                bgColor=(255, 0, 255), textColor=(255, 255, 255),
                                scale=3, thickness=3)
    
    print(fps)

    fps, img = fpsReader.update(img)

    cv2.imshow("Image", img)

    cv2.waitKey(1)