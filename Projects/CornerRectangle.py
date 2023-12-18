import cv2
import cvzone

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    cv2.rectangle(img, (200, 200), (500, 400), (255, 0, 255), 3)

    cvzone.cornerRect(img, (200, 200, 300, 200))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
