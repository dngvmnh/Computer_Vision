import cv2 as cv
import mediapipe as mp
import time
import os

wCam, hCam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# folderPath = ""
# myList = os.listdir(folderPath)
# overlayList = []
# for imPath in myList:
#     image = cv.imread(f'{folderPath}/{imPath}')
#     # print(f'{folderPath}/{imPath}')
#     overlayList.append(image)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0 
tipIds = [4, 8, 12, 16, 20]

while (cap.isOpened()):
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    res = hands.process(imgRGB)

    lmList = []
    if res.multi_hand_landmarks :
        for hand_lm in res.multi_hand_landmarks :
            for id, lm in enumerate(hand_lm.landmark) :
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        totalFingers = fingers.count(1)

        # h, w, c = overlayList[totalFingers - 1].shape
        # img[0:h, 0:w] = overlayList[totalFingers - 1]

        cv.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv.FILLED)
        cv.putText(img, str(totalFingers), (45, 375), cv.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)
        
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS: {int(fps)}', (400, 70), cv.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv.imshow("Image", img)
    cv.waitKey(1)