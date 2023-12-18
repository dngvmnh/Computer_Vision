import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0 

while (capture.isOpened()) :
    ret, frame = capture.read()
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks :
        for hand_lm in results.multi_hand_landmarks :
            for id, lm in enumerate(hand_lm.landmark) :
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 0 : 
                    cv.circle(frame, (cx, cy), 10, (0,255,0), cv.FILLED)
            mpDraw.draw_landmarks(frame, hand_lm, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (50,50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv.imshow("frame", frame)
    cv.waitKey(1)