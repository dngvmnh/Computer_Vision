import cv2 as cv
import mediapipe as mp
import os
import random
import time

pTime = 0
video_folder_path = "C:/Users/dngvm/Projects/Computer Vision/Videos"
video_files = os.listdir(video_folder_path)
random_video_filename = random.choice(video_files)
# video_filepath = os.path.join(video_folder_path, random_video_filename)
# capture = cv.VideoCapture(f'{video_folder_path}/{random_video_filename}')
capture = cv.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose() 

while (capture.isOpened()) :
    ret, frame = capture.read()
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    if results.pose_landmarks :
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark) :
            h, w, c = frame.shape
            cx, cy = int(lm.x*w), int(lm.y*h )
            cv.circle(frame, (cx,cy), 5, (0,255,0), cv.FILLED)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (50,50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv.imshow("frame", frame)
    cv.waitKey(1)



