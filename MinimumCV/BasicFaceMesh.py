import cv2 as cv
import mediapipe as mp
import time
import os
import random

pTime = 0
video_folder_path = "C:/Users/dngvm/Projects/Computer Vision/Videos"
# video_files = os.listdir(video_folder_path)
# random_video_filename = random.choice(video_files)
# video_filepath = os.path.join(video_folder_path, random_video_filename)
# capture = cv.VideoCapture(f'{video_folder_path}/{random_video_filename}')
capture = cv.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)

while (capture.isOpened()) :
    ret, frame = capture.read()

    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)

    if results.multi_face_landmarks : 
        for face_lm in results.multi_face_landmarks :
            mpDraw.draw_landmarks(frame, face_lm, mpFaceMesh.FACEMESH_CONTOURS,  drawSpec, drawSpec)
            for id, lm in enumerate(face_lm.landmark) : 
                ih, iw, ic = frame.shape
                x, y = int(lm.x*iw), int(lm.y*ih)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(frame, str(int(fps)), (50,50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    cv.imshow('Frame', frame)
    cv.waitKey(1)
