import cv2 as cv
import mediapipe as mp
import time
import os
import random

pTime = 0
video_folder_path = "C:/Users/dngvm/Projects/Computer Vision/Videos"
video_files = os.listdir(video_folder_path)
random_video_filename = random.choice(video_files)
# video_filepath = os.path.join(video_folder_path, random_video_filename)
# capture = cv.VideoCapture(f'{video_folder_path}/{random_video_filename}')
capture = cv.VideoCapture(0)
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while (capture.isOpened()) :
    ret, frame = capture.read()

    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = faceDetection.process(frameRGB)

    if results.detections :
        for id, detection in enumerate(results.detections) : 
            # mpDraw.draw_detection(frame, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                    int(bboxC.width*iw), int(bboxC.height*ih)
            cv.rectangle(frame, bbox, (0,255,0), 2)
            cv.putText(frame, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1] -20), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(frame, str(int(fps)), (50,50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    cv.imshow('Frame', frame)
    cv.waitKey(1)

