from ultralytics import YOLO
import cv2 as cv
import cvzone
import math

cap = cv.VideoCapture(0) 
cap.set(3, 1280)
cap.set(4, 720)
# cap = cv.VideoCapture("Oject Detection/YOLO_intro/Videos/ppe-1.mp4") 
# cap = cv.VideoCapture("Oject Detection/YOLO_intro/Videos/ppe-2.mp4") 
# cap = cv.VideoCapture("Oject Detection/YOLO_intro/Videos/ppe-3.mp4") 

model = YOLO("Computer_Vision/PPEDetection/ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)
while True:
    ret, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if conf>0.5:
                if currentClass =='NO-Hardhat' or currentClass =='NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0,255)
                elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask":
                    myColor =(0,255,0)
                else:
                    myColor = (0, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor, offset=5)
                cv.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv.imshow("Image", img)
    cv.waitKey(1)
