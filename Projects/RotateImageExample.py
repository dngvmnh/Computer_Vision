import cv2
from cvzone.Utils import rotateImage 

cap = cv2.VideoCapture(0)

while True:
    
    success, img = cap.read() 

    imgRotated60 = rotateImage(img, 60, scale=0.2, keepSize=False) 

    imgRotated60KeepSize = rotateImage(img, 60, scale=1, keepSize=True) 

    cv2.imshow("img", img) 
    cv2.imshow("imgRotated60", imgRotated60)  
    cv2.imshow("imgRotated60KeepSize", imgRotated60KeepSize) 

    cv2.waitKey(1) 
    