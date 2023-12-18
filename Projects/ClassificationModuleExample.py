from cvzone.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture(0)  
maskClassifier = Classifier(f'', f'')

while True:
    ret, img = cap.read() 
    prediction = maskClassifier.getPrediction(img)
    print(prediction) 
    cv2.imshow("Image", img)
    cv2.waitKey(1)  