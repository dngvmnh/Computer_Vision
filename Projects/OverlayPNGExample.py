import cv2
import cvzone

cap = cv2.VideoCapture(0)

imgPNG = cvzone.downloadImageFromUrl(
    url='https://github.com/cvzone/cvzone/blob/master/Results/cvzoneLogo.png?raw=true',
    keepTransparency=True)

imgPNG = cv2.imread("Oject_Detection/freeCodeCamp Projects/cvzoneLogo.png",cv2.IMREAD_UNCHANGED)

while True :
    
    success, img = cap.read()

    imgOverlay = cvzone.overlayPNG(img, imgPNG, pos=[-30, 50])
    imgOverlay = cvzone.overlayPNG(img, imgPNG, pos=[200, 200])
    imgOverlay = cvzone.overlayPNG(img, imgPNG, pos=[500, 400])

    cv2.imshow("imgOverlay", imgOverlay)
    cv2.waitKey(1)