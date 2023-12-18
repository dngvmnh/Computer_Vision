from ultralytics import YOLO
import cv2 as cv
import os, random

NEW_LENGTH = 720

folder_path = "C:/Users/dngvm/Projects/Oject Detection/YOLO_intro/Images"
files = os.listdir(folder_path)
random_filename = random.choice(files)
img = cv.imread(f'{folder_path}/{random_filename}')

def resize_image(image, new_length):
 
    height, width = image.shape[:2]
    aspect_ratio = width / float(height)

    if width > height:
        new_width = new_length
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = new_length
        new_width = int(new_height * aspect_ratio)

    resized_image = cv.resize(image, (new_width, new_height))

    return resized_image

model = YOLO('Computer_Vision/YOLO_Weights/yolov8n.pt')
img = resize_image(img, NEW_LENGTH)
results = model(img, show=True)
cv.waitKey(0)