import face_recognition
import cv2
import os

output_directory = 'Computer_Vision/AttendanceSystemIDs'
os.makedirs(output_directory, exist_ok=True)

input_image_paths = os.listdir('Computer_Vision/AttendanceSystemPreIDs')

for input_image_path in input_image_paths :

    image = face_recognition.load_image_file(f'Computer_Vision/AttendanceSystemPreIDs/{input_image_path}')
    resized_image = cv2.resize(image, (216,216))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    output_image_path = os.path.join(output_directory, f'{os.path.basename(input_image_path)}.jpg')
    cv2.imwrite(output_image_path, resized_image)

