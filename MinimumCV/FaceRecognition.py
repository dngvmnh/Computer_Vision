import face_recognition
import os
import cv2 as cv
import pickle

def save_known_faces(known_faces, known_names, filename):
    with open(filename, 'wb') as file:
        pickle.dump((known_faces, known_names), file)

def load_known_faces(filename):
    with open(filename, 'rb') as file:
        known_faces, known_names = pickle.load(file)
    return known_faces, known_names

def name_to_color(name):
    color = [(ord(name.lower())-10)*10 for name in name[:3]]
    return color

def process_img() : 
    KNOWN_FACES_DIR = 'Computer Vision/KnownFaces'
    UNKNOWN_FACES_DIR = 'Computer Vision/UnknownFaces'
    TOLERANCE = 0.5
    FRAME_THICKNESS = 1
    FONT_THICKNESS = 1
    # MODEL = 'hog'  
    MODEL = 'cnn'
    IMG_SIZE = 1000
    KNOWN_FACES_FILE = 'KnownFaces'

    if os.path.exists(KNOWN_FACES_FILE):
        print('Loading pre-saved known faces...')
        known_faces, known_names = load_known_faces(KNOWN_FACES_FILE)
    else:
        print('Loading known faces...')
        known_faces = []
        known_names = []

        for name in os.listdir(KNOWN_FACES_DIR):
            for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
                image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
                image = cv.resize(image, (IMG_SIZE, IMG_SIZE))

                # encoding = face_recognition.face_encodings(image)[0]
                encoding = face_recognition.face_encodings(image)

                if len(encoding) > 0:
                    encoding = encoding[0]

                    known_faces.append(encoding)
                    known_names.append(name)

        save_known_faces(known_faces, known_names, KNOWN_FACES_FILE)


    print('Processing unknown faces...')

    for filename in os.listdir(UNKNOWN_FACES_DIR):

        print(f'Filename {filename}', end='')

        image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
        image = cv.resize(image, (IMG_SIZE, IMG_SIZE)) 
        locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        print(f', found {len(encodings)} face(s)')

        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            match = None
            if True in results:
                match = known_names[results.index(True)]

                print(f'{match}')

                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                color = name_to_color(match)
                cv.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
                
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 20)

                cv.rectangle(image, top_left, bottom_right, color, cv.FILLED)
                cv.putText(image, match, (face_location[3] + 3, face_location[2] + 13), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), FONT_THICKNESS)
                 
        cv.imshow(filename, image)
        cv.waitKey(0)
        cv.destroyWindow(filename)

def process_vid() : 

    KNOWN_FACES_DIR = 'Computer Vision/KnownFaces'
    TOLERANCE = 0.5
    FRAME_THICKNESS = 1
    FONT_THICKNESS = 1
    MODEL = 'hog'  
    # MODEL = 'cnn'
    IMG_SIZE = 1000
    KNOWN_FACES_FILE = 'KnownFaces'

    video = cv.VideoCapture(0)

    if os.path.exists(KNOWN_FACES_FILE):
        print('Loading pre-saved known faces...')
        known_faces, known_names = load_known_faces(KNOWN_FACES_FILE)
    else:
        print('Loading known faces...')
        known_faces = []
        known_names = []

        for name in os.listdir(KNOWN_FACES_DIR):
            for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
                image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
                image = cv.resize(image, (IMG_SIZE, IMG_SIZE))

                # encoding = face_recognition.face_encodings(image)[0]
                encoding = face_recognition.face_encodings(image)

                if len(encoding) > 0:
                    encoding = encoding[0]

                    known_faces.append(encoding)
                    known_names.append(name)

        save_known_faces(known_faces, known_names, KNOWN_FACES_FILE)


    print('Processing unknown faces...')



    while video.isOpened() : 

        # print(f'Filename {filename}', end='')

        # image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
        # image = cv.resize(image, (IMG_SIZE, IMG_SIZE)) 

        ret, image = video.read()

        locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)
        # image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        print(f', found {len(encodings)} face(s)')

        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            match = None
            if True in results:
                match = known_names[results.index(True)]

                print(f'{match}')

                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                color = name_to_color(match)
                cv.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
                
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 20)

                cv.rectangle(image, top_left, bottom_right, color, cv.FILLED)
                cv.putText(image, match, (face_location[3] + 3, face_location[2] + 13), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), FONT_THICKNESS)
                
                    
        cv.imshow('Frame', image)
        if cv.waitKey(1) == ord("q") :
            break

if __name__ == '__main__' : 
    # process_img()
    process_vid()
