import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("Computer_Vision/AttendanceSystem/attendancesystem-974ac-firebase-adminsdk-izwto-9d65d3530d.json")
firebase_admin.initialize_app(cred, {'databaseURL': "https://attendancesystem-974ac-default-rtdb.firebaseio.com/", 'storageBucket': "attendancesystem-974ac.appspot.com"})

def main() : 

    DURATION = 5
    DISPLAY_DURATION = 60

    bucket = storage.bucket()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    imgBackground = cv2.imread('Computer_Vision/AttendanceSytemSource/background.png')
    modePathList = os.listdir('Computer_Vision/AttendanceSytemSource')

    imgModeList = []
    for path in modePathList:
        imgModeList.append(cv2.imread(os.path.join('Computer_Vision/AttendanceSytemSource', path)))


    if os.path.exists('EncodeFile'):
        print('Loading pre-saved encode')
        with open('EncodeFile', 'rb') as file:
            encodeListKnown, studentIds = pickle.load(file)
    else:
        print('Loading new encode')
        imgList = []
        encodeList = []
        studentIDs = []
        IDs_path = os.listdir('Computer_Vision/AttendanceSystemIDs')

        for ID in IDs_path :
            imgList.append(cv2.imread(f'Computer_Vision/AttendanceSystemIDs/{ID}'))
            studentIDs.append(os.path.splitext(ID)[0])

        for img in imgList :
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)
            if len(encode) > 0:
                        encode = encode[0]
                        encodeList.append(encode)

        encodeListKnown = encodeList
        encodeListKnownWithIds = [encodeListKnown, studentIDs]
        file = open('EncodeFile', 'wb')
        pickle.dump(encodeListKnownWithIds, file)
        file.close()

    modeType = 0
    counter = 0
    imgStudent = []

    while True:
        success, img = cap.read()

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        imgBackground[162:162 + 480, 55:55 + 640] = img
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        if faceCurFrame:

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                matchIndex = np.argmin(faceDis)
                index = list(zip(faceDis, studentIds))
                matchFound = min(index, key=lambda t: t[0])

                if matches[matchIndex]:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    id = matchFound[1]
                    if counter == 0:
                        cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                        cv2.imshow("Face Attendance", imgBackground)
                        cv2.waitKey(1)
                        counter = 1
                        modeType = 1

            if counter != 0:
                if counter == 1:
                    id_name = os.path.splitext(id)[0]
                    studentInfo = db.reference(f'Students/{id_name}').get()
                    blob = bucket.get_blob(f'Computer_Vision/AttendanceSystemIDs/{id}.jpg')
                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                    imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                    datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                    if secondsElapsed > DURATION:
                        ref = db.reference(f'Students/{id_name}')
                        studentInfo['total_attendance'] += 1
                        year = ref.child('total_attendance').get()
                        year = int(year/365)+1
                        ref.child('year').set(year)     
                        ref.child('total_attendance').set(studentInfo['total_attendance'])
                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modeType = 3
                        counter = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if modeType != 3:

                    if DISPLAY_DURATION < counter < DISPLAY_DURATION + 60:
                        modeType = 2

                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                    if counter <= DISPLAY_DURATION:
                        cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(id_name), (1006, 493),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                        (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2
                        cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                        imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

                    counter += 1

                    if counter >= DISPLAY_DURATION + 60:
                        counter = 0
                        modeType = 0
                        studentInfo = []
                        imgStudent = []
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
        else:
            modeType = 0
            counter = 0
        cv2.imshow("Face Attendance", imgBackground)
        cv2.waitKey(1)

def addData2DB() :

    ref = db.reference('Students')

    data = {
        "220201":
            {
                "name": "Lu Duy Anh",
                "major": "Biology",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "G",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220202":
            {
                "name": "Nguyen Ngoc Minh Anh",
                "major": "History",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "A",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220203":
            {
                "name": "Nguyen Tram Anh",
                "major": "Brain Health",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "A",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220204":
            {
                "name": "To Tuyet Anh",
                "major": "Data Science",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "J",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220205":
            {
                "name": "Pham Quoc Bao",
                "major": "Riot Games",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "P",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220206":
            {
                "name": "Truong Thai Bao",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "G",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220207":
            {
                "name": "Nguyen Thanh Cong",
                "major": "Computer Science",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "T",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220208":
            {
                "name": "Lam Hieu Duy",
                "major": "Tencent",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "G",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220209":
            {
                "name": "Nguyen Quoc Duong",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "J",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220210":
            {
                "name": "Nguyen Phat Dat",
                "major": "Homeless",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "G",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220211":
            {
                "name": "Vu Minh Dang",
                "major": "Computer Science",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "A",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220212":
            {
                "name": "Dinh Minh Bao Han",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "E",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220213":
            {
                "name": "Ngo Thanh Hung",
                "major": "Mathematics - Language",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "E",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220214":
            {
                "name": "To Quynh Huong",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "G",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220216":
            {
                "name": "Do Quoc Khanh",
                "major": "Environment",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "T",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220217":
            {
                "name": "Mai Viet Minh Khoi",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "P",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220218":
            {
                "name": "Truong Cong Minh Khue",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "O",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220219":
            {
                "name": "Huynh Dat Kien",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "J",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220220":
            {
                "name": "Duong Hong Lien",
                "major": "Riot Games",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "A",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220221":
            {
                "name": "Pham Tu Man",
                "major": "English",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "G",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220222":
            {
                "name": "Nguyen Kim Hoang Nam",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "O",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220223":
            {
                "name": "Ngo Man Nghi",
                "major": "Robotics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "E",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220224":
            {
                "name": "Dang Bao Ngoc",
                "major": "Bus",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "E",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220227":
            {
                "name": "Vo Nguyen Thanh Nhan",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "G",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220228":
            {
                "name": "Tran Ngoc Thao Nhi",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "E",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220230":
            {
                "name": "Ho Tan Phuc",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "P",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220231":
            {
                "name": "Duong Van Tai",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "E",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220232":
            {
                "name": "Ly Tran Thai",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "E",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220233":
            {
                "name": "Ngyen Minh Thien",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "O",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220234":
            {
                "name": "Tran Ngoc Diem Thy",
                "major": "Mathematics",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "G",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220235":
            {
                "name": "Nguyen Ngoc Thuy Tien",
                "major": "English",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "A",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220236":
            {
                "name": "Bui Minh Truc",
                "major": "Biology",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "E",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220237":
            {
                "name": "Nguyen Thao Uyen",
                "major": "Biology",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "O",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220238":
            {
                "name": "Bui Duc Vinh",
                "major": "Trash",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "T",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
        "220239":
            {
                "name": "Le Win",
                "major": "SAT",
                "starting_year": 2022,
                "total_attendance": 0,
                "standing": "E",    #T-Terrible  P-Poor  J-Just  A-Average  G-Good  E-Excellent  O-Outstanding
                "year": 1,
                "last_attendance_time": "2023-01-01 00:00:00"
            },
    }

    for key, value in data.items():
        ref.child(key).set(value)

def addImage2DB() :

    folderPath = 'Computer_Vision/AttendanceSystemIDs'
    pathList = os.listdir(folderPath)
    for path in pathList:
        fileName = f'{folderPath}/{path}'
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.upload_from_filename(fileName)

if __name__ == '__main__' :
    # addData2DB()
    # addImage2DB()
    main()