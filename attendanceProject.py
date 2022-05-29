import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = "imagetest"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)



def findEncodings(images):
    encodeList = []
    registered_counter = 0
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(face_recognition.face_encodings(img)):
            print("registered: ",classNames[registered_counter])
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            registered_counter += 1
    return encodeList

def markAttendance(name):


encodeListKnown = findEncodings(images)
print("----Encoding Complete----")


# Using the Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,0),2)
            cv2.rectangle(img,(x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name,(x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)




# faceLoc = face_recognition.face_locations(imgAg)[0]
# encodeAg = face_recognition.face_encodings(imgAg)[0]
# cv2.rectangle(imgAg,(faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]), (255,0,255), 2)
#
# faceLocTest = face_recognition.face_locations(imgAgTest)[0]
# encodeAgTest = face_recognition.face_encodings(imgAgTest)[0]
# cv2.rectangle(imgAgTest,(faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]), (255,0,255), 2)
#
# results = face_recognition.compare_faces([encodeAg], encodeAgTest)
# faceDis = face_recognition.face_distance([encodeAg], encodeAgTest)