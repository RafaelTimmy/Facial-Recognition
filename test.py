import cv2
import numpy as np
import face_recognition

imgAg = face_recognition.load_image_file("DATA/ARIANA GRANDE.jpg")
imgAg = cv2.cvtColor(imgAg, cv2.COLOR_BGR2RGB)
imgAgTest = face_recognition.load_image_file("DATA/BARACK OBAMA.jpg")
imgAgTest = cv2.cvtColor(imgAgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgAg)[0]
encodeAg = face_recognition.face_encodings(imgAg)[0]
cv2.rectangle(imgAg,(faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]), (255,0,255), 2)

faceLocTest = face_recognition.face_locations(imgAgTest)[0]
encodeAgTest = face_recognition.face_encodings(imgAgTest)[0]
cv2.rectangle(imgAgTest,(faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]), (255,0,255), 2)

results = face_recognition.compare_faces([encodeAg], encodeAgTest)
faceDis = face_recognition.face_distance([encodeAg], encodeAgTest)
print(results, faceDis)
cv2.putText(imgAgTest, f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('ARIAN GRANDE', imgAg)
cv2.imshow('ARIAN Test', imgAgTest)
cv2.waitKey(0)

