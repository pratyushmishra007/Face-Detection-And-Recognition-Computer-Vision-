import numpy as np
import cv2 as cv
import os

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

haar_cascade=cv.CascadeClassifier(r'C:\Users\pratyush mishra\Desktop\py tutorial\projects(openCV)\haar_face.xml')

# features=np.load('features.npy',allow_pickle=True)
# lables=np.load('lables.npy')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\pratyush mishra\Desktop\py tutorial\face_trained.yml')

img=cv.imread('C:/Users/pratyush mishra/Desktop/py tutorial/projects(openCV)/Faces/val/elton_john/3.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to grayscale
cv.imshow('person',gray)
#Detect the face in the image

faces_rect=haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+w]
    lable,confidence=face_recognizer.predict(faces_roi)
    print(f'lable={people[lable]}with a confidence of{confidence}')

    cv.putText(img,str(people[lable]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected Image',img)

cv.waitKey(0)
