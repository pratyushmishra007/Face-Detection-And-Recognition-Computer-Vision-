import cv2 as cv
import os
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

DIR=r'C:\Users\pratyush mishra\Desktop\py tutorial\projects(openCV)\Faces\train'

haar_cascade=cv.CascadeClassifier('C:/Users/pratyush mishra/Desktop/py tutorial/projects(openCV)/haar_face.xml')

features=[]
lables=[]

def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        lable=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)
            if img_array is None:
                continue
            
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            for(x,y,w,h) in face_rect:
                faces_roi=gray[y:y+h,x:x+w]
                features.append(faces_roi)
                lables.append(lable)

create_train()
print("training done------------------------")

# print(f'length of the features={len(features)}')
# print(f'length of the lables={len(lables)}')

features=np.array(features,dtype='object')
lables=np.array(lables)

face_recognizer=cv.face.LBPHFaceRecognizer_create()

face_recognizer.write('C:/Users/pratyush mishra/Desktop/py tutorial/face_trained.yml')

#Train the recognizer on the features list and lable list

face_recognizer.train(features,lables)

np.save('features.npy',features)
np.save('lables.npy',lables)


