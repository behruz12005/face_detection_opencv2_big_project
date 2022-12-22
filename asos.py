#pylint:disable=no-member

import numpy as np
import cv2 as cv
import pandas as pd
from datetime import datetime

now = datetime.now()
Data = now.strftime("%d/%m/%Y")
Time = now.strftime("%H:%M:%S")

# DataBase
rows = []
df = pd.read_csv('../mohirfest/cvs_file/Mohir.csv')

# Train qilingan kodni olib predicct berish qismi



haar_cascade = cv.CascadeClassifier('hear_face.xml')

people = ['Behruz','ramazon']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# BU yerda sinab kurish


cap = cv.VideoCapture(0)
stop = True
while stop:
    ret, frame = cap.read()
    faces_rect = haar_cascade.detectMultiScale(frame, 1.1, 4)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')
        if people[label] in people and confidence < 40:



            if len(df[df.Name == people[label]]) == 0:
                df.loc[len(df)] = [people[label],Data,Time,'11B']
            else:
                print("Sen oldin kiritilgasan")
            stop = False


        cv.putText(frame, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv.imshow('Detected Face', frame)

    if cv.waitKey(1) == ord('q'):
        stop = False




