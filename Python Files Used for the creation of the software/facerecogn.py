import cv2
import numpy as np
import pickle
import sys
import os
from pathlib import Path

home = str(Path.home())

mesImages = home + "\\Pictures"


cap= cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

os.chdir(mesImages)

recognizer.read("trainner.yml")

# get the folder name :
labels = {}


labels={}

labelpickle = mesImages + "/labels.pickle"

with open(labelpickle,"rb") as f :
     old_labels = pickle.load(f)
     labels = { v:k for k,v in old_labels.items()}
try:
    while (True):
        ret, frame = cap.read()

        imggray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imggray,scaleFactor =1.5, minNeighbors=1)

        for(x,y,w,h) in faces :

            roi_gray = imggray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w] #rgb frame with a rectangle




            id_, conf = recognizer.predict(roi_gray)
            if conf>=45 and conf<= 85:
                font= cv2.FONT_HERSHEY_DUPLEX
                name = labels[id_]
                color = (255,255,255)
                stroke = 2


                #text handling :
                
                if (len(name) > 8 ) : #when the name is too long, we shorten it in ImShow()
                    name = name[:6] + '...'
                    
                name = name.replace('-',' ')

                width = x + w
                height = y + h
                
                cv2.putText(frame, name,(x,y-3),font,1,color,stroke)

                color = (255,160,0)
                stroke = 2
                cv2.rectangle(frame,(x,y),(width,height),color,stroke)

        cv2.imshow('Face Recognition', frame)
        KeyCode = cv2.waitKey(1) 


        if cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_VISIBLE) <1:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()


    cap.release()
    cv2.destroyAllWindows()
    sys.exit()
except Exception :
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()
