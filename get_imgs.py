import numpy as np
import cv2
import os
from pathlib import Path
import codecs
import os.path
from os import path
import time  #we'll create a timer so that when one minute passes the window closes
import sys


home = str(Path.home())

mesImages = home + "\\Pictures"

datays = mesImages + "/data.ys"

with codecs.open(datays,"r","utf-8") as f : 
        lines = f.readlines()



ImgNum = 0


DIR = lines[0];

DIRSize = len(DIR)-1;

DIR = DIR[:DIRSize];  #remove the \n 



start_time = time.time()

end_time = 0

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


cap = cv2.VideoCapture(0)

def saveImage(id):

    id = str(id)


    while (True):
        
        ret, frame = cap.read()

        if lines[2] != 0:
            
            ImgNum = int(lines[2])  # what ever value we stopped at it
        end_time =time.time()
        if ( ( (end_time - start_time) ) > 60):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()    
        try:
            imggray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
            faces = faceCascade.detectMultiScale(imggray,scaleFactor =1.5, minNeighbors=5)

            for(x,y,w,h) in faces :

                roi_color = frame[y:y+h, x:x+w] #rgb frame with a rectangle
                
                ImgNum+=1
                
                lines[2] = ImgNum;
               
                img_name =  id + str(ImgNum) + '.png'

               
                os.chdir(DIR)


                cv2.imwrite(img_name, roi_color)
                
                print(os.path.join(DIR,r'{}'.format(img_name)))
                color = (255,0,0)
                stroke = 2
                x_cord = x + w
                y_cord = y + h
                cv2.rectangle(frame, (x,y),(x_cord,y_cord), color, stroke)
            cv2.imshow('Enregistrement des photos', frame)
            KeyCode = cv2.waitKey(1)
            
        except Exception as e:
               print(e)
               cap.release()
               cv2.destroyAllWindows()
               sys.exit()

        if cv2.getWindowProperty('Enregistrement des photos', cv2.WND_PROP_VISIBLE) <1:
           cap.release()
           cv2.destroyAllWindows()
           sys.exit()

    
            
    cap.release()
    cv2.destroyAllWindows()



while True :


        StudentName =  str(lines[1])  # the line with the student name
        StudentName = StudentName[:len(StudentName)-1]
        StudentName.lower()

        saveImage(StudentName)



















