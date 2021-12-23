import os
from PIL import Image
import numpy as np
import cv2
import pickle
import codecs
import sys
from pathlib import Path

home = str(Path.home())


mesImages =  home + "/Pictures"

datays = mesImages + "/data.ys"

with codecs.open(datays,'r','utf8') as f :
    lines = f.readlines()


B_DIR = lines[0]   #path


B_DIRSize = len(B_DIR)-1;

B_DIR = B_DIR[:B_DIRSize];  #remove the \n

lines.append('\n none')
#split directory to remove
ParentDIR, ChildDIR = os.path.split(B_DIR)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
y_labels = []
x_train = []
label_ids ={}


try:
    for root, dirs, files in os.walk(ParentDIR):
        for file in files :
            if file.endswith("png") or file.endswith("jpg"):
                lines[len(lines)-1] = file


                    
                path = os.path.join(root,file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id+=1

                id_ = label_ids[label]

                pil_image = Image.open(path).convert("L") #grayscale conv
                size = (550,550)
                final_img  = pil_image.resize(size,Image.ANTIALIAS)
                image_array = np.array(final_img,"uint8")

                faces = faceCascade.detectMultiScale(image_array,scaleFactor =1.5, minNeighbors=1)


                for(x,y,w,h) in faces:
                    roi = image_array[y:y+h,x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    labelys = mesImages + "/labels.pickle"
    with open(labelys,"wb") as f :
        pickle.dump(label_ids, f)

    recognizer.train(x_train,np.array(y_labels))
    
    os.chdir(mesImages)
    recognizer.save("./trainner.yml")

except Exception:
    print(Exception)
    sys.exit()
