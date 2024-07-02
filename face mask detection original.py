# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:40:32 2024

@author: Admin
"""

import cv2
import numpy as np
import os


import tensorflow as tf
print(tf.__version__)
print(cv2.__version__)
import platform
print(platform.system(), platform.release())




os.chdir(r"C:/Users/Admin/Desktop/phase mask")
model=tf.keras.models.load_model("hunnymodel.h5")

haarcascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



#Haarcascade--->
#in haarcascade classififer it stores the features of images with that ibject in the form of
#pixel corrlatin in an xml file







'''
open webcam
'''
cap=cv2.VideoCapture(0)# to open camera
#0 means port number, phone ka ip address b daal skte h ism hum, if lapi and phn are attached to same wifi


#cap is abuffer where frames are stored
while cap.isOpened():
    b,frame=cap.read()
    
    faces=haarcascade.detectMultiScale(frame,scaleFactor=1.10,minNeighbors=4)
    
    #this function is used to find all cordinates where there is face
    #it returms list of coordinates(x,y,w,h)
    
    for x,y,w,h in faces:
        face=frame[y:y+h,x:x+w]
        cv2.imwrite('face.jpg',face)
        face=tf.keras.preprocessing.image.load_img("face.jpg",target_size=(150,150,3))
        #converting format of face into keras format

        
        face=tf.keras.preprocessing.image.img_to_array(face)
        #converting  image into numpy array
        
        #converting images into 4d
        
        face=np.expand_dims(face,axis=0)
        #(150,150,3)-->(1,150,150,3)

        
        
        ann=model.predict(face)
        
        if ann >0.5:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,"without mask",(x//2,y//2),0,2,(0,0,255),3)
        else:
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,"with mask",(x//2,y//2),0,2,(0,255,0),3)
            
        
    
    cv2.imshow("window",frame)
    if cv2.waitKey(1)==113:
        
        
        
        
        
        break
    
    
cap.release()
cv2.destroyAllWindows()


