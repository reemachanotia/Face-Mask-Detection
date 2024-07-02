
"""
Created on Mon Feb 26 10:18:15 2024

@author: Admin
"""
'''
CNN-convonutional neural network
'''

# cnn models are used to automatically select the best features from noisy data(ex:images)
# while training and feed forward neural network

'''
environment
'''
#a space to do particular project with special specification





'''
open image ang videp capture
'''
import cv2

img=cv2.imread(r"C:\Users\HP\Desktop\images.jpg")
#here img is in the form of numpy array
cv2.imshow("window",img)#to show image of frame

cv2.waitKey(0)#pause the wndow
#here 0 means infinite time pause and we pass millisecond to show it
cv2.destroyAllWindows()#to remove the window from memory






'''
open webcam
'''
cap=cv2.VideoCapture(0)# to open camera
#0 means port number, phone ka ip address b daal skte h ism hum, if lapi and phn are attached to same wifi


#cap is abuffer where frames are 
while cap.isOpened():
    b,frame=cap.read()
    #read fxn returns two items--boolvalue,Frame
    cv2.imshow("window",frame)
    if cv2.waitKey(1)==113:
        break
    
    
cap.release()
cv2.destroyAllWindows()


#cap=cv2.VideoCapture(index, apiPreference, params)

print(ord("q"))
