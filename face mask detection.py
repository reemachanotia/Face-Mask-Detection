# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:17:22 2024

@author: Admin
"""

import tensorflow as tf
print(tf.__version__)


#mask detection system(yes/No)--Classification
#object detection system
#collect images of each object
#make test and train folder
#make folder of each class in both test and train

#image data generator
#cnn
#training

'''
data augmentatioin------is concept of making more data from previous data
'''



'''
image datagenerators---- will make the batches of images by augementing them
'''

train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2,rotation_range=0.2,shear_range=0.2,rescale=1/255)

test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_dataset=train_datagen.flow_from_directory("train",target_size=(150,150),
                                                class_mode='binary',batch_size=16)



test_dataset=test_datagen.flow_from_directory("test",target_size=(150,150),
                                                class_mode='binary',batch_size=16)



'''
building CNN Model
'''
cnn=tf.keras.models.Sequential()



#kernel--it is a non matrix filter which is convolved over
# images and extract particular feature from that image
#kernel results in feature map


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,input_shape=(150,150,3),activation='relu'))

#when kernel size is small then feature map will contain smaller info.also which
# also result in slow speed.
# when kernel size is big then speed will fast but accuracy can be low


'''
pooling----reducing the size of feature map by taking average, 
max of each window in feature
'''


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#we can also make features from prewound features

#now adding next cnn layer

cnn.add(tf.keras.layers.Conv2D(filters=16,kernel_size=3,activation='relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))




'''
flatten layer---is responsible to convert data into id so that to feed up hidden layer
'''
cnn.add(tf.keras.layers.Flatten())


cnn.add(tf.keras.layers.Dense(units=12,activation='relu'))

cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

'''
compiling
'''
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics="accuracy")

'''
fit
'''

cnn.fit(train_dataset,validation_data=test_dataset,epochs=50)



cnn.save("reemamodel.h5")














