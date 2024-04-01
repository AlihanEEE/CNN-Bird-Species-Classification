import cv2
import numpy as np
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from CustomDataGen import CustomDataGen
import sys

train_image_size = (224, 224)
X_train, X_test, y_train, y_test = [], [], [], []

# Open a file
# This would print all the files and directories
counter_1=0
folders = glob("archive\\train\\*")

for folder in folders[:100]:
    for image_p in glob(folder+"\\*.jpg"): 
        X_train.append (image_p) 
        one_hot=np.zeros(len(folders[:100]))
        one_hot[counter_1]=1
        y_train.append(one_hot)
    counter_1+=1
        

counter_2=0
folders = glob("archive\\test\\*")

for folder in folders[:100]:
    for image_p in glob(folder+"\\*.jpg"): 
        X_test.append (image_p) 
        one_hot=np.zeros(len(folders[:100]))
        one_hot[counter_2]=1
        y_test.append(one_hot)
    counter_2+=1

X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

print(f'X_train shape:{X_train.shape} - X_test shape:{X_test.shape} - y_train shape:{y_train.shape} - y_test shape:{y_test.shape}' )

traingen = CustomDataGen(X_train, y_train, img_size=train_image_size,augmentation=3, shuffle=True)

valgen = CustomDataGen(X_test, y_test, img_size=train_image_size, shuffle=False)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 5,1, input_shape=(224,224,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 5,1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 5,1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, 5,1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.BatchNormalization(axis=bn_axis),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    # tf.keras.layers.BatchNormalization(axis=bn_axis),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='softmax')

])

early_stop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=3, verbose=0,
    mode='auto'
)

filepath = "training_2/" +input("Model name: ")+".hdf5"

modelCheckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch'
)

reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=10, verbose=0,
    mode='min', min_delta=0.0001, cooldown=0, min_lr=0
)

optimizer = tf.keras.optimizers.Adam(lr=0.0001) 
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy']
              )

model.fit(traingen,
          validation_data=valgen, 
          epochs=10,
          callbacks=[early_stop_cb,modelCheckpt,reduceLR]
          )

