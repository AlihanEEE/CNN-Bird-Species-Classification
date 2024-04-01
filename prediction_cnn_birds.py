import tensorflow as tf
from glob import glob
import cv2
import numpy as np

import matplotlib.pyplot as plt

modepath = "training_2/first_training.hdf5"
# Load the model from the .hd5f file
models = tf.keras.models.load_model(modepath)
# --------------------------------------------------------
folders = glob("archive\\train\\*")

class_names = []

for name in folders[:100]:
        class_names.append(name.split("\\")[-1])
print(class_names)

def preprocess(img):
        resized = cv2.resize(img, (224, 224))
        n_resized = resized / 255.0
        n_resized = np.expand_dims(n_resized, axis=0)
        return n_resized

imgpath = glob("test_images\\*.jpg")

for imge in imgpath:
        # Load the image to classify
        org_image = cv2.imread(imge)

        image = preprocess(org_image.copy())
        
        

        # Predict the class of the image
        predictions = models.predict(image)

        # Get the class with the highest probability
        bird_class = predictions.argmax()

        cv2.imshow(class_names[bird_class], org_image)
        cv2.waitKey(0)


