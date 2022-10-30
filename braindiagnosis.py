
print("-----------------------------------------------------------------")
print("Welcome to BrainDiagnose! This application scans a brain MRI scan, and by harnessing the power of deep learning and artificial intelligence, generates a diagnosis based on what it sees.")
print("Created by Azhaan Salam")
print("This project is licensed under the MIT License")
print("-----------------------------------------------------------------")
print("Setting up AI & Resources...")
print("-----------------------------------------------------------------")


import numpy as np
import sys
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

size = (150, 150)

model = tf.keras.models.load_model(sys.argv[1])
print("-----------------------------------------------------------------")
img = cv2.imread(sys.argv[2])
img = cv2.resize(img, size)
img = np.array(img)
classification = model.predict(
            [img.reshape(1, 150, 150, 3)]
)
diagnosis = classification.argmax()
print(diagnosis)