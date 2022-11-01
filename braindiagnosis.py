
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
if diagnosis == 1:
    print("This seems to be a malignant tumor in the brain. A malignant tumor is a cancerous tumor with abnormal cells that will grow rapidly and spread to other tissues. This can cause a veirtey of side effects.")
elif diagnosis == 2:
    print("This seems to be a benign tumor. A benign tumor is a growth of cells in the brain, and slowly spreads. They don't usually cause problems, however, they can become large and create pressure in the brain, which can be harmful.")
elif diagnosis == 3:
    print("No tumor detected.")
else:
    print("This seems to be a pituitary tumor, which is a tumor that forms in your pituitary gland. Most of these tumors are noncancerous. However, these tumors can cause the pituitary gland to act abnormally, which can lead to many problems such as too many hormones being created in the body.")
print("-----------------------------------------------------------------")
