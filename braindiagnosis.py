
print("-----------------------------------------------------------------")
print("Welcome to BrainDiagnose! This application analyzes a brain MRI scan, and by harnessing the power of deep learning and artificial intelligence, generates a diagnosis based on what it sees.")
print("Created by Azhaan Salam")
print("This project is licensed under the MIT License")
print("-----------------------------------------------------------------")
print("Setting up AI & Resources...")
print("-----------------------------------------------------------------")
import numpy as np
import sys, os
from time import sleep
import tensorflow as tf
import cv2, shutil
import matplotlib.pyplot as plt
size = (150, 150)
model = tf.keras.models.load_model(sys.argv[1])

def main():
    while True:
        print("-----------------------------------------------------------------")
        print("Waiting for images to be placed in '"+sys.argv[2]+"' for diagnosis")
        print("Designated output folder: " + sys.argv[3])
        images = waitforimages(sys.argv[2], sys.argv[3])
        print("-----------------------------------------------------------------")
        print("RESULTS")
        for i in range(len(images)):
            img = images[i]
            img = np.array(img)
            classification = model.predict(
                        [img.reshape(1, 150, 150, 3)]
            )
            diagnosis = classification.argmax()
            print(f"Image {i+1}:", end=' ')
            if diagnosis == 1:
                print("This seems to be a malignant tumor in the brain. A malignant tumor is a cancerous tumor with abnormal cells that will grow rapidly and spread to other tissues. This can cause a veirtey of side effects.")
            elif diagnosis == 2:
                print("This seems to be a benign tumor. A benign tumor is a growth of cells in the brain, and slowly spreads. They don't usually cause problems, however, they can become large and create pressure in the brain, which can be harmful.")
            elif diagnosis == 3:
                print("No tumor detected.")
            else:
                print("This seems to be a pituitary tumor, which is a tumor that forms in your pituitary gland. Most of these tumors are noncancerous. However, these tumors can cause the pituitary gland to act abnormally, which can lead to many problems such as too many hormones being created in the body.")
def waitforimages(foldername, trashbin):
    while True:
        images = []
        for file in os.listdir(os.path.join(foldername)):
            try:
                img = cv2.imread(os.path.join(foldername, file))
                img = cv2.resize(img, size)
                if img is not None:
                    images.append(img)
            except:
                print("error")
                pass
            if file is not None:
                shutil.move(os.path.join(foldername, file), trashbin)
        if len(images) > 0:
            return images
        sleep(2)
main()