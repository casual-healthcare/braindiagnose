import cv2
import numpy as np
import os
from sys import argv
import tensorflow as tf
import visualkeras
# import needed libs

EPOCHS = 10 
IMG_WIDTH = 200
IMG_HEIGHT = 200
NUM_CATEGORIES = 5

"""
===== KEY =====
1. Glioma
2. Meningioma
3. No tumor
4. Pituitary
===============
"""



def main():
    if len(argv) not in [3, 4]:
        x = "train"
        y = "test"
    else:
        x = argv[1]
        y = argv[2]
    print("Loading training data...")
    ximg, xlab = load_data(x)  # load training images
    print("Successfuly loaded training data!")
    print("Loading testing data...")
    yimg, ylab = load_data(y)  # load testing images
    print("Done loading data!")
    print("Preparing to train neural network...")
    xlab = tf.keras.utils.to_categorical(xlab)
    ylab = tf.keras.utils.to_categorical(ylab)
    xtrain, xlabel = np.array(ximg), np.array(xlab)
    ytest, ylabel = np.array(yimg), np.array(ylab)
    model = get_model()  
    visualkeras.layered_view(model)
    model.fit(xtrain, xlabel, epochs=EPOCHS) # get the model to train
    model.evaluate(ytest,  ylabel, verbose=2)  # test the model to check how well it did
    if len(argv) == 4:  # save function
        filename = argv[3]
        confirm = input("Save? ")
        if confirm.lower() != "n" and confirm.lower() != "no":
            model.save(filename)
            print(f"Model saved to {filename}.")


def load_data(data_dir):
    # function to load and format images
    images = []
    labels = []
    size = (IMG_WIDTH, IMG_HEIGHT)
    for folder in os.listdir(os.path.join(data_dir)):
        for filename in os.listdir(os.path.join(data_dir, str(folder))):     # for every images
            img = cv2.imread(os.path.join(data_dir, str(folder), filename))
            #print(img.shape)  # turn this on if you want to see every image's shape
            img = cv2.resize(img, size)  # resize the images
            if img is not None:
                images.append(img)
                labels.append(folder)
    return (images, labels)


def get_model():  # get the model
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            50, (5, 5), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(350, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(150, activation="relu"),
        tf.keras.layers.Dense(80, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    CHOSEN = model1
    CHOSEN.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    CHOSEN.summary()
    return CHOSEN


if __name__ == "__main__":  # run program
    main()
