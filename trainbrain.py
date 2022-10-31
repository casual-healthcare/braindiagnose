from cv2 import imread, resize
import numpy as np
import os
from sys import argv
import tensorflow as tf
import visualkeras
from PIL import ImageFont

# import needed libs

EPOCHS = 10 
IMG_WIDTH = 150
IMG_HEIGHT = 150
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
    print("Done loading data!")
    print("Preparing to train neural network...")
    xlab = tf.keras.utils.to_categorical(xlab)
    xtrain, xlabel = np.array(ximg), np.array(xlab)
    model = get_model()  
    font = ImageFont.truetype("Roboto-Thin.ttf", 32)
    visualkeras.layered_view(model, to_file='output8.png', legend=True, font=font, draw_volume=False, spacing=100)
    model.fit(xtrain, xlabel, epochs=EPOCHS) # get the model to train
    print("Loading testing data...")
    print("Done loading data!")

    yimg, ylab = load_data(y)  # load testing images
    ylab = tf.keras.utils.to_categorical(ylab)
    ytest, ylabel = np.array(yimg), np.array(ylab)
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
            img = imread(os.path.join(data_dir, str(folder), filename))
            #print(img.shape)  # turn this on if you want to see every image's shape
            img = resize(img, size)  # resize the images
            if img is not None:
                images.append(img)
                labels.append(folder)
    return (images, labels)


def get_model():  # get the model
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            50, (10, 10), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            25, (10, 10), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model3 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            25, (10, 10), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 5)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model4 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            25, (10, 10), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model5 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            25, (10, 10), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model6 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            25, (5, 5), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model7 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            25, (5, 5), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(125, activation="relu"),
        tf.keras.layers.Dense(75, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    CHOSEN = model7
    CHOSEN.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    CHOSEN.summary()
    return CHOSEN


if __name__ == "__main__":  # run program
    main()
