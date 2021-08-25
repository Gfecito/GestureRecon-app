from flask import Flask, render_template, request, redirect, jsonify
import os
from werkzeug.utils import secure_filename
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import PIL

# Constants
app = Flask(__name__)
img_height = img_width = 180
weightList = pickle.load(open('model_keras.pkl', 'rb'))
allowedExtensions = ["png", "jpg", "jpeg"]
# Path to save the uploaded images on local system
app.config["IMAGES"] = "/Users/santi/Desktop/'Tiny Desktop'/Work/proj-vis-eff/static/imageUpload"



# BACKEND

def createModel(weights):
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    modelBuilt = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10)
    ])

    modelBuilt.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    modelBuilt.set_weights(weights)

    return modelBuilt


def imagePreprocessing(img):
    model = createModel(weightList)
    # Preprocessing image for model
    class_names = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb', '06_index', '07_ok', '08_palm_moved', '09_c',
                  '10_down']

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    textResult = [class_names[np.argmax(score)], 100 * np.max(score)]
    return textResult


def predictModel(image='testImage.png'):
    img = keras.preprocessing.image.load_img(
        image, target_size=(img_height, img_width)
    )
    return imagePreprocessing(img)

# DONT FORGET TO USE ESCAPE()

def chooseAnimation(gesture):
    animations = {
        "None": "NoAnimation",
        "01_palm": "Palm",
        "02_l": "L",
        "03_fist": "Fist",
        "04_fist_moved": "FistMove",
        "05_thumb": "Thumb",
        "06_index": "Index",
        "07_ok": "Ok",
        "08_palm_moved": "PalmMove",
        "09_c": "C",
        "10_down": "Downwards"
    }

    gestures = {
        "None": "No gesture",
        "01_palm": "Open palm",
        "02_l": "L shape",
        "03_fist": "Closed fist",
        "04_fist_moved": "A fist moving",
        "05_thumb": "Thumb up",
        "06_index": "Index up",
        "07_ok": "ok sign",
        "08_palm_moved": "Open palm moving",
        "09_c": "C shape",
        "10_down": "Get down sign"
    }

    return [animations[gesture], gestures[gesture]]


@app.route("/", methods=['GET', 'POST'])
def baseHtml():
    [animation, gesture] = chooseAnimation("None")
    html = render_template("index.html", animation=animation, gesture=gesture, certainty="N/A")
    return html


@app.route('/predict', methods=['POST'])
def predict():
    prediction = predictModel("/Users/santi/Desktop/Tiny Desktop/Work/proj-vis-eff/static/imageUpload/gesture2.png")

    [animation, gesture] = chooseAnimation(prediction[0])
    return render_template('index.html', animation=animation, gesture=gesture, certainty=str(prediction[1])[:5]+"%")
