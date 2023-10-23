from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import tensorflow as tf
import numpy as np

import sys
import os


arguments = sys.argv
if (len(arguments) == 2) :
    model = load_model("predict_model.h5")

    img_path = sys.argv[1]

    class_labels = ['benign', 'malignant']


    for filename in os.listdir(img_path):
        img_path = sys.argv[1]
        if filename.endswith(".jpg"):
            img_path = os.path.join(img_path, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

            preds = model.predict(x)

            pred = np.argmax(preds, axis=-1)
            print(f"Image: {filename}, Predicted Class: {class_labels[pred[0]]}")

else:
    print("Usage: python3 predicty.py image_file_name")
