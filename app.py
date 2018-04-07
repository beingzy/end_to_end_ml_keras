"""set up a http API server to serve ML model
   reference: https://goo.gl/ck5Mq9

   start server:
   $ipython app.py

   inference model
   $curl -X POST -F image=@data/test_dog.jpg "http://127.0.0.1:5000/predict"

   author: Yi Zhang <beingzy@gmail.com>
   date: 2018/04/07
"""
import os
import sys
import logging
import io
import json

import numpy as np
import flask

from keras.applications import imagenet_utils
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from PIL import Image


app = flask.Flask(__name__)
model = None


def load_model():
    global model
    model = ResNet50(weights="imagenet")


def prepare_image(image, target):
    """
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


@app.route('/predict', methods=["POST"])
def predict():
    from flask import request
    # initialize the data dictionary that will be returned
    # from view
    resp = {'success': False}

    if request.method == "POST":
        if request.files.get("image"):
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_image(image, target=(224, 224))

            # classify the iput data and then initialize the list of
            # predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            resp["predictions"] = []

            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": prob * 1.}
                resp["predictions"].append(r)

            resp["success"] = True

    return json.dumps(resp)


if __name__ == "__main__":
    load_model()
    app.run()
