#!/bin/sh
export FLASK_APP=./resnet50/app.py
source activate keras_tf
flask run -h 0.0.0.0
