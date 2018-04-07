#!/bin/sh
export FLASK_APP=./app.py
source activate keras_tf
flask run -h 0.0.0.0
