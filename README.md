## Introduction
In this project, we will create a number of examples to illustrate how to train
neural nets of various architecture and how to serve trained model via RESTful
to support real-time application.

### Examples:
### Serve RestNet50
   * serve RestNet50 with RESTful API:
   ```bash
   conda activate keras_tf
   ipython resnet50.py
   ```
   RESTful API server URL: 'http://127.0.0.1:5000/'

   * request for analyzing image
   ```bash
   curl -X POST -F image=@data/test_dog.jpg "http://127.0.0.1:5000/predict"
   ```
