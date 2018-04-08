## Recognize Hand-written Digit Images with 2D Convolutional NeuralNets

### Introduction
In this project, we will develop service empowered by 2D CNN to recognize numbers on handwritten images.

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
