"""
"""
import os
import sys
import logging
import csv
from datetime import datetime

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.datasets import mnist
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd


logging.basicConfig(format='%(asctime)-15s %(messsage)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

## -----------------------
## model constructor
## -----------------------
def prepare_data(data):
    """data: np.ndarray
    """
    data = data.reshape(data.shape[0], 28, 28, 1)
    data = data.astype('float32')
    data /= 255.
    return data


def build_convnet():
    """build multi-level classifier with convolutional neuralnets
    """
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def build_transfer_convnet(base_model):
    """
    """
    raise NotImplementedError


## ------------------------
## split training/test data
## ------------------------
if __name__ == "__main__":
    train_config = dict(
        batch_size = 25,
        epochs = 20
    )

    num_classes = 10
    img_rows, img_cols = 28, 28

    # prepare training and test data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = prepare_data(x_train)
    x_test = prepare_data(x_test)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # construct covnet and training/validation
    model = build_convnet()
    print('start training model (total epochs: {})'.format(train_config['epochs']))
    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        **train_config)

    # export trained model and history information
    sys.stdout.write('training is completed')
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = 'mnist_{}.h5'.format(dt_str)
    model_path = os.path.join('models', model_name)
    model.save(model_path)
    sys.stdout.write('saved lastest model at {}'.format(model_path))

    outfile = os.path.join('history',
        'mnist_{}_history'.format(dt_str))
    history_df = pd.DataFrame(history.history)
    hustory_df.to_csv(outfile, columns=True, index=False)
    sys.stdout.write('succeed in exporting history file!')
