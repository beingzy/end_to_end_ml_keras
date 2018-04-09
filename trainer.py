"""train 2D convnet for MNIST

   author: Yi Zhang <beingzy@gmail.com>
   date: 2018/04/07
"""
import os
import sys
import logging
from datetime import datetime

import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from utils.dog_vs_cat_feature_extractor import extract_features

logging.basicConfig(format='%(asctime)-15s %(messsage)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# setup
DATA_DIR = os.path.join(os.getcwd(), 'data', 'dog_vs_cat')
MODEL_DIR = None
EVALUATION_DIR = None
train_dir = os.path.join(DATA_DIR, 'train')
test_dir = os.path.join(DATA_DIR, 'test')


datagen_setting = dict(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen = ImageDataGenerator(**datagen_setting)


train_generator = datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

validation_generator = datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

# ---------------------------
# define convnet architecture
# ---------------------------
def build_model(base_model):
    """
    """
    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])

    return model


def plot_history(history, metrics='acc'):
    """
    """
    raise NotImplementedError


if __name__ == "__main__":
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = 'convnet_dog_vs_cat_{}.h5'.format(dt_str)
    model_path = os.path.join('models', model_name)

    vgg16_conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3))

    conv_net = build_model(base_model=vgg16_conv_base)
    sys.stdout.write('start training...')
    history = conv_net.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
    )

    sys.stdout.write('training is completed')
    mode.save(model_path)
    sys.stdout.write('saved lastest model at {}'.format(model_path))
