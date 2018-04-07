"""train 2D convnet for MNIST

   author: Yi Zhang <beingzy@gmail.com>
   date: 2018/04/07
"""
import os
import sys
import logging

import keras
import tensorflow as tf


logging.basicConfig(format='%(asctime)-15s %(messsage)s', stream=sys.stdout)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
