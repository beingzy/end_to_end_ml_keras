"""wrapper for trained keras model
"""
from keras.applications import ResNet50


class ModelBase(object):
    _MODEL = None

    @classmethod
    def predict(cls, data):
        if cls._MODEL is None:
            raise ImportError("model is not loaded!")
        else:
            return cls._MODEL.predict(data)


class ModelResNet50(ModelBase):

    @classmethod
    def load_model(cls):
        cls._MODEL = ResNet50(weights="imagenet")

    @classmethod
    def predict(cls, data):
        if cls._MODEL is None:
            cls.load_model()
        return cls._MODEL.predict(data)
