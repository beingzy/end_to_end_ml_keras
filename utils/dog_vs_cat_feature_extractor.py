import os
import numpy as np


def extract_features(conv_base,
                     directory,
                     sample_count,
                     generator_args=None,
                     target_size=(150, 150),
                     batch_size=20):
    """transform images stroed in directory to tensors with conv_base model
       to do the feature extraction
    """
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))

    generator = (ImageDataGenerator(**generator_args)
                 .flow_from_directory(
                     directory,
                     target_size=target_size,
                     batch_size=batch_size))

    ii = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[ii*batch_size: (ii+1)* batch_size] = features_batch
        labels[ii*batch_size: (ii+1) * batch_size] = labels_batch
        ii += 1
        if ii * bathc_size >= sample_count:
            break

    return features, labels
