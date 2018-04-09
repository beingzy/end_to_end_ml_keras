"""
"""
import matplotlib.pyplot as plt


def display_image(image_path):
    """
    """
    raise NotImplementedError


def deviance_plot(history, metric='acc', smoothing=False):
    """
    """
    if metric not in history:
        raise Error('fail to find {} in history!'.format(matric))

    train_metric = history[metric]
    valid_metric = history['val_'+metric]
    epochs = range(1, len(train_metric)+1)

    #ax, fig = plt.figure()
    plt.plot(epochs, train_metric, 'bo', label='training {}'.format(metric))
    plt.plot(epochs, valid_metric, 'b', label='validation {}'.format(metric))
    plt.title("Training vs. Validation: {}".format(matric))
    plt.legend()
    plt.figure()
