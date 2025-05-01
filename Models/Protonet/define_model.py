 
"""
 
Code adapted from https://github.com/barnrang/Prototypical-network-keras-reimplementation

"""




"""
model.py

Defines the convolutional backbone and utility functions for use in a few-shot learning
Siamese or Prototypical Network architecture for image classification.

Includes:
- A CNN encoder (`conv_net`) used as a feature extractor.
- Distance metrics (L1, L2).
- Custom loss and accuracy functions.

"""

import numpy as np
import numpy.random as rng
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Activation,
    BatchNormalization
)

eps = 1e-12
input_shape = (128, 128, 3)


def W_init(shape, name=None):
    """
    Initialize convolutional weights using a small Gaussian distribution.

    Args:
        shape (tuple): Shape of the weight tensor.
        name (str, optional): Name of the variable.

    Returns:
        tf.Variable: Initialized weights.
    """
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def b_init(shape, name=None):
    """
    Initialize biases with values centered around 0.5 (as used in some Siamese networks).

    Args:
        shape (tuple): Shape of the bias tensor.
        name (str, optional): Name of the variable.

    Returns:
        tf.Variable: Initialized biases.
    """
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def conv_net():
    """
    Build the convolutional neural network used as the encoder in a Siamese/ProtoNet.

    Returns:
        tf.keras.Sequential: The CNN model instance.
    """
    convnet = Sequential()
    for i in range(4):
        convnet.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
        convnet.add(BatchNormalization())
        convnet.add(Activation('relu'))
        convnet.add(MaxPooling2D())
        # Optional dropout for regularization (commented out)
        # convnet.add(Dropout(0.2))
    convnet.add(Flatten())
    return convnet


def l1_distance(x, y):
    """
    Compute L1 (Manhattan) distance between two tensors.

    Args:
        x (tf.Tensor): First tensor.
        y (tf.Tensor): Second tensor.

    Returns:
        tf.Tensor: L1 distance.
    """
    return tf.reduce_sum(tf.maximum(tf.abs(x - y), eps), axis=1, keepdims=True)


def l2_distance(x, y):
    """
    Compute L2 (Euclidean) distance between two tensors.

    Args:
        x (tf.Tensor): First tensor.
        y (tf.Tensor): Second tensor.

    Returns:
        tf.Tensor: L2 distance.
    """
    return tf.sqrt(tf.reduce_sum(tf.maximum(tf.square(x - y), eps), axis=1, keepdims=True))


def hinge_loss(target, pred, h=1.0):
    """
    Compute hinge loss, typically used for pairwise comparisons.

    Args:
        target (tf.Tensor): Ground truth labels (not used in this variant).
        pred (tf.Tensor): Predicted distances or logits.
        h (float): Margin threshold.

    Returns:
        tf.Tensor: Hinge loss value.
    """
    return tf.reduce_mean(tf.maximum(pred + h, 0.0))


def acc(target, pred):
    """
    Compute accuracy as the proportion of correct predictions (using a distance threshold).

    Args:
        target (tf.Tensor): Ground truth values (e.g., 0 or 1).
        pred (tf.Tensor): Model predictions (e.g., distances).

    Returns:
        tf.Tensor: Accuracy score.
    """
    result = tf.cast(tf.less(pred, target), dtype=tf.float32)
    return tf.reduce_mean(result)
