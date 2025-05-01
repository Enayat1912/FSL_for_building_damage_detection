




"""
efficientnet_model.py

Defines an EfficientNetB7-based image classification model with optional data augmentation and
fully connected layers for multi-class classification.

Author: Your Name
Date: 2025-05-01
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy


def efficientnet_classification_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Builds and compiles an EfficientNetB7-based image classification model.

    Args:
        input_shape (tuple): Shape of input images, default (224, 224, 3).
        num_classes (int): Number of target classes.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
    # Data augmentation block
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomZoom(0.2)
    ])

    # Base model with pre-trained ImageNet weights
    base_model = EfficientNetB7(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze base model layers

    # Input layer
    inputs = Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Build and compile model
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )

    return model


# Run a summary if executed directly
if __name__ == "__main__":
    model = efficientnet_classification_model()
    model.summary()




