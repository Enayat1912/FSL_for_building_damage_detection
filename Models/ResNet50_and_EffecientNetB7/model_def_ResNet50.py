"""
Adoptted from https://github.com/DIUx-xView/xView2_baseline/model/model.py
"""

"""
resnet50_model.py

Defines a hybrid CNN classifier using ResNet50 (pre-trained) and custom convolutional layers
for multi-class image classification. Augmentation, feature fusion, and dense layers are added
on top for enhanced performance.


"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy


def Resnet50_classification_model(input_shape=(128, 128, 3), num_classes=4):
    """
    Constructs a hybrid classification model combining ResNet50 features and custom CNN features.

    Args:
        input_shape (tuple): Shape of input images, e.g., (128, 128, 3).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    # Load pre-trained ResNet50 without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model

    # Input layer
    inputs = Input(shape=input_shape)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomZoom(0.2)
    ])
    x_aug = data_augmentation(inputs)

    # Custom CNN branch
    x = Conv2D(32, (5, 5), padding='same', activation='relu')(x_aug)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    # ResNet branch
    base_features = base_model(x_aug)
    base_features = GlobalAveragePooling2D()(base_features)

    # Concatenate custom and ResNet features
    merged = Concatenate()([x, base_features])

    # Fully connected layers
    x = Dense(1024, activation='relu')(merged)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )

    return model


if __name__ == "__main__":
    model = Resnet50_classification_model()
    model.summary()

