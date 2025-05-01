



"""
data_generator.py

Defines a batch-based Keras-compatible data generator for multi-class image classification.
Reads image paths and labels from a CSV and loads data on the fly for training or testing.

Expected CSV format:
    uuid,label

"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    """
    Keras-compatible data generator for image classification.

    Attributes:
        csv_file (str): Path to the CSV file containing image UUIDs and labels.
        data_dir (str): Path to the image directory.
        batch_size (int): Number of samples per batch.
        dim (tuple): Dimensions to resize images (width, height).
        num_classes (int): Number of output classes.
    """

    def __init__(self, csv_file, data_dir, batch_size, dim, num_classes):
        """
        Initialization.

        Args:
            csv_file (str): Path to CSV file with image filenames and labels.
            data_dir (str): Path to image directory.
            batch_size (int): Size of each data batch.
            dim (tuple): Target dimensions for images (width, height).
            num_classes (int): Total number of classes for one-hot encoding.
        """
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dim = dim
        self.num_classes = num_classes
        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch.

        Returns:
            int: Number of batches.
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.

        Args:
            index (int): Index of the batch.

        Returns:
            tuple: Batch of images and their one-hot encoded labels.
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data.iloc[indexes]
        images, labels = self.__data_generation(batch_data)
        return images, labels

    def on_epoch_end(self):
        """
        Updates indexes after each epoch. Useful for shuffling if needed.
        """
        self.indexes = np.arange(len(self.data))

    def __data_generation(self, batch_data):
        """
        Generates data for the current batch.

        Args:
            batch_data (pd.DataFrame): Subset of data with filenames and labels.

        Returns:
            tuple: Arrays of images and labels.
        """
        images = []
        labels = []

        for _, row in batch_data.iterrows():
            img_path = os.path.join(self.data_dir, row['uuid'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.dim)
                    images.append(img / 255.0)  # Normalize image
                    labels.append(tf.keras.utils.to_categorical(row['labels'], num_classes=self.num_classes))
                else:
                    print(f"[Warning] Unreadable image: {img_path}")
            else:
                print(f"[Warning] Missing image: {img_path}")

        return np.array(images), np.array(labels)


