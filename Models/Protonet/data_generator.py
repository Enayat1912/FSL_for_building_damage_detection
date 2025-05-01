
"""
Code adapted from https://github.com/barnrang/Prototypical-network-keras-reimplementation
"""



"""
data_generator.py

Provides a custom `DataGenerator` class for episodic training of few-shot learning models
(e.g., prototypical networks or Siamese networks). The generator creates support and query
sets on the fly for each batch, enabling dynamic class sampling.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import random
import os
import cv2


def crop_center(img, cropx, cropy):
    """
    Crop the center region of an image.

    Args:
        img (np.ndarray): Input image array.
        cropx (int): Width of the cropped image.
        cropy (int): Height of the cropped image.

    Returns:
        np.ndarray: Center-cropped image.
    """
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, :]


class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for few-shot learning. Generates episodic batches
    consisting of support (sample) and query images for N-way K-shot learning.

    Attributes:
        csv_file (str): Path to CSV file containing image filenames and class labels.
        data_dir (str): Path to directory where images are stored.
        dim (tuple): Target image size (height, width).
        n_channels (int): Number of image channels (e.g., 3 for RGB).
        way (int): Number of classes per episode.
        shot (int): Number of support examples per class.
        query (int): Number of query examples per class.
        num_batch (int): Number of batches (episodes) per epoch.
        random_seed (int): Optional random seed for reproducibility.
    """

    def __init__(self, csv_file, data_dir, dim=(128, 128), n_channels=3,
                 way=4, shot=5, query=5, num_batch=20, random_seed=None):
        self.csv_file = csv_file
        self.dataset_path = data_dir
        self.dim = dim
        self.n_channels = n_channels
        self.way = way
        self.shot = shot
        self.query = query
        self.num_batch = num_batch
        self.random_seed = random_seed

        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        self.build_data(self.csv_file)
        self.on_epoch_end()

    def build_data(self, csv_file):
        """
        Loads class labels and image filenames from the CSV file.

        Args:
            csv_file (str): Path to the CSV file.
        """
        with open(csv_file, 'r') as file:
            instances = [line.rstrip() for line in file.readlines()]

        label_dict = {}
        for instance in instances[1:]:  # Skip header
            try:
                _, filename, class_id = instance.split(',')
                label_dict[filename] = class_id
            except ValueError:
                print(f"Skipping invalid row: {instance}")
        self.class_data = label_dict
        self.n_classes = 4  # TODO: Dynamically infer from data

    def __len__(self):
        """
        Returns the number of batches per epoch.

        Returns:
            int: Number of batches (episodes) per epoch.
        """
        return self.num_batch

    def __getitem__(self, index):
        """
        Generates a single batch (episode).

        Args:
            index (int): Batch index (unused, for Keras compatibility).

        Returns:
            tuple: ((X_sample, X_query), label) pair.
        """
        X_sample, X_query, label = self.__data_generation()
        return (X_sample, X_query), label

    def on_epoch_end(self):
        """
        Called at the end of every epoch. Can be used to shuffle data.
        """
        pass

    def __data_generation(self):
        """
        Generates one batch of data containing support and query sets.

        Returns:
            tuple: X_sample, X_query, and one-hot encoded query labels.
        """
        X_sample = np.empty((self.way, self.shot, *self.dim, self.n_channels))
        X_query = np.empty((self.way, self.query, *self.dim, self.n_channels))
        label = np.empty(self.way * self.query)

        for i in range(self.way):
            temp_list = [k for k, v in self.class_data.items() if int(v) == int(i)]

            total_required = self.shot + self.query
            if len(temp_list) < total_required:
                print(f"Class {i} has only {len(temp_list)} images. Needed: {total_required}.")
                sample_idx = random.sample(temp_list, len(temp_list))
                num_support = min(len(sample_idx), self.shot)
                num_query = len(sample_idx) - num_support
            else:
                sample_idx = random.sample(temp_list, total_required)
                num_support = self.shot
                num_query = self.query

            # Support images
            for j, img_file in enumerate(sample_idx[:num_support]):
                img_path = os.path.join(self.dataset_path, img_file)
                img_array = cv2.imread(img_path)
                img_resized = cv2.resize(img_array, self.dim)
                X_sample[i][j] = img_resized / 255.0

            # Query images
            for m, img_file in enumerate(sample_idx[num_support:num_support + num_query]):
                img_path = os.path.join(self.dataset_path, img_file)
                img_array = cv2.imread(img_path)
                img_resized = cv2.resize(img_array, self.dim)
                X_query[i][m] = img_resized / 255.0

            # Label for queries
            label[i * self.query: i * self.query + num_query] = i

        self.produced_classes = label
        return X_sample, X_query, to_categorical(label, num_classes=self.way)


