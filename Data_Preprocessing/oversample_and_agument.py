"""
Code adapted from https://github.com/DIUx-xView/xView2_baseline/blob/master/model/process_data.py

xview2-baseline Copyright 2019 Carnegie Mellon University. BSD-3

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation a
nd/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived 
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. 
Please see Copyright notice for non-US Government use and distribution.
"""




"""
augment_oversample_data

This script performs data augmentation and oversampling for imbalanced image classification datasets.
It reads image paths and labels from a CSV, performs specified transformations on minority classes,
and saves the augmented dataset into a specified directory without splitting into training/validation.

Author: Your Name
Date: 2025-05-01
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

# Class-to-int encoding (optional, adjust based on your dataset)
damage_intensity_encoding = {
    'destroyed': 3,
    'major-damage': 2,
    'minor-damage': 1,
    'no-damage': 0
}

# Oversampling multipliers per class
oversample_times = {
    3: 8,  # destroyed
    2: 1,  # major-damage
    1: 1,  # minor-damage
    0: 0   # no-damage
}

available_transformations = 8  # Number of transformation types


def transform(image, transformation):
    """Apply data augmentation transformation based on index."""
    if transformation == 0:
        return cv2.flip(image, 1)
    elif transformation == 1:
        return cv2.flip(image, 0)
    elif transformation == 2:
        return cv2.flip(image, -1)
    elif transformation == 3:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif transformation == 4:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif transformation == 5:
        alpha = np.random.uniform(0.8, 1.2)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    elif transformation == 6:
        return cv2.resize(image, (int(image.shape[1] * 0.9), int(image.shape[0] * 0.9)))
    elif transformation == 7:
        return cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    return image


def augment_and_oversample(data_csv_path, images_dir, output_csv_path, output_images_dir):
    """
    Performs data augmentation and oversampling on an imbalanced dataset.

    Args:
        data_csv_path (str): Path to input CSV file.
        images_dir (str): Path to directory containing original images.
        output_csv_path (str): Path to save the augmented label CSV.
        output_images_dir (str): Path to save all original and augmented images.
    """
    data_df = pd.read_csv(data_csv_path)
    os.makedirs(output_images_dir, exist_ok=True)

    augmented_data = {'uuid': [], 'labels': []}

    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Augmenting and Oversampling"):
        uuid = row['uuid']
        label = int(row['labels'])
        img_path = os.path.join(images_dir, uuid)

        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        img_array = cv2.imread(img_path)
        if img_array is None:
            print(f"Unreadable image: {img_path}")
            continue

        # Save original image
        output_path = os.path.join(output_images_dir, uuid)
        try:
            if not os.path.exists(output_path):
                cv2.imwrite(output_path, img_array)
            augmented_data['uuid'].append(uuid)
            augmented_data['labels'].append(label)
        except Exception as e:
            print(f"Error saving original image {uuid}: {e}")
            continue

        # Perform augmentation for oversampling
        if label in oversample_times and oversample_times[label] > 0:
            for i in range(oversample_times[label]):
                transformed = transform(img_array, i % available_transformations)
                new_uuid = f"{uuid.split('.png')[0]}_aug{i}.png"
                transformed_path = os.path.join(output_images_dir, new_uuid)
                try:
                    cv2.imwrite(transformed_path, transformed)
                    augmented_data['uuid'].append(new_uuid)
                    augmented_data['labels'].append(label)
                except Exception as e:
                    print(f"Error saving transformed image {new_uuid}: {e}")
                    continue

    # Save final CSV with all augmented samples
    pd.DataFrame(augmented_data).to_csv(output_csv_path, index=False)
    print(f"\n Augmented data CSV saved to: {output_csv_path}")
    print(f"Images saved to: {output_images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data augmentation and oversampling script (no train/val split).")
    parser.add_argument('--data_csv_path', required=True, help='Path to the input CSV with uuid and labels')
    parser.add_argument('--images_dir', required=True, help='Path to directory containing original images')
    parser.add_argument('--output_csv_path', required=True, help='Path to save the final augmented CSV')
    parser.add_argument('--output_images_dir', required=True, help='Path to save all processed images')

    args = parser.parse_args()

    augment_and_oversample(
        data_csv_path=args.data_csv_path,
        images_dir=args.images_dir,
        output_csv_path=args.output_csv_path,
        output_images_dir=args.output_images_dir
    )



