




"""
augment_and_split.py

This script performs data augmentation and oversampling for imbalanced image classification datasets.
It reads image paths and labels from a CSV, performs specified transformations on minority classes, and
splits the dataset into training and validation sets. Augmented images and new CSVs are saved for training.

"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

# Class-to-int encoding (adjust based on dataset)
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

available_transformations = 8  # Total transformations supported


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


def oversample_training_data(data_csv_path, images_dir, output_csv_dir, output_images_dir, val_split_pct=0.2):
    """Main oversampling and split logic."""
    data_df = pd.read_csv(data_csv_path)
    x_data = data_df['uuid'].tolist()
    y_data = data_df['labels'].tolist()

    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=val_split_pct, stratify=y_data, random_state=42
    )

    os.makedirs(output_images_dir, exist_ok=True)
    train_images_dir = os.path.join(output_images_dir, "train_images")
    val_images_dir = os.path.join(output_images_dir, "val_images")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)

    # Oversample training data
    train_data = {'uuid': [], 'labels': []}
    for uuid, label in tqdm(zip(x_train, y_train), total=len(x_train), desc="Oversampling Training Data"):
        img_path = os.path.join(images_dir, uuid)
        if not os.path.exists(img_path):
            print(f"Missing training image: {img_path}")
            continue

        img_array = cv2.imread(img_path)
        if img_array is None:
            print(f"Unreadable image: {img_path}")
            continue

        # Save original
        train_image_path = os.path.join(train_images_dir, uuid)
        if not os.path.exists(train_image_path):
            try:
                cv2.imwrite(train_image_path, img_array)
                train_data['uuid'].append(uuid)
                train_data['labels'].append(label)
            except Exception as e:
                print(f"Error saving original image {uuid}: {e}")
                continue

        # Oversample
        label_int = int(label)
        if label_int in oversample_times:
            for i in range(oversample_times[label_int]):
                transformed = transform(img_array, i % available_transformations)
                new_uuid = f"{uuid.split('.png')[0]}_aug{i}.png"
                transformed_path = os.path.join(train_images_dir, new_uuid)
                try:
                    cv2.imwrite(transformed_path, transformed)
                    train_data['uuid'].append(new_uuid)
                    train_data['labels'].append(label)
                except Exception as e:
                    print(f"Error saving transformed image {new_uuid}: {e}")
                    continue

    # Save training CSV
    train_csv_path = os.path.join(output_csv_dir, "train.csv")
    pd.DataFrame(train_data).to_csv(train_csv_path, index=False)

    # Save validation images and CSV
    val_data = {'uuid': [], 'labels': []}
    for uuid, label in tqdm(zip(x_val, y_val), total=len(x_val), desc="Saving Validation Data"):
        img_path = os.path.join(images_dir, uuid)
        if not os.path.exists(img_path):
            print(f"Missing validation image: {img_path}")
            continue

        img_array = cv2.imread(img_path)
        if img_array is not None:
            val_image_path = os.path.join(val_images_dir, uuid)
            cv2.imwrite(val_image_path, img_array)
            val_data['uuid'].append(uuid)
            val_data['labels'].append(label)

    val_csv_path = os.path.join(output_csv_dir, "validation.csv")
    pd.DataFrame(val_data).to_csv(val_csv_path, index=False)

    print(f"\n Training CSV saved to: {train_csv_path}")
    print(f" Validation CSV saved to: {val_csv_path}")
    print(f" Images saved to: {output_images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oversample and split image dataset.")
    parser.add_argument('--data_csv_path', required=True, help='Path to the input CSV with uuid and labels')
    parser.add_argument('--images_dir', required=True, help='Path to directory containing raw images')
    parser.add_argument('--output_csv_dir', required=True, help='Path to save generated train/val CSVs')
    parser.add_argument('--output_images_dir', required=True, help='Path to save processed images')
    parser.add_argument('--val_split_pct', type=float, default=0.2, help='Validation split ratio (default: 0.2)')

    args = parser.parse_args()

    oversample_training_data(
        data_csv_path=args.data_csv_path,
        images_dir=args.images_dir,
        output_csv_dir=args.output_csv_dir,
        output_images_dir=args.output_images_dir,
        val_split_pct=args.val_split_pct
    )



