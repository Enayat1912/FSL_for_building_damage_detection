"""
Code adapted from https://github.com/DIUx-xView/xView2_baseline/blob/master/model/process_data.py

xview2-baseline Copyright 2019 Carnegie Mellon University. BSD-3

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

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
process_xbd_dataset

Extracts building-level image crops from the xBD (xView2) dataset using polygon annotations.
Saves cropped building images and generates label CSVs. Optionally splits into train/val/test sets.

"""

import os
import cv2
import json
import math
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from shapely import wkt
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

# Default label encoding
damage_intensity_encoding = defaultdict(lambda: 0, {
    'destroyed': 3,
    'major-damage': 2,
    'minor-damage': 1,
    'no-damage': 0
})


def save_labeled_data(filenames, labels, csv_path):
    """Save UUID-label pairs to CSV."""
    df = pd.DataFrame({'uuid': filenames, 'labels': labels})
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved label CSV: {csv_path}")


def process_img(img_array, polygon_pts, scale_pct=0.8):
    """Crop and return an image patch around the building polygon."""
    height, width, _ = img_array.shape
    xcoords, ycoords = polygon_pts[:, 0], polygon_pts[:, 1]
    xmin, xmax = np.min(xcoords), np.max(xcoords)
    ymin, ymax = np.min(ycoords), np.max(ycoords)

    xdiff, ydiff = xmax - xmin, ymax - ymin
    xmin = max(int(xmin - xdiff * scale_pct), 0)
    xmax = min(int(xmax + xdiff * scale_pct), width)
    ymin = max(int(ymin - ydiff * scale_pct), 0)
    ymax = min(int(ymax + ydiff * scale_pct), height)

    return img_array[ymin:ymax, xmin:xmax, :]


def process_data(input_path, output_img_dir, output_csv_dir,
                 undersample_threshold=0, val_split=0.0, test_split=0.0):
    """Main processing function for xBD image and label extraction."""
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)

    x_data, y_data = [], []

    # Get all image paths
    disasters = [d for d in os.listdir(input_path) if not d.startswith('.')]
    image_paths = []
    for disaster in disasters:
        image_dir = os.path.join(input_path, disaster, "images")
        if not os.path.exists(image_dir): continue
        for img in os.listdir(image_dir):
            if img.endswith(".png"):
                image_paths.append(os.path.join(image_dir, img))

    logging.info(f"Found {len(image_paths)} total post-disaster images.")

    for img_path in tqdm(image_paths, desc="Processing Images"):
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)

            label_path = img_path.replace("images", "labels").replace(".png", ".json")
            if not os.path.exists(label_path):
                continue

            with open(label_path, "r") as f:
                label_data = json.load(f)

            for feat in label_data['features']['xy']:
                uid = feat['properties']['uid']
                try:
                    damage_type = feat['properties'].get('subtype', 'no-damage')
                    damage_label = damage_intensity_encoding[damage_type]
                except Exception:
                    continue

                if undersample_threshold and y_data.count(damage_label) >= undersample_threshold:
                    continue

                try:
                    polygon_geom = wkt.loads(feat['wkt'])
                    if not isinstance(polygon_geom, Polygon):
                        continue
                    polygon_pts = np.array(polygon_geom.exterior.coords)
                    poly_img = process_img(img_array, polygon_pts)
                except Exception as e:
                    logging.warning(f"Invalid polygon in {uid}: {e}")
                    continue

                out_filename = f"{uid}.png"
                out_path = os.path.join(output_img_dir, out_filename)

                try:
                    cv2.imwrite(out_path, poly_img)
                    x_data.append(out_filename)
                    y_data.append(damage_label)
                except Exception as e:
                    logging.warning(f"Failed to save image {uid}: {e}")
        except Exception as e:
            logging.warning(f"Skipping image {img_path}: {e}")

    # Handle dataset splits
    train_csv = os.path.join(output_csv_dir, "train.csv")
    val_csv = os.path.join(output_csv_dir, "val.csv")
    test_csv = os.path.join(output_csv_dir, "test.csv")

    if test_split > 0:
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=test_split, random_state=42
        )
        save_labeled_data(x_test, y_test, test_csv)
    else:
        x_train, y_train = x_data, y_data

    if val_split > 0:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=val_split / (1 - test_split), random_state=42
        )
        save_labeled_data(x_val, y_val, val_csv)

    save_labeled_data(x_train, y_train, train_csv)


def main():
    parser = argparse.ArgumentParser(description="Process xBD dataset into cropped building images and CSV labels.")
    parser.add_argument('--input_dir', required=True, help='Path to the root xBD dataset directory')
    parser.add_argument('--output_dir', required=True, help='Directory to save cropped building images')
    parser.add_argument('--output_csv_dir', required=True, help='Directory to save label CSVs')
    parser.add_argument('--undersample_threshold', type=int, default=0, help='Max samples per class (0 = no limit)')
    parser.add_argument('--val_split', type=float, default=0.0, help='Validation split fraction (e.g., 0.1)')
    parser.add_argument('--test_split', type=float, default=0.0, help='Test split fraction (e.g., 0.1)')

    args = parser.parse_args()

    logging.info("Starting xBD image processing...")
    process_data(
        input_path=args.input_dir,
        output_img_dir=args.output_dir,
        output_csv_dir=args.output_csv_dir,
        undersample_threshold=args.undersample_threshold,
        val_split=args.val_split,
        test_split=args.test_split
    )
    logging.info(" Processing complete.")


if __name__ == '__main__':
    main()



