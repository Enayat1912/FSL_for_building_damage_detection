"""
Code adapted from https://github.com/DIUx-xView/xView2_baseline/blob/master/model/damage_inference.py

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
extract_building_polygons.py

Extracts individual building polygons from a post-disaster image using xBD-formatted labels.
Saves each cropped building image and outputs a CSV file with damage labels.

"""

import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from shapely import wkt
from collections import defaultdict

# Encoding damage intensity categories
damage_intensity_encoding = defaultdict(lambda: 0, {
    'destroyed': 3,
    'major-damage': 2,
    'minor-damage': 1,
    'no-damage': 0
})


def process_img(img_array, polygon_pts, scale_pct=0.8):
    """
    Crops a building polygon from the full image with padding.

    Args:
        img_array (np.ndarray): Input image as array.
        polygon_pts (np.ndarray): Coordinates of the building polygon.
        scale_pct (float): Expansion percentage for cropping bounds.

    Returns:
        np.ndarray: Cropped image region.
    """
    height, width, _ = img_array.shape
    xcoords = polygon_pts[:, 0]
    ycoords = polygon_pts[:, 1]
    xmin, xmax = np.min(xcoords), np.max(xcoords)
    ymin, ymax = np.min(ycoords), np.max(ycoords)

    xdiff, ydiff = xmax - xmin, ymax - ymin
    xmin = max(int(xmin - xdiff * scale_pct), 0)
    xmax = min(int(xmax + xdiff * scale_pct), width)
    ymin = max(int(ymin - ydiff * scale_pct), 0)
    ymax = min(int(ymax + ydiff * scale_pct), height)

    return img_array[ymin:ymax, xmin:xmax, :]


def process_img_poly(img_path, label_path, output_dir, output_csv_path):
    """
    Processes an image and its label JSON to extract and save building polygon crops.

    Args:
        img_path (str): Path to the post-disaster image.
        label_path (str): Path to the corresponding label JSON.
        output_dir (str): Directory where cropped images will be saved.
        output_csv_path (str): CSV file path to save the extracted labels.
    """
    x_data, y_data, wkt_data = [], [], []

    print(f"Reading image: {img_path}")
    print(f"Reading labels: {label_path}")

    img_array = np.array(Image.open(img_path).convert("RGB"))

    with open(label_path, 'r') as f:
        label_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for feat in label_data['features']['xy']:
        try:
            damage_type = feat['properties']['subtype']
            polygon_geom = wkt.loads(feat['wkt'])
            polygon_pts = np.array(polygon_geom.exterior.coords)

            poly_uuid = feat['properties']['uid']
            image_filename = poly_uuid + ".png"

            damage_label = damage_intensity_encoding[damage_type]
            poly_img = process_img(img_array, polygon_pts, scale_pct=0.8)

            output_path = os.path.join(output_dir, image_filename)
            cv2.imwrite(output_path, poly_img)

            x_data.append(poly_uuid)
            y_data.append(damage_label)
            wkt_data.append(feat['wkt'])

        except KeyError as e:
            print(f"Skipping feature due to missing key: {e}")
        except Exception as e:
            print(f"Error processing polygon {feat['properties'].get('uid', '?')}: {e}")

    if len(x_data) != len(y_data):
        raise ValueError("Mismatch in lengths of extracted data.")

    df = pd.DataFrame({'uuid': x_data, 'labels': y_data})
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df)} samples to {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract building polygon images and labels from xBD post-disaster data.")
    parser.add_argument('--image_path', required=True, help='Path to the post-disaster image (PNG)')
    parser.add_argument('--label_path', required=True, help='Path to the corresponding JSON label file')
    parser.add_argument('--output_dir', required=True, help='Directory to save extracted polygon images')
    parser.add_argument('--output_csv', required=True, help='Path to save the label CSV')

    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not os.path.isfile(args.label_path):
        raise FileNotFoundError(f"Label file not found: {args.label_path}")

    process_img_poly(
        img_path=args.image_path,
        label_path=args.label_path,
        output_dir=args.output_dir,
        output_csv_path=args.output_csv
    )


if __name__ == '__main__':
    main()


 

