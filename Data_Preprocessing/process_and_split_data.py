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
from PIL import Image
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
import random
import argparse
import logging
import json
import cv2
import datetime

import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO)

# Configurations
UNDERSAMPLE_THRESHOLD = 0
VALIDATION_SPLIT = 0
TEST_SPLIT = 0

damage_intensity_encoding = defaultdict(lambda: 0)
damage_intensity_encoding['destroyed'] = 3
damage_intensity_encoding['major-damage'] = 2
damage_intensity_encoding['minor-damage'] = 1
damage_intensity_encoding['no-damage'] = 0


def save_labeled_data(x, y, csv_path):
    data_array = {'uuid': x, 'labels': y}
    df = pd.DataFrame(data = data_array)
    df.to_csv(csv_path)

def process_img(img_array, polygon_pts, scale_pct):
    """Process Raw Data into

            Args:
                img_array (numpy array): numpy representation of image.
                polygon_pts (array): corners of the building polygon.

            Returns:
                numpy array: .

    """

    height, width, _ = img_array.shape

    xcoords = polygon_pts[:, 0]
    ycoords = polygon_pts[:, 1]
    xmin, xmax = np.min(xcoords), np.max(xcoords)
    ymin, ymax = np.min(ycoords), np.max(ycoords)

    xdiff = xmax - xmin
    ydiff = ymax - ymin

    #Extend image by scale percentage
    xmin = max(int(xmin - (xdiff * scale_pct)), 0)
    xmax = min(int(xmax + (xdiff * scale_pct)), width)
    ymin = max(int(ymin - (ydiff * scale_pct)), 0)
    ymax = min(int(ymax + (ydiff * scale_pct)), height)

    return img_array[ymin:ymax, xmin:xmax, :]


def process_data(input_path, output_path, output_csv_path):
    """Process Raw Data into

        Args:
            dir_path (path): Path to the xBD dataset.
            data_type (string): String to indicate whether to process
                                train, test, or holdout data.

        Returns:
            x_data: A list of numpy arrays representing the images for training
            y_data: A list of labels for damage represented in matrix form

    """
    x_data = []
    y_data = []

    disasters = [folder for folder in os.listdir(input_path) if not folder.startswith('.')]
    disaster_paths = ([input_path + "/" +  d + "/images" for d in disasters])
    image_paths = []
    image_paths.extend([(disaster_path + "/" + pic) for pic in os.listdir(disaster_path)] for disaster_path in disaster_paths)
    img_paths = np.concatenate(image_paths)

    for img_path in tqdm(img_paths):

        img_obj = Image.open(img_path)
        img_array = np.array(img_obj)

        #Get corresponding label for the current image
        label_path = img_path.replace('png', 'json').replace('images', 'labels')
        label_file = open(label_path)
        label_data = json.load(label_file)

        for feat in label_data['features']['xy']:

            # only images post-disaster will have damage type
            try:
                damage_type = feat['properties']['subtype']
            except: # pre-disaster damage is default no-damage
                damage_type = "no-damage"
                continue

            poly_uuid = feat['properties']['uid'] + ".png"

            if (UNDERSAMPLE_THRESHOLD == 0 or y_data.count(damage_intensity_encoding[damage_type]) < UNDERSAMPLE_THRESHOLD):
                y_data.append(damage_intensity_encoding[damage_type])

                polygon_geom = shapely.wkt.loads(feat['wkt'])
                polygon_pts = np.array(list(polygon_geom.exterior.coords))
                poly_img = process_img(img_array, polygon_pts, 0.8)
                cv2.imwrite(output_path + "/" + poly_uuid, poly_img)
                x_data.append(poly_uuid)

    output_train_csv_path = os.path.join(output_csv_path, "train.csv")
    output_test_csv_path = os.path.join(output_csv_path, "test.csv")
    output_val_csv_path = os.path.join(output_csv_path, "val.csv")

    if (TEST_SPLIT > 0):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=TEST_SPLIT, random_state=1)
        if (VALIDATION_SPLIT > 0):
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VALIDATION_SPLIT/(1-TEST_SPLIT), random_state=1)
            save_labeled_data(x_val, y_val, output_val_csv_path)

        save_labeled_data(x_test, y_test, output_test_csv_path)
        save_labeled_data(x_train, y_train, output_train_csv_path)

    elif (VALIDATION_SPLIT > 0):
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=VALIDATION_SPLIT/(1-TEST_SPLIT), random_state=1)
        save_labeled_data(x_val, y_val, output_val_csv_path)
        save_labeled_data(x_train, y_train, output_train_csv_path)
        
    else:
        save_labeled_data(x_data, y_data, output_train_csv_path)
    

def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--input_dir',
                        required=True,
                        metavar="/path/to/xBD_input",
                        help="Full path to the parent dataset directory")
    parser.add_argument('--output_dir',
                        required=True,
                        metavar='/path/to/xBD_output',
                        help="Path to new directory to save images")
    parser.add_argument('--output_dir_csv',
                        required=True,
                        metavar='/path/to/xBD_output_csv',
                        help="Path to new directory to save csv")
    args = parser.parse_args()

    logging.info("Started Processing for Data")
    process_data(args.input_dir, args.output_dir, args.output_dir_csv)
    logging.info("Finished Processing Data")


if __name__ == '__main__':
    main()
