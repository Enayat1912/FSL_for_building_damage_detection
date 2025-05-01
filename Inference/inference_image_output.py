




"""
generate_inference_mask

Generates a grayscale mask image from an inference JSON file where each building polygon is filled
according to its damage classification level.

Author: Your Name
Date: 2025-05-01
"""

import os
import json
import argparse
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon
from cv2 import fillPoly, imwrite


def open_json(json_file_path):
    """
    Loads the inference JSON file and extracts the polygon data.

    Args:
        json_file_path (str): Path to the inference JSON file.

    Returns:
        list: List of polygon features with WKT and subtype.
    """
    with open(json_file_path, 'r') as jf:
        json_data = json.load(jf)
        return json_data['features']['xy']


def create_image(inference_data, image_size=(1024, 1024)):
    """
    Creates a grayscale mask image with polygons filled by damage category.

    Args:
        inference_data (list): Parsed JSON features with 'subtype' and 'wkt'.
        image_size (tuple): Shape of the output image (height, width).

    Returns:
        np.ndarray: Grayscale image mask with damage classifications.
    """
    damage_key = {
        'un-classified': 1,
        'no-damage': 1,
        'minor-damage': 2,
        'major-damage': 3,
        'destroyed': 4
    }

    mask_img = np.zeros((image_size[0], image_size[1], 1), dtype=np.uint8)

    for poly in inference_data:
        try:
            damage = poly['properties']['subtype']
            polygon = wkt.loads(poly['wkt'])
            poly_np = np.array(polygon.exterior.coords, np.int32)
            fillPoly(mask_img, [poly_np], damage_key.get(damage, 1))
        except Exception as e:
            print(f"Skipping polygon due to error: {e}")
            continue

    return mask_img


def save_image(mask, output_path):
    """
    Saves the grayscale mask to the specified output path.

    Args:
        mask (np.ndarray): The image array.
        output_path (str): Path to save the image.
    """
    imwrite(output_path, mask)
    print(f"Inference image saved at {output_path}")


def create_inference_image(json_input_path, image_output_path):
    """
    Main function to generate and save the inference image from JSON.

    Args:
        json_input_path (str): Path to the inference results JSON file.
        image_output_path (str): Path where the mask image will be saved.
    """
    inference_data = open_json(json_input_path)
    mask = create_image(inference_data)
    save_image(mask, image_output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert inference JSON to grayscale mask image.")
    parser.add_argument('--json_input_path', required=True, help='Path to inference JSON file')
    parser.add_argument('--image_output_path', required=True, help='Path to save grayscale mask image')

    args = parser.parse_args()
    create_inference_image(args.json_input_path, args.image_output_path)


if __name__ == '__main__':
    main()
