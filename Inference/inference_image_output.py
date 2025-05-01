import json
from shapely import wkt
from shapely.geometry import Polygon
import numpy as np
from cv2 import fillPoly, imwrite

def open_json(json_file_path):
    """
    Opens the inference JSON file and extracts the localized polygon data.
    :param json_file_path: Path to open inference JSON file.
    :returns: The JSON data dictionary of localized polygons and their classifications.
    """
    with open(json_file_path) as jf:
        json_data = json.load(jf)
        inference_data = json_data['features']['xy']
        return inference_data

def create_image(inference_data):
    """
    Creates an 8-bit grayscale image with polygons filled according to their classification.
    :param inference_data: JSON data dictionary of localized polygons and classifications.
    :returns: A NumPy array of the grayscale image.
    """
    damage_key = {'un-classified': 1, 'no-damage': 1, 'minor-damage': 2, 'major-damage': 3, 'destroyed': 4}

    # Initialize an empty mask
    mask_img = np.zeros((1024, 1024, 1), np.uint8)

    for poly in inference_data:
        damage = poly['properties']['subtype']
        coords = wkt.loads(poly['wkt'])  # Parse the polygon from WKT format

        # Convert the polygon coordinates to a NumPy array
        poly_np = np.array(coords.exterior.coords, np.int32)
        
        # Fill the polygon on the mask
        fillPoly(mask_img, [poly_np], damage_key[damage])
    
    return mask_img

def save_image(polygons, output_path):
    """
    Saves the filled polygon mask as an image.
    :param polygons: NumPy array with filled polygons from create_image().
    :param output_path: Path to save the final output image.
    """
    imwrite(output_path, polygons)

def create_inference_image(json_input_path, image_output_path):
    """
    Generates the inference image from the JSON output.
    :param json_input_path: Path to the final inference JSON file.
    :param image_output_path: Path to save the final inference image.
    """
    # Get the inference data from the JSON
    inference_data = open_json(json_input_path)

    # Create a mask with filled polygons
    polygon_array = create_image(inference_data)

    # Save the mask as an image
    save_image(polygon_array, image_output_path)

if __name__ == '__main__':
    # Paths for your project
    json_input_path = "/content/drive/MyDrive/Thesis/protonet_inference_output/inference_results.json"
    image_output_path = "/content/drive/MyDrive/Thesis/protonet_inference_output/inference_image.png"

    # Creating the inference image
    create_inference_image(json_input_path, image_output_path)
    print(f"Inference image saved at {image_output_path}")
