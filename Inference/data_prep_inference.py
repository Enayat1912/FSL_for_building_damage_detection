from PIL import Image
import numpy as np
import pandas as pd
import cv2
import os
import json
import shapely.wkt
from shapely.geometry import Polygon
from collections import defaultdict

damage_intensity_encoding = defaultdict(lambda: 0)
damage_intensity_encoding['destroyed'] = 3
damage_intensity_encoding['major-damage'] = 2
damage_intensity_encoding['minor-damage'] = 1
damage_intensity_encoding['no-damage'] = 0

def process_img(img_array, polygon_pts, scale_pct):
    """Process Raw Data into extracted polygon images."""
    height, width, _ = img_array.shape

    # Find the four corners of the polygon
    xcoords = polygon_pts[:, 0]
    ycoords = polygon_pts[:, 1]
    xmin, xmax = np.min(xcoords), np.max(xcoords)
    ymin, ymax = np.min(ycoords), np.max(ycoords)
  

    xdiff = xmax - xmin
    ydiff = ymax - ymin

    # Extend image by scale percentage
    xmin = max(int(xmin - (xdiff * scale_pct)), 0)
    xmax = min(int(xmax + (xdiff * scale_pct)), width)
    ymin = max(int(ymin - (ydiff * scale_pct)), 0)
    ymax = min(int(ymax + (ydiff * scale_pct)), height)

    return img_array[ymin:ymax, xmin:xmax, :]

def process_img_poly(img_path, label_path, output_dir, output_csv):
    """
    Process the input image and labels to extract building polygons and save them as individual images.
    """
    x_data = []
    y_data = []
    wkt_data = []
    
    print(f"Processing image: {img_path}")
    print(f"Reading labels from: {label_path}")

    # Load the input image
    img_obj = Image.open(img_path)
    img_array = np.array(img_obj)

    # Load the label JSON
    with open(label_path, 'r') as label_file:
        label_data = json.load(label_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over features in the JSON
    for feat in label_data['features']['xy']:
        try:
            # Extract damage type and geometry
            damage_type = feat['properties']['subtype']
            polygon_geom = shapely.wkt.loads(feat['wkt'])
            polygon_pts = np.array(list(polygon_geom.exterior.coords))
            
            # Generate a unique filename for the polygon
            poly_uuid = feat['properties']['uid']  # No .png suffix
            image_filename = poly_uuid + ".png"  # Use .png for saving image files

            # Append damage label and WKT to the lists
            y_data.append(damage_intensity_encoding[damage_type])
            wkt_data.append(feat['wkt'])
            poly_img = process_img(img_array, polygon_pts, 0.8)
            
            # Save the polygon image
            cv2.imwrite(os.path.join(output_dir, image_filename), poly_img)
            x_data.append(poly_uuid)  # Append the UID without .png to x_data
            
            print(f"Processed polygon: {image_filename}")
        
        except KeyError as e:
            print(f"Skipping feature due to missing key: {e}")
        except Exception as e:
            print(f"Error processing polygon: {e}")

    # Validate that all lists have the same length
    if len(x_data) != len(y_data) or len(x_data) != len(wkt_data):
        raise ValueError("All arrays (x_data, y_data, wkt_data) must be of the same length")

    # Save results to a CSV file
    data_array = {'uuid': x_data, 'labels': y_data}
    df = pd.DataFrame(data=data_array)
    print(f"Saving CSV to: {output_csv}")
    df.to_csv(output_csv, index=False)



if __name__ == '__main__':
    # Define paths directly in the script
    input_img = "/content/drive/MyDrive/Thesis/ProtoNet/protonet_inference_data/mexico-earthquake_00000076_post_disaster.png"
    label_path = "/content/drive/MyDrive/Thesis/ProtoNet/protonet_inference_data/mexico-earthquake_00000076_post_disaster.json"
    output_dir = "/content/drive/MyDrive/Thesis/ProtoNet/protonet_inference_output/ image"
    output_csv = "/content/drive/MyDrive/Thesis/ProtoNet/protonet_inference_output/inference.csv"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Verify the image file exists
    if not os.path.isfile(input_img):
        raise FileNotFoundError(f"Image file not found: {input_img}")

    # Process the image and save results
    process_img_poly(input_img, label_path, output_dir, output_csv)

