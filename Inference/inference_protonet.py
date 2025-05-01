import sys
sys.path.append('/content/drive/MyDrive/Thesis/ProtoNet/protonet_train_test')  # Import protonet modules

from PIL import Image
import time
import numpy as np
import pandas as pd
import os
import json
import datetime

import tensorflow as tf
from keras.models import load_model
from data_generator import DataGenerator
from util.tensor_op import *
from util.loss import proto_dist
from util.tensor_op import reduce_tensor, slice_tensor_and_sum, reshape_query


# Damage intensity encoding
damage_intensity_encoding = {
    3: 'destroyed',
    2: 'major-damage',
    1: 'minor-damage',
    0: 'no-damage'
}

def run_inference(test_data, test_csv, model_weights, output_json_path):
    # Load the pre-trained model

    # Load the model with custom objects
    print(f"Loading model weights from: {model_weights}")
    model = load_model(
        model_weights,
        custom_objects={
            'reduce_tensor': reduce_tensor,
            'slice_tensor_and_sum': slice_tensor_and_sum,
            'reshape_query': reshape_query,
            'proto_dist': proto_dist,
            'tf': tf  
        }
    )

    # Load test data CSV
    df = pd.read_csv(test_csv)
    samples = df["uuid"].count()

    print(f"Number of samples in test data: {samples}")

    test_gen = DataGenerator(
        csv_file=test_csv,
        data_dir=test_data,
        way=4,
        query=20,
        shot=20,
        num_batch=20
    )

    print(f"DataGenerator initialized with {len(test_gen)} batches.")


    # Run predictions
    print("Running predictions...")
    predictions = model.predict(test_gen)
    predicted_indices = np.argmax(predictions, axis=1)
    print(f"Number of predictions: {len(predicted_indices)}")

    # Create predictions JSON
    predictions_json = {}
    for i, row in df.iterrows():
        filename_raw = row["uuid"]
        filename = filename_raw.split(".")[0]
        try:
            predictions_json[filename] = damage_intensity_encoding[predicted_indices[i]]
        except Exception as e:
            print(f"Error for file {filename_raw}: {e}")
            continue

    # Save predictions to JSON
    print(f"Saving predictions to: {output_json_path}")
    with open(output_json_path, 'w') as outfile:
        json.dump(predictions_json, outfile, indent=4)
    print("Predictions saved successfully.")

if __name__ == '__main__':
    # Define paths for your dataset
    test_data = "/content/drive/MyDrive/Thesis/ProtoNet/protonet_inference_output/image"
    test_csv = "/content/drive/MyDrive/Thesis/ProtoNet/protonet_inference_output/inference.csv"
    model_weights = "/content/drive/MyDrive/Thesis/ProtoNet/models/saved_model.keras"
    output_json = "/content/drive/MyDrive/Thesis/ProtoNet/protonet_inference_output/inference.json"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # Run inference
    run_inference(test_data, test_csv, model_weights, output_json)



