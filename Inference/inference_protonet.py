





"""
inference_protonet.py

Runs inference using a trained Prototypical Network model on a labeled dataset.
Outputs predictions as a JSON file mapping each image UUID to a damage category.

"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from data_generator import DataGenerator
from util.tensor_op import reduce_tensor, slice_tensor_and_sum, reshape_query
from util.loss import proto_dist

# Label decoding
damage_intensity_encoding = {
    3: 'destroyed',
    2: 'major-damage',
    1: 'minor-damage',
    0: 'no-damage'
}


def run_inference(test_data_dir, test_csv_path, model_path, output_json_path):
    """
    Loads a trained ProtoNet model and generates damage predictions for test images.

    Args:
        test_data_dir (str): Path to directory with test images.
        test_csv_path (str): CSV file containing image UUIDs and labels.
        model_path (str): Path to the saved model (.keras or .h5).
        output_json_path (str): Path to save the prediction JSON.
    """
    if not os.path.exists(test_csv_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Test CSV or model weights file not found.")

    print(f"Loading model weights from: {model_path}")
    model = load_model(
        model_path,
        custom_objects={
            'reduce_tensor': reduce_tensor,
            'slice_tensor_and_sum': slice_tensor_and_sum,
            'reshape_query': reshape_query,
            'proto_dist': proto_dist,
            'tf': tf
        }
    )

    df = pd.read_csv(test_csv_path)
    print(f"Found {len(df)} test samples.")

    test_gen = DataGenerator(
        csv_file=test_csv_path,
        data_dir=test_data_dir,
        way=4,
        query=20,
        shot=20,
        num_batch=20
    )

    print(f"Data generator initialized with {len(test_gen)} batches.")
    print("Running predictions...")
    predictions = model.predict(test_gen)
    predicted_indices = np.argmax(predictions, axis=1)
    print(f"Inference complete for {len(predicted_indices)} samples.")

    # Construct JSON output
    predictions_json = {}
    for i, row in df.iterrows():
        filename = row["uuid"].split(".")[0]
        try:
            predictions_json[filename] = damage_intensity_encoding[predicted_indices[i]]
        except Exception as e:
            print(f"Error with sample {filename}: {e}")
            continue

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(predictions_json, f, indent=4)
    print(f"Saved predictions to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on test data using a trained ProtoNet model.")
    parser.add_argument('--test_data_dir', required=True, help='Directory containing test images')
    parser.add_argument('--test_csv_path', required=True, help='CSV with image UUIDs and labels')
    parser.add_argument('--model_path', required=True, help='Path to the trained .keras or .h5 model file')
    parser.add_argument('--output_json_path', required=True, help='Path to save the output predictions JSON')

    args = parser.parse_args()

    run_inference(
        test_data_dir=args.test_data_dir,
        test_csv_path=args.test_csv_path,
        model_path=args.model_path,
        output_json_path=args.output_json_path
    )


if __name__ == '__main__':
    main()




