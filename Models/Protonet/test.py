"""

Code adopted from: https://github.com/EftyK/FSL_for_urban_damage.git

"""


"""
test.py

Evaluates a trained Prototypical Network model using a custom DataGenerator.
Generates predictions and calculates classification metrics like accuracy, 
confusion matrix, and a classification report.


"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from data_generator import DataGenerator
from util.tensor_op import reduce_tensor, reshape_query, proto_dist

RANDOM_SEED = 123


def parse_args():
    """
    Parse command-line arguments for model evaluation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained model on test data.')
    parser.add_argument('--path', required=True, help='Path to the saved model')
    parser.add_argument('--test_dir', required=True, help='Path to test image directory')
    parser.add_argument('--test_csv', required=True, help='Path to test CSV file')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from: {args.path}")
    model = load_model(
        args.path,
        custom_objects={
            'tf': tf,
            'reduce_tensor': reduce_tensor,
            'reshape_query': reshape_query,
            'proto_dist': proto_dist
        }
    )

    test_datagen = DataGenerator(
        csv_file=args.test_csv,
        data_dir=args.test_dir,
        way=4,
        shot=5,
        query=5,
        num_batch=10,
        random_seed=RANDOM_SEED
    )

    predictions = []
    true_labels = []

    max_batches = 10
    print("Running predictions...")

    for i, (inputs, labels) in enumerate(test_datagen):
        if i >= max_batches:
            break

        batch_predictions = model.predict(inputs, batch_size=24)
        predictions.extend(np.argmax(batch_predictions, axis=-1))
        true_labels.extend(np.argmax(labels, axis=-1))

        print(f"Processed batch {i + 1}/{min(len(test_datagen), max_batches)}")

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    print(f"Total predictions: {len(predictions)}")
    print(f"Total true labels: {len(true_labels)}")

    if len(predictions) > 0 and len(true_labels) > 0:
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))

        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
    else:
        print("No predictions or true labels were collected.")


if __name__ == "__main__":
    main()





