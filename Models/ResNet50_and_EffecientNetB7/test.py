




"""
efficientnet_test.py

Evaluates a trained EfficientNetB7 model on a test dataset using a custom DataGenerator.
Outputs classification metrics including accuracy, confusion matrix, and classification report.


"""

import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_generator import DataGenerator

RANDOM_SEED = 123


def parse_args():
    """
    Parse command-line arguments for testing the model.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate a trained EfficientNet model on test data.')
    parser.add_argument('--path', required=True, help='Path to the saved model')
    parser.add_argument('--test_dir', required=True, help='Path to directory of test images')
    parser.add_argument('--test_csv', required=True, help='Path to test CSV file')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from: {args.path}")
    model = load_model(args.path)

    test_datagen = DataGenerator(
        csv_file=args.test_csv,
        data_dir=args.test_dir,
        batch_size=16,
        dim=(128, 128),
        num_classes=4
    )

    predictions = []
    true_labels = []

    print("Running predictions on test data...")

    for i, (batch_data, batch_labels) in enumerate(test_datagen):
        batch_predictions = model.predict(batch_data, batch_size=32, verbose=0)
        predictions.extend(np.argmax(batch_predictions, axis=-1))
        true_labels.extend(np.argmax(batch_labels, axis=-1))
        print(f"Processed batch {i + 1}/{len(test_datagen)}")

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Total true labels: {len(true_labels)}")

    if len(predictions) > 0 and len(true_labels) > 0:
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))

        print("Confusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
    else:
        print("No predictions or true labels were collected.")


if __name__ == "__main__":
    main()





