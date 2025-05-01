"""
Adopted from: https://github.com/EftyK/FSL_for_urban_damage/protonet/data_generator.py
"""



"""
train.py

Trains an EfficientNetB7-based or ResNet50_based classifier for multi-class image classification (e.g., building damage detection).
Uses Keras' ImageDataGenerator, training/validation CSVs, and callbacks like checkpointing, early stopping, and LR reduction.

"""

import argparse
import os
import json
import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    ReduceLROnPlateau,
    EarlyStopping
)
from data_generator import DataGenerator
from model import efficientnet_classification_model # can be replaced with ResNet50_based classifier


def parse_args():
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train EfficientNetB7 classifier")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device index to use")
    parser.add_argument('--data_dir', required=True, help="Directory containing image data")
    parser.add_argument('--train_csv', required=True, help="Path to training CSV file")
    parser.add_argument('--val_csv', required=True, help="Path to validation CSV file")
    parser.add_argument('--model_in', default=None, help="Path to input model weights (optional)")
    parser.add_argument('--model_out', required=True, help="Path to save best model weights")
    parser.add_argument('--train_history', required=True, help="Path to save training history JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Model setup
    model = efficientnet_classification_model(input_shape=(224, 224, 3), num_classes=4)
    if args.model_in:
        print(f"Loading model weights from {args.model_in}")
        model.load_weights(args.model_in)

    # Callbacks
    log_dir = os.path.join(
        "/content/drive/MyDrive/Thesis/ProNet/logs",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        ModelCheckpoint(
            filepath=args.model_out,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            mode="min"
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            verbose=1,
            min_lr=1e-6
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            start_from_epoch=5
        )
    ]

    # Data generators
    batch_size = 32
    train_generator = DataGenerator(
        csv_file=args.train_csv,
        data_dir=args.data_dir,
        batch_size=batch_size,
        dim=(224, 224),
        num_classes=4
    )
    val_generator = DataGenerator(
        csv_file=args.val_csv,
        data_dir=args.data_dir,
        batch_size=batch_size,
        dim=(224, 224),
        num_classes=4
    )

    # Model training
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    with open(args.train_history, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved to {args.train_history}")

    # Save final model
    model.save(args.model_out)
    print("Training completed and model saved.")


if __name__ == "__main__":
    main()



