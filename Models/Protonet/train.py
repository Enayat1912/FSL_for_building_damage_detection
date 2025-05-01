
"""
Adopted from: https://github.com/EftyK/FSL_for_urban_damage/blob/main/protonet/data_generator.py
"""



"""
train.py

Trains a few-shot learning model (Prototypical Network) for building damage detection using custom
convolutional architecture and a metric-based distance classifier.

This script supports loading input data via CSVs, uses episodic training for FSL, and saves model weights
and training history.

"""

import argparse
import os
import json
import datetime

import tensorflow as tf
from tensorflow.keras import callbacks as cb
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Lambda

import numpy as np
import random

from data_generator import DataGenerator
from model import conv_net, proto_dist, reshape_query, reduce_tensor
from util.tensor_op import *
from util.loss import *


def parse_args():
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a prototypical network for building damage detection.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU index to use")
    parser.add_argument('--data_dir', required=True, help="Directory containing image data")
    parser.add_argument('--train_csv', required=True, help="Path to training CSV file")
    parser.add_argument('--val_csv', required=True, help="Path to validation CSV file")
    parser.add_argument('--model_out', required=True, help="Path to save the best model")
    parser.add_argument('--model_in', default=None, help="Path to load model weights (optional)")
    parser.add_argument('--train_history', required=True, help="Path to save training history JSON")
    return parser.parse_args()


def scheduler(epoch, lr):
    """
    Custom learning rate scheduler that halves the learning rate every 10 epochs.

    Args:
        epoch (int): Current epoch number.
        lr (float): Current learning rate.

    Returns:
        float: Updated learning rate.
    """
    if epoch > 0 and epoch % 10 == 0:
        return lr / 2
    return lr


def build_model(input_shape):
    """
    Construct the full model for prototypical network training.

    Args:
        input_shape (tuple): Shape of the input tensor, e.g., (None, 128, 128, 3)

    Returns:
        tf.keras.Model: Compiled model ready for training.
    """
    conv = conv_net()
    conv_5d = TimeDistributed(conv)

    sample = Input(input_shape)
    out_feature = conv_5d(sample)
    out_feature = Lambda(reduce_tensor)(out_feature)

    inp = Input(input_shape)
    map_feature = conv_5d(inp)
    map_feature = Lambda(reshape_query)(map_feature)

    pred = Lambda(proto_dist)([out_feature, map_feature])  # negative distance
    model = Model([sample, inp], pred)

    return model


def main():
    """
    Main training loop for prototypical network model.

    Loads data, builds the model, sets callbacks, performs training,
    and saves the model and training history.
    """
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Model and input config
    input_shape = (None, 128, 128, 3)
    batch_size = 16
    train_way = 4
    train_query = 20
    val_way = 4
    shot = 20
    base_lr = 0.001

    log_dir = os.path.join(
        "/content/drive/MyDrive/Thesis/ProNet/logs",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # Build and compile model
    model = build_model(input_shape)
    if args.model_in:
        print(f"Loading model weights from {args.model_in}")
        model.load_weights(args.model_in)

    model.compile(
        optimizer=Adam(base_lr),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    # Data loaders
    train_loader = DataGenerator(
        csv_file=args.train_csv,
        data_dir=args.data_dir,
        way=train_way,
        query=train_query,
        shot=shot,
        num_batch=32
    )

    val_loader = DataGenerator(
        csv_file=args.val_csv,
        data_dir=args.data_dir,
        way=val_way,
        shot=shot
    )

    # Callbacks
    callbacks = [
        cb.TensorBoard(log_dir=log_dir, histogram_freq=1),
        cb.ModelCheckpoint(
            filepath=args.model_out,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        cb.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            start_from_epoch=1
        ),
        cb.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.4,
            patience=2,
            min_lr=1e-8
        ),
        cb.LearningRateScheduler(scheduler)
    ]

    # Train model
    print(model.summary())
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    with open(args.train_history, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved to {args.train_history}")
    print("END OF TRAINING")


if __name__ == "__main__":
    main()
