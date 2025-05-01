import argparse
import os
import cv2
import datetime 
import json

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', type=int, default=0)
    parser.add_argument('--data_dir', required=True, help="Data directory")
    parser.add_argument('--train_csv', required=True, help="Train CSV file path")
    parser.add_argument('--val_csv', required=True, help="Validation CSV file path")
    parser.add_argument('--model_out', required=True, help="Path to save the model")
    parser.add_argument('--model_in', default=None, help="Path to a saved model")  # Include model_in
    return parser.parse_args()

# Hardcode arguments for Google Colab
class Args:
    gpu = 0
    data_dir = "/content/drive/MyDrive/Thesis/Dataset/equal_dataset/images"
    train_csv = "/content/drive/MyDrive/Thesis/Dataset/equal_dataset/train.csv"
    val_csv = "/content/drive/MyDrive/Thesis/Dataset/equal_dataset/val.csv"
    model_out = "/content/drive/MyDrive/Thesis/ProtoNet/models/model6.keras"
    train_history= "/content/drive/MyDrive/Thesis/ProtoNet/history/history6.json"
    model_in = "/content/drive/MyDrive/Thesis/ProtoNet/ProNet_dataset_model/protonets.hdf5"

args = Args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)




import tensorflow as tf

from tensorflow.keras import callbacks as cb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model, save_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers as rg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model


import numpy.random as rng
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import random
from data_generator import DataGenerator
from model import conv_net, hinge_loss, l2_distance, acc, l1_distance
from util.tensor_op import *
from util.loss import *



input_shape = (None,128,128,3)
batch_size = 16
train_way = 4
train_query = 20
val_way = 4
shot = 20
lr = 0.002


LOG_DIR = '/content/drive/MyDrive/Thesis/ProNet/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def scheduler(epoch):
    global lr
    if epoch % 10 == 0:
        lr /= 2
    return lr

# class SaveConv(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         if epoch % 50 == 0:
#             save_model(conv, f"model/omniglot_conv_{epoch}_{shot}_{val_way}")

if __name__ == "__main__":
    conv = conv_net()
    conv_5d = TimeDistributed(conv)
    sample = Input(input_shape)
    out_feature = conv_5d(sample)
    out_feature = Lambda(reduce_tensor)(out_feature)
    inp = Input(input_shape)
    map_feature = conv_5d(inp)
    map_feature = Lambda(reshape_query)(map_feature)
    pred = Lambda(proto_dist)([out_feature, map_feature]) #negative distance
    combine = Model([sample, inp], pred)

    # Add model weights if provided by user
    if args.model_in is not None:
        combine.load_weights(args.model_in)

    optimizer = Adam(0.001)

    #Stopping when the val loss not improving
    early_stopping = tf. keras.callbacks.EarlyStopping(start_from_epoch=1, monitor='val_loss', patience=10, restore_best_weights=True)

    #Set up tensorboard logging
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)


    #Filepath to save model weights
    filepath = args.model_out 

    checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    monitor="val_loss",  
    verbose=1,
    save_best_only=True,
    mode="min",
)

    combine.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

    train_loader = DataGenerator(csv_file=args.train_csv, data_dir=args.data_dir, way=train_way, query=train_query, shot=shot, num_batch=32)
    val_loader = DataGenerator(csv_file=args.val_csv, data_dir=args.data_dir, way=val_way, shot=shot)


    
    #augmented_train_loader = train_loader.map(lambda x, y: (data_augmentation(x), y))

    print(combine.summary())

    reduce_lr = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=1e-8)

    lr_sched = cb.LearningRateScheduler(scheduler)
    
    tensorboard = cb.TensorBoard()


    #Training begins
    history= combine.fit(train_loader ,
            validation_data=val_loader ,
            epochs=50,
            callbacks=[tensorboard_callbacks, lr_sched, checkpoints,early_stopping],
            verbose=1)

    with open(args.train_history, 'w') as f:

      json.dump(history.history, f)  

    print("Training history saved!")


    print("END OF TRAINING")