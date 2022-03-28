import tensorflow as tf
import rasterio as rio
import netCDF4 as nc
import numpy as np
import argparse
import random
import json
import os


from make_config import *
import resnet


def train_test_split(files, split):

    """ Split data into training and validation sets """

    train_num = int(len(files) * split)

    train = random.sample(files, train_num)
    val = [i for i in files if i not in train]

    return train, val


def get_train():

    """ Training data generator """

    for file in train_files:
        print("Train File: ", file)
        img = ( np.array(nc.Dataset(file, "r")["ims"][0:1])[0], np.array(nc.Dataset(file, "r")["migrants"][0:1]) )
        yield img


def get_val():

    """ Validation data generator """

    for file in val_files:
        print("Val File: ", file)
        img = ( np.array(nc.Dataset(file, "r")["ims"][0:1])[0], np.array(nc.Dataset(file, "r")["migrants"][0:1]) )
        yield img




if __name__ == "__main__":

    #################################################################################
    # Parse Input Agruments
    #################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("rank")
    parser.add_argument("world_size")
    args = parser.parse_args()

    print("World size: ", args.world_size)


    #################################################################################
    # Initialize the TF_CONFIG variable with the nodes participating in training
    #################################################################################
    ips_direc = "/sciclone/home20/hmbaier/tflow/ips/"
    os.environ["TF_CONFIG"] = make_config(int(args.rank) - 1, ips_direc, 45962)
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


    #################################################################################
    # Split the imagery file names into training & validation sets
    #################################################################################
    IMAGERY_DIR = "/sciclone/scr-mlt/hmbaier/cropped/"
    files = os.listdir(IMAGERY_DIR)
    files = [IMAGERY_DIR + i for i in files if "484" in i]

    print("Number of nCDF files: ", len(files))

    train_files, val_files = train_test_split(files, .75)
    train_dataset = tf.data.Dataset.from_generator(generator = get_train, output_types=(tf.float32, tf.float32)).batch(int(args.world_size))
    val_dataset = tf.data.Dataset.from_generator(generator = get_val, output_types=(tf.float32, tf.float32)).batch(int(args.world_size))

    print("Number of training files: ", len(train_files))
    print("Number of validation files: ", len(val_files)) 


    #################################################################################
    # Set up the model and distribute the dataset using distributed strategy
    #################################################################################
    with strategy.scope():

        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = strategy.experimental_distribute_dataset(val_dataset)

        multi_worker_model = resnet.resnet56(img_input = tf.keras.layers.Input((None, None, 3)), classes = 1)

        multi_worker_model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss = tf.keras.losses.MeanAbsoluteError(),
            metrics=[tf.keras.losses.MeanAbsoluteError()]
        )

    print("Number of steps per epoch: ", int(len(files) // int(args.world_size)))


    #################################################################################
    # Train the model
    #################################################################################
    multi_worker_model.fit(train_dataset,
                        steps_per_epoch = int(len(files) // int(args.world_size)),
                        validation_data = val_dataset)
