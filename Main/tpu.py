"""
to do:
    - go through "train" functions and check whether all parameters are being used
"""


import tensorflow as tf
import numpy as np
import time
import os
from datetime import datetime

from io import BytesIO
from tensorflow.python.lib.io import file_io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision. Computes the precision, a
    metric for multi-label classification of how many selected items are
    relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    """Computes the F1 Score
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r) / (p + r + K.epsilon())

class cnn_model:
    def __init__(self):
        """
        initializes the model and defines the graph. There will always be one more
        dense layer than defined.
        """
        # hard features
        self.optimizer = "adam"
        self.loss = "binary_crossentropy"
        self.tpu_instance_name = "dominique-c-a-paul"

    def new_model(self, x_data, y_data, num_classes, config, name=None):
        # mutable features
        self.conv_layers = int(config["conv_layers"])
        self.conv_filters = int(config["conv_filters"])
        self.conv_stride = (1,1)
        self.kernel_size = (3,3)
        self.pool_size = (2,2)
        self.pool_stride = (2,2)
        self.dense_layers = int(config["dense_layers"])
        self.dense_neurons = int(config["dense_neurons"])
        self.dropout_rate_dense = config["dropout_rate_dense"]
        self.learning_rate = config["learning_rate"]
        self.activation_fn = config["activation_fn"]

        # we give the model a name that describes its parameters
        if bool(name):
            self.name = name
        else:
            self.name = "conv_size_{}_filters_{}_dense_{}_dropout_{}_dense_{}_lr_{}_act_{}".format(self.conv_layers, self.conv_filters,
                                              self.dense_layers, self.dropout_rate_dense, self.dense_layers,
                                              self.learning_rate, self.activation_fn)

        self.x_train, self.x_val, self.y_train, self.y_val = self.load_data(x_data, y_data)
        self.model = self.create_model(num_output_classes=num_classes)


    def load_data(self, x_train, y_train):
        # check whether input is numpy format or a link to google cloud storage
        if isinstance(x_train, str):
            if "gs" in x_train:
                f = BytesIO(file_io.read_file_to_string(x_train, binary_mode=True))
                x_train1 = np.load(f)
            else:
                x_train1 = np.load(x_train)

        if isinstance(y_train, str):
            if "gs" in y_train:
                f = BytesIO(file_io.read_file_to_string(y_train, binary_mode=True))
                y_train1 = np.load(f)
            else:
                y_train1 = np.load(y_train)

        # create train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_train1,
                                                            y_train1,
                                                            train_size=0.8,
                                                            random_state = 1) # random state during training, has to be removed later on
        return(x_train, x_val, y_train, y_val)

    def create_model(self, num_output_classes):
        input_shape = self.x_train.shape[1:]

        # defining the model
        model = Sequential()
        model.add(Conv2D(filters=self.conv_filters,
                         kernel_size=self.kernel_size,
                         activation=self.activation_fn,
                         input_shape=input_shape,
                         padding="SAME",
                         strides=self.conv_stride
                        ))
        model.add(Conv2D(filters=self.conv_filters,
                         kernel_size=self.kernel_size,
                         activation=self.activation_fn,
                         padding="SAME",
                         strides=self.conv_stride
                        ))
        model.add(MaxPooling2D(pool_size=self.pool_size,
                 strides=self.pool_stride))
        model.add(BatchNormalization())

        for _ in range(self.conv_layers-1):
            model.add(Conv2D(filters=self.conv_filters,
                             kernel_size=self.kernel_size,
                             activation=self.activation_fn,
                             padding="SAME",
                             strides=self.conv_stride
                             #input_shape=input_shape
                          ))
            model.add(Conv2D(filters=self.conv_filters,
                             kernel_size=self.kernel_size,
                             activation=self.activation_fn,
                             padding="SAME",
                             strides=self.conv_stride
                             #input_shape=input_shape
                          ))
            model.add(MaxPooling2D(pool_size=self.pool_size,
                     strides=self.pool_stride))
            model.add(BatchNormalization())

        model.add(Flatten())
        for i in range(self.dense_layers):
            model.add(Dense(self.dense_neurons, activation=self.activation_fn))
            model.add(Dropout(self.dropout_rate_dense))
        model.add(Dense(num_output_classes, activation='softmax')) # softmax remains unchanged

        return model

    def train(self, epochs, batch_size, tb_logs_dir=None,learning_rate=None, optimizer=None, loss=None, verbose=False, on_tpu=False):
        """
        trains the model.

        If the initial config file contained parameters for training then
        these dont have to be defined but can still be overridden
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
        if optimizer is None:
            optimizer = self.optimizer
        if loss is None:
            loss = self.loss

        early_stopping_callback = EarlyStopping(monitor="val_loss",
                                                patience=5)
        callbacks = [early_stopping_callback]

        if bool(tb_logs_dir):
            date_time = datetime.now().strftime('%Y-%m-%d-%H%M%S')
            log_name = os.path.join(tb_logs_dir, "{}_{}".format(self.name, date_time))

            # defining callbacks for training
            tensorboard_callback = TensorBoard(log_dir=log_name,
                                    write_graph=True,
                                    write_images=True)
            callbacks += [tensorboard_callback]

        # model has to be compiled differently when on tpu
        if on_tpu:
            self.train_on_tpu(epochs, batch_size, learning_rate, optimizer, loss, callbacks)
        else:
            self.train_on_cpu(epochs, batch_size, learning_rate, optimizer, loss, callbacks, verbose)


    def train_on_cpu(self, epochs, batch_size, learning_rate, optimizer, loss, callbacks, verbose):
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 2 )
        self.y_val = tf.keras.utils.to_categorical(self.y_val, 2 )
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', f1_score])
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       verbose=verbose, callbacks=callbacks, validation_data=(self.x_val, self.y_val))

    def train_on_tpu(self, epochs, batch_size, learning_rate, optimizer, loss, callbacks):
        self.model = tf.contrib.tpu.keras_to_tpu_model(self.model, strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(self.tpu_instance_name)))
        self.model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3, ), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['sparse_categorical_accuracy', f1_score])

        # has to be optimised to really train a epoch with full data
        self.hist = self.model.fit_generator(
            self.train_gen(batch_size),
            epochs=epochs,
            steps_per_epoch=10, # still have to change this
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks
            )

        self.model = self.model.sync_to_cpu()

    def train_gen(self, batch_size):
        """
        Generator function for train_on_tpu which provides batches of data
        generator function for training the model on a tpu
        """
        while True:
            offset = np.random.randint(0, self.x_train.shape[0] - batch_size)
            # print(self.x_train[offset:offset+batch_size].shape, self.y_train[offset:offset + batch_size].shape)
            yield self.x_train[offset:offset+batch_size], self.y_train[offset:offset + batch_size]

    def predict(self, x_data):
        predictions = self.model.predict(x_data)
        return predictions

    def predict_classes(self,x_data):
        predicted_classes = self.model.predict_classes(x_data)
        return(predicted_classes)


    def save_model(self, folder_path="", file=None):
        if bool(file_path) == False:
            file = self.name
        if bool(folder_path):
            file_path = os.path.join(folder_path, file + ".HDF5")
        tf.keras.models.save_model(self.model,
                                   file_path,
                                   overwrite=True,
                                   include_optimizer=False) # we dont need the optimizer as we only finished ready models
        print("Model: {} was saved".format(name))

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path, compile=False)
        print("Model loaded successfully")

if __name__ == "__main__":

    config_v1 = {
        "conv_layers": 4,
        "conv_filters": 128,
        "dense_layers": 5,
        "dense_neurons": 20,
        "dropout_rate_dense": 0.2,
        "learning_rate": 1e-04,
        "activation_fn": "relu"
    }

    # offline
    x_train = "/Users/dominiquepaul/xBachelorArbeit/Daten/3-Spring19/1-OwnNetwork/np_array_files/x_train.npy"
    y_train = "/Users/dominiquepaul/xBachelorArbeit/Daten/3-Spring19/1-OwnNetwork/np_array_files/class_labels_train.npy"
    # online
    x_train_url = 'gs://data-imr-unisg/np_array_files/x_train.npy'
    y_train_url = 'gs://data-imr-unisg/np_array_files/class_labels_trainp.npy'

    # model
    print("New Model")
    m1 = cnn_model()
    m1.new_model(x_train_url, y_train_url, 2, config_v1)
    print("Train model...")
    m1.train(epochs=2, batch_size=256,on_tpu=True, tb_logs_dir="gs://data-imr-unisg/logs/")
    print("Save Model")
    m1.save_model(file_path="my_model")

    print("New Model 2")
    m2 = cnn_model()
    print("Load model 2")
    m2.load_model("my_model.HDF5")

    print("Everything worked")
