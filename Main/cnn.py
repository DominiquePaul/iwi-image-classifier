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

from preprocessing import join_npy_data

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


    def new_model(self, x_data, y_data, config, name=None):
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
        self.activation_fn = "relu"
        # detecting how many different classes the training data contains
        self.num_output_classes = len(set(y_data))

        # we give the model a name that describes its parameters
        if bool(name):
            self.name = name
        else:
            self.name = "conv_size_{}_filters_{}_dense_{}_dropout_{}_dense_{}_lr_{}_".format(self.conv_layers, self.conv_filters,
                                              self.dense_layers, self.dropout_rate_dense, self.dense_layers,
                                              self.learning_rate)

        # create train and validation sets
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_data,
                                                             y_data,
                                                             train_size=0.8,
                                                             random_state = 1) # random state during training, has to be removed later on

        self.model = self.create_model()

    def create_model(self):

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
        model.add(Dense(self.num_output_classes, activation='softmax')) # softmax remains unchanged

        return model

    def train(self, epochs, batch_size, tb_logs_dir=None, on_tpu=None, verbose=False):
        """
        trains the model.

        If the initial config file contained parameters for training then
        these dont have to be defined but can still be overridden
        """

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
        if bool(on_tpu):
            self.train_on_tpu(on_tpu, epochs, batch_size, callbacks)
        else:
            self.train_on_cpu(epochs, batch_size, callbacks, verbose)


    def train_on_cpu(self, epochs, batch_size, callbacks,verbose):
        print("converting y_train")
        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_output_classes)
        print("converting y_val")
        self.y_val = tf.keras.utils.to_categorical(self.y_val, self.num_output_classes)
        print("compiling")
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy', f1_score])
        print("training")
        self.hist = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       verbose=verbose, callbacks=callbacks, validation_data=(self.x_val, self.y_val))

    def train_on_tpu(self, tpu_instance_name, epochs, batch_size, optimizer, callbacks):
        self.model = tf.contrib.tpu.keras_to_tpu_model(self.model, strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(tpu_instance_name)))
        self.model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate, ), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['sparse_categorical_accuracy', f1_score])

        # has to be optimised to really train a epoch with full data
        self.hist = self.model.fit_generator(
            self.train_gen(batch_size),
            epochs=epochs,
            steps_per_epoch=10, #len(self.x_train)//batch_size,
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
            yield self.x_train[offset:offset+batch_size], self.y_train[offset:offset + batch_size]

    def predict(self, x_data):
        predictions = self.model.predict(x_data)
        return predictions

    def predict_classes(self,x_data):
        predicted_classes = self.model.predict_classes(x_data)
        return(predicted_classes)

    def save_model(self, folder_path="", file=None):
        if bool(file) == False:
            file = self.name
        file_path = os.path.join(folder_path, file)
        if file_path[-5:] != ".HDF5":
            file_path += ".HDF5"

        tf.keras.models.save_model(self.model,
                                   file_path,
                                   overwrite=True,
                                   include_optimizer=False) # we dont need the optimizer as we only finished ready models
        print("Model: {} was saved".format(file))

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path, compile=False)
        print("Model loaded successfully")



if __name__ == "__main__":

    file_path=['../Data/np_files4/car_image_package_train_val_split_0.npy']

    x_train, y_train, x_test, y_test, conversion = join_npy_data(file_path, training_data_only=False)

    x_train, y_train, x_test, y_test = x_train[:20], y_train[:20], x_test[:20], y_test[:20]

    # define the general parameters of the model
    config_v1 = {
        "conv_layers": 4,
        "conv_filters": 128,
        "dense_layers": 5,
        "dense_neurons": 20,
        "dropout_rate_dense": 0.2,
        "learning_rate": 1e-04,
    }

    # instantiate the class
    m1 = cnn_model()
    # construct a new model and the configuration file and data to be usef for training
    m1.new_model(x_data=x_train, y_data=y_train, config=config_v1)
    # start training and state where to save the logs
    m1.train(epochs=2, batch_size=256, on_tpu=None, tb_logs_dir="./log_files", verbose=True)
    # save the trained model with its parameters
    m1.save_model(file="./model_v1.HDF5")

    # to load a previosuly saved model we create a new instance of the cnn class
    m2 = cnn_model()
    # load the model paramters from our disk
    m2.load_model(file_path="./model_v1.HDF5")
    # predict and print the classes for previously unseen images
    print(m2.predict_classes(x_test))

    print(m2.predict(x_test))



"""
config_v1 = {
    "conv_layers": 4,
    "conv_filters": 128,
    "dense_layers": 5,
    "dense_neurons": 20,
    "dropout_rate_dense": 0.2,
    "learning_rate": 1e-04,
}


data_url=['../data/np_files4/car_image_package_train_val_split_0.npy']
x_train_alt, y_train_alt, x_test_alt, y_test_alt, conversion = join_npy_data(data_url, training_data_only=False)


y_train_alt

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

x_train, y_train,x_test, y_test = x_train[:20], y_train[:20],x_test[:20], y_test[:20]

x_train.shape

import cv2
x_train_new = []
for x in x_train:
    x_train_new.extend([cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)])
x_train_new = np.array(x_train_new)

x_test_new = []
for x in x_test:
    x_test_new.extend([cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)])
x_test_new = np.array(x_test_new)

x_test_new.shape

y_train_alt[0]=2

# model
m1 = cnn_model()
m1.new_model(x_train_new, y_train_alt[:20], 3, config_v1)
m1.train(epochs=2, batch_size=256, on_tpu=None, tb_logs_dir=None, verbose=True)
m1.save_model(file="my_model_mnist.HDF5")

m2 = cnn_model()
m2.load_model("my_model_mnist.HDF5")
print(m2.predict_classes(x_test_new))
"""












#
