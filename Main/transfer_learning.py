import os
import sys
import numpy as np
from os.path import dirname
import tensorflow as tf
from datetime import datetime

import sklearn
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

sys.path.append(dirname("./modules/"))
from preprocessing import join_npy_data
import inception
from inception import transfer_values_cache, transfer_values



inception.maybe_download()

"""
add a reference for the three functions
"""

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

class Transfer_net:
    def __init__(self):
        self.backend_model = inception.Inception()

    def create_network(self, layers, neurons, dropout_rate, num_output_classes):
        """creates a network used to evaluate the transfer values
        """
        self.num_output_classes = num_output_classes
        model = keras.Sequential()

        for i in range(int(layers)):
            model.add(keras.layers.Dense(int(neurons), activation="relu"))
            model.add(keras.layers.Dropout(dropout_rate))

        model.add(keras.layers.Dense(self.num_output_classes, activation='softmax'))
        self.model = model

    def load_transfer_data(self, images):
        """
        Transforms image values into transfer values of the last layer of the inception network
        """
        transfer_values = transfer_values(images=images, model=self.backend_model)
        return transfer_values

    def load_or_cache_transfer_data(self, images, file_path):
        """Function that returns raw images into transfer values and saves them
        """
        if file_path[-4:] != ".npy":
            file_path = file_path + ".npy"
        transfer_values = transfer_values_cache(cache_path=file_path,
                                                images=images,
                                                model=self.backend_model)
        return transfer_values

    def train(self, x_train, y_train, learning_rate, epochs, batch_size, tb_logs_dir=None, verbose=False):
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=25)
        callbacks = [early_stopping_callback]

        if bool(tb_logs_dir):
            date_time = datetime.now().strftime('%Y-%m-%d-%H%M%S')
            log_name = os.path.join(tb_logs_dir, "{}_{}".format("transfer_net", date_time))
            # defining callbacks for training
            tensorboard_callback = TensorBoard(log_dir=log_name, write_graph=True, write_images=True)
            callbacks += [tensorboard_callback]

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state = 1)
        y_train = tf.keras.utils.to_categorical(y_train, self.num_output_classes)
        y_val = tf.keras.utils.to_categorical(y_val, self.num_output_classes)

        self.model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy', f1_score])
        self.hist = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                       verbose=verbose, callbacks=callbacks, validation_data=(x_val, y_val))

    def predict(self, images):
       if images.shape[1] != 2048:
           images = transfer_values(images=images, model=self.backend_model)
       preds = self.model.predict(images)
       return preds

    def predict_classes(self, images):
        if images.shape[1] != 2048:
            images = transfer_values(images=images, model=self.backend_model)
        preds = self.model.predict_classes(images)
        return preds

    def save_model(self, file_path):
        if file_path[-5:] != ".HDF5":
            file_path += ".HDF5"
        tf.keras.models.save_model(self.model,
                                   file_path,
                                   overwrite=True,
                                   include_optimizer=False) # we dont need the optimizer as we only finished ready models
        print("Model was saved to {}".format(file_path))

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path, compile=False)
        print("Model loaded successfully")



if __name__ == "__main__":

    inception.maybe_download()
    file_path=['../Data/np_files4/car_image_package_train_val_split_0.npy']
    x_train, y_train, x_test, y_test, conversion = join_npy_data(file_path, training_data_only=False)

    x_train, y_train, x_test, y_test = x_train[:20], y_train[:20], x_test[:20], y_test[:20]
    y_train[0] = 2

    t_net = Transfer_net()
    t_net.create_network(layers=5, neurons=100, dropout_rate=0.5, num_output_classes=3)
    x_train = t_net.load_or_cache_transfer_data(x_train, file_path="../Data/transfernet_files/x_train1")
    t_net.train(x_train, y_train, learning_rate=1e-06, epochs=5, batch_size=256, verbose=True, tb_logs_dir="../logs")

    t_net.save_model("./ello_test_trans.HDF5")

    t_net2 = Transfer_net()
    t_net2.load_model("./ello_test_trans.HDF5")

    preds = t_net2.predict_classes(x_test)

    sklearn.metrics.accuracy_score(y_test, preds)
    sklearn.metrics.f1_score(y_test, preds)
    sklearn.metrics.confusion_matrix(y_test, preds)
































#
