import os

import numpy as np
import pandas as pd

from tpu_v3 import cnn_model

# parameters:
NUM_MODELS = 3
TRAIN_ON_TPU = True
FOLDER_PATH = "/Users/dominiquepaul/xBachelorArbeit/Daten/3-Spring19/ensemble_savings"

class Cnn_ensemble_model:
    def __init__(self):
        self.models = []

    def new_ensemble_model(self, num_models, x_data, y_data, num_classes, config):
        for i in range(num_models):
            self.models[i] = cnn_model()
            self.models[i].new_model(x_data=x_data, y_data=y_data, num_classes=num_classes, config=config)
        self.num_models = num_models

    def train_models(self, epochs, batch_size, log_dir=None, on_tpu=False):
        for i in range(self.num_models):
            self.models[i].train(log_dir=log_dir, on_tpu=on_tpu)

    def predict_class(self, x_data):
        predictions = []
        for i in range(self.num_models):
            single_model_predictions = self.models[i].predict_class
            predictions += [single_model_predictions]

        predictions = np.stack(predictions)
        pred_df = pd.DataFrame(predictions)
        ensemble_predictions = np.array(pred_df.mode(axis=1))
        return ensemble_predictions

    def save_ensemble(self, folder_name, overwrite=False):
        """
        Saves models to a new folder with a specific name
        """
        if os.path.exists(folder_name):
            if overwrite:
                print("Overwriting existing folder...")
                os.makedirs(folder_name)
                for i in range(self.num_models):
                    model_file_path = os.path.join(folder_name, "ensemble_model_{}".format(i))
                    self.models[i].save_model(folder_path=None, name=model_file_path)
            else:
                raise ValueError("folder exists and no permission to overwrite. If you are sure you want to overwrite this folder then set the function parameter 'overwrite' to True")
        else:
            os.makedirs(folder_name)
            for i in range(self.num_models):
                model_file_path = os.path.join(folder_name, "ensemble_model_{}".format(i))
                self.models[i].save_model(folder_path="", file=model_file_path)
        print("All Models of the ensemble saved to folder {}".format(folder_name))

    def load_ensemble(self, folder_path):
        if os.path.exists(folder_path) is False:
            raise  ValueError("Folder not found. Please be sure that you have specified a folder and not a file path")
        else:
            for i,(dirpath, dirnames, filename) in enumerate(os.walk(folder_path)):
                if ".HDF5" in filename:
                    model_file_path = os.path.join(folder_path, filename)
                    self.models[i] = cnn_model()
                    self.models[i].load_model(model_file_path)
                else:
                    print("{} was not considered as it does not appear to be a saved model (.HDF5 extension missing)".format(filename))
            print("{} models were loaded".format(len(self.models)))



# ideally load this from the best paramaters found in hyperoptimisation before
if __name__ == "__main__":

    config_v1 = {
        "conv_layers": 2,
        "conv_filters": 64,
        "dense_layers": 2,
        "dense_neurons": 10,
        "dropout_rate_dense": 0.2,
        "learning_rate": 1e-04,
        "activation_fn": "relu"
    }

    x_train_url = 'gs://data-imr-unisg/np_array_files/x_train.npy'
    y_train_url = 'gs://data-imr-unisg/np_array_files/class_labels_trainp.npy'


    from io import BytesIO
    from tensorflow.python.lib.io import file_io

    x_test_url = "gs://data-imr-unisg/np_array_files/x_test.npy"
    f1 = BytesIO(file_io.read_file_to_string(x_test_url, binary_mode=True))
    x_test = np.load(f1)

    em1 = Cnn_ensemble_model()
    em1.new_ensemble_model(NUM_MODELS, x_train_url, y_train_url, 2, config_v1)
    em1.train_models(epochs=1, batch_size=256, on_tpu=TRAIN_ON_TPU)
    em1.save_ensemble(FOLDER_PATH)

    em2 = Cnn_ensemble_model()
    em2.load_ensemble(FOLDER_PATH)
    predictions = em2.predict_class(x_test[1:10])

    print(predictions)
