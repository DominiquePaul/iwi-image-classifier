import numpy as np
import pandas as pd

from tpu_v3 import cnn_model


# parameters:
NUM_MODELS = 10
TRAIN_ON_TPU = True

# offline
x_train = "/Users/dominiquepaul/xBachelorArbeit/Daten/3-Spring19/1-OwnNetwork/np_array_files/x_train.npy"
y_train = "/Users/dominiquepaul/xBachelorArbeit/Daten/3-Spring19/1-OwnNetwork/np_array_files/class_labels_train.npy"
# online
x_train_url = 'gs://data-imr-unisg/np_array_files/x_train.npy'
y_train_url = 'gs://data-imr-unisg/np_array_files/class_labels_trainp.npy'

x_pred_raw = None

# build models
models = []

# ideally load this from the best paramaters found in hyperoptimisation before
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

for i in range(NUM_MODELS):
    models[i] = cnn_model()
    models[i].new_model(x_train_url, y_train_url, 2, config_v1, name="ensemble_model_v1_{}".format(i))
    models[i].train(on_tpu=True, epochs= 10, batch_size=256)
    
# prediction
predictions = []
for i in range(len(models)):
    ind_predictions = models[i].predict(x_pred_raw)
    predictions += [ind_predictions]

# merge Predictions 
predictions = np.stacked(predictions)
pred_df = pd.DataFrame(predictions)
ensemble_predictions = pred_df.mode(axis=1)
