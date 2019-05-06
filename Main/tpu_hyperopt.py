"""
Adjustments Missing:

If I want the scripts to work together as a whole I should write the Hyper Optimization as
    a function of some sorts so it can be called from another script. This means rewriting
    variables as a function parameters
"""

import csv
import numpy as np
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.stochastic import sample
from timeit import default_timer as timer

from tpu import cnn_model
from preprocessing import join_npy_data

MAX_EVALS = 20

data_url=['gs://data-imr-unisg/np_array_files/car_image_package_train_val_split_0.npy']
x_train, y_train, x_test, y_test , conversion = join_npy_data(data_url, training_data_only=False, gcp_source=True)

# data =["/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_image_package_train_test_split0.npy"]
# x_train, y_train, x_test, y_test , conversion = join_npy_data(data, gcp_source=False)


# File to save first results
out_file = 'out_files/custom_nn_hyperopt.csv'
with open(out_file, 'w') as csv_file:
    writer = csv.writer(csv_file)
    # Write the headers to the file
    writer.writerow(['conv_layers', 'conv_filters', 'dense_layers', 'dense_neurons',
                    'dropout_rate_dense', 'learning_rate', 'run_time', 'val_loss',
                    'val_accuracy','val_f1', 'train_loss', 'train_accuracy', 'train_f1'])

# hyperparameter optimization with hyperopt
def objective(params):
    m_opt=cnn_model()
    m_opt.new_model(x_train, y_train, 2, params)
    start = timer()
    # add a special logs directory to see what is happening during each iteration
    m_opt.train(on_tpu=True, epochs=1000, batch_size=256, tb_logs_dir="gs://data-imr-unisg/logs_hyperopt/")
    run_time = timer() - start
    val_loss = m_opt.hist.history["val_loss"][-1]
    val_accuracy = m_opt.hist.history["val_sparse_categorical_accuracy"][-1]
    val_f1 = m_opt.hist.history["val_f1_score"][-1]
    train_loss = m_opt.hist.history["loss"][-1]
    train_accuracy = m_opt.hist.history["sparse_categorical_accuracy"][-1]
    train_f1 = m_opt.hist.history["f1_score"][-1]

    output_vals = [params["conv_layers"], params["conv_filters"], params["dense_layers"], params["dense_neurons"],
                params["dropout_rate_dense"], params["learning_rate"],
                run_time, val_loss, val_accuracy, val_f1, train_loss, train_accuracy, train_f1]

    # adding lines to csv
    with open(out_file, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(output_vals)

    return {"loss": val_loss,
            "params": params,
            "status": hyperopt.STATUS_OK}

# Define the search space
space = {
    "conv_layers": hp.quniform("conv_layers", 4, 8, 1),
    "conv_filters": hp.quniform("conv_filters", 2, 128, 1),
    "dense_layers": hp.quniform("dense_layers", 1, 5, 1),
    "dense_neurons": hp.quniform("dense_neurons", 2, 100, 1),
    "dropout_rate_dense": hp.uniform("dropout_rate_dense",0,0.9),
    "learning_rate": hp.loguniform('learning_rate', np.log(1e-02), np.log(1e-06)),
}

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = Trials())

# write best parameters as to disk
with open('./out_files/best_custom_nn_parameters.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in best.items():
       writer.writerow([key, value])

print("Finished hyperopt. Best parameters are:")
print(best)
