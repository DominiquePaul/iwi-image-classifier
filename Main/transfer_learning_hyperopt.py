import numpy as np
import csv
import hyperopt
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.stochastic import sample
from timeit import default_timer as timer

from transfer_learning import Transfer_net
from preprocessing import join_npy_data

EPOCHS = 10000
MAX_EVALS = 20

# File to save first results
out_file = 'out_files/transfer_learning_hyperopt_out.csv'
with open(out_file, 'w') as csv_file:
    writer = csv.writer(csv_file)
    # Write the headers to the file
    writer.writerow(['neurons','layers','dropout_rate','learning_rate','run_time', 'val_loss', 'val_accuracy','val_f1', 'train_loss', 'train_accuracy', 'train_f1'])

automotive_pckgs = ["/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/car_image_package_train_test_split0.npy"]
x_data, y_train, _, _, conversion = join_npy_data(automotive_pckgs)

# hyperparameter optimization with hyperopt
def objective(params):
    t_net = Transfer_net("/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/transfernet_files", 2)
    t_net.create_network(layers=params["layers"], neurons=params["neurons"], dropout_rate=params["dropout_rate"])
    x_train = t_net.cache_transfer_data(x_data, img_group_name="x_train_T6")
    start = timer()
    t_net.train(x_train, y_train, learning_rate=params["learning_rate"], epochs=EPOCHS, batch_size=256, verbose=True, tb_logs_dir="./log_files/transfer_net/")
    run_time = timer() - start

    val_loss = t_net.hist.history["val_loss"][-1]
    val_accuracy = t_net.hist.history["acc"][-1]
    val_f1 = t_net.hist.history["val_f1_score"][-1]
    train_loss = t_net.hist.history["loss"][-1]
    train_accuracy = t_net.hist.history["acc"][-1]
    train_f1 = t_net.hist.history["f1_score"][-1]

    # adding lines to csv
    with open(out_file, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([params['neurons'],params['layers'], params['dropout_rate'], params['learning_rate'],
                        run_time, val_loss, val_accuracy, val_f1, train_loss, train_accuracy, train_f1])

    return {"loss": val_loss,
            "params": params,
            "status": hyperopt.STATUS_OK}

# Define the search space
space = {
    "neurons": hp.quniform("neurons",1,100,1),
    "layers": hp.quniform("layers",1,50,1),
    "dropout_rate": hp.uniform("dropout_rate",0,1),
    "learning_rate": hp.loguniform('learning_rate', np.log(1e-02), np.log(1e-07))
}

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = Trials())

# write best parameters as to disk
with open('./out_files/best_transfer_learning_parameters_v2.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in best.items():
       writer.writerow([key, value])

print("Finished hyperopt and saved all results to {}. Best parameters are:".format(out_file))
print(best)
