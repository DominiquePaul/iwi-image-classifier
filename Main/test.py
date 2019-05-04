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
from tqdm import tqdm


from io import BytesIO
from tensorflow.python.lib.io import file_io
from tpu_v4 import cnn_model


MAX_EVALS = 20

def join_npy_data(list1, gcp_source=False):

    for element in tqdm(list1):
        if gcp_source:
            f = BytesIO(file_io.read_file_to_string(element, binary_mode=True))
            a = np.load(f)
    return a

data_url='gs://data-imr-unisg/np_array_files/car_image_package_train_val_split0.npy'
join_npy_data([data_url], gcp_source=True)
