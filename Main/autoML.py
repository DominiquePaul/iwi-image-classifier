import sys
from os.path import dirname
import pandas as pd
import numpy as np
import preprocessing as prep

# for cars
car_image_folders = ["/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/car/car",
                    "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/car/no_car"]
car_json_folder = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/json_files/cars"
car_names_raw, car_files = prep.load_images(car_image_folders)
car_labels = prep.read_label_json(car_json_folder)
car_names, car_files = prep.return_labelled_images(car_labels, car_names_raw, car_files)


csv_out = car_names.copy()
csv_out["file_name"] = "gs://imr-unisg-vcm/car_images/" + csv_out["file_name"]
csv_out = csv_out[["file_name", "label"]].copy()
csv_out["label"] = pd.factorize(csv_out["label"])[0]

csv_out.to_csv("/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/auto_ml/car_overview_sheet.csv", index=False, header=False)





























































#
