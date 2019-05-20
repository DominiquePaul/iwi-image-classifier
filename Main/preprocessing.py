"""
A script that gathers important functions for processing and managing the data


Interesting links which were considered for implementation and might be relevant for other users:
    getting images from imagenet:
        https://medium.com/coinmonks/how-to-get-images-from-imagenet-with-python-in-google-colaboratory-aeef5c1c45e5
        https://colab.research.google.com/drive/1MALKxRqmNdjBUXJ-6V4PFYU6inPWq7Qe#scrollTo=vVPl9aGooPC9

    for data augmentation:
        https://github.com/aleju/imgaug
        https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
"""


import os
import cv2
import json
import time
import signal
import random
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import BytesIO
from bs4 import BeautifulSoup
from tensorflow.python.lib.io import file_io
from sklearn.model_selection import train_test_split


def load_images(folders):
    """
    loads all images from a folder and its subfolders

    loads images from a list of folders and returns a list with the names
    and a numpy array of the images


    Args:
        folders: python list containing file paths to folders (not images)

    Returns:
        file_name_list: python list with names of all files contained in folder
        image_array: np.array with files in pixel value format
    """

    file_name_list = []
    image_array = []

    for folder in folders:
        for subdir, dirs, files in os.walk(folder):
            for i,file in enumerate(tqdm(files)):
                file_path = os.path.join(subdir, file)
                image = load_single_image(file_path) # returns a numpy array
                if image is not None:
                    file_name_list.extend([file])
                    image_array += [image]
    image_array = np.array(image_array)
    file_name_list = pd.DataFrame(file_name_list, columns=["file_name"])

    return(file_name_list, image_array)

def load_single_image(path):
    """
    loads a single image from a local file and returns in resized

    corrupt images are skipped

    Args:
        path: string that is a path to a local file

    Returns:
        img: pixel value image, resized to a globally set value
    """
    if ".jpg" not in path:
        print("Not an image: {}".format(path))
    if is_corrupt(path):
        print("Skipping corrupt image: {}".format(path))
        return None
    else:
        try:
            img = cv2.imread(path)
            img = resize_image(img)
        except:
            return None
        return img

def is_corrupt(path):
    """
    Checks whether an image is damaged and cannot be loaded

    Sometimes images can be saved in a false format or can be damaged when
    transferred between different mediums. This function can be used whether
    this is the case to skip loading these images

    Args:
        path: string that is a path to a local file

    Returns:
        bool: boolean value indicating whether the file is can be read in or not

    """
    try:
        img = cv2.imread(path)
    except:
        return True

    if img is None:
        print("Skipping corrupt image {}".format(path))
        return True

    return False

def resize_image(img):
    """
    skews or pads an image depending on a global parameter

    whether the image is padded or skewed depends on the global paramater called
    'allow_skewing'

    Args:
        img: np.array containing pixel values of an image

    Returns:
        img: np.array in a new format with skewed or padded image
    """

    if allow_skewing:
        img = skew_image(img)
    else:
        img = pad_image(img)
    return img

def skew_image(img):
    """
    skews an image to a square size using inter cubic polation

    size depends on the global parameter resize_to_size

    Args:
        img: np.array with pixel values

    Returns:
        img: np.array with pixel values of a newly skewed image
    """
    img = cv2.resize(img,(resize_to_size, resize_to_size), interpolation = cv2.INTER_CUBIC)
    return img

def pad_image(img):
    """
    resizes and pads an image to a specified square size

    size depends on the global parameter resize_to_size

    Args:
        img: np.array containing pixel values of an image

    Returns:
        img: np.array with pixel values of a newly sized and padded image
    """

    size = img.shape[0:2 ]
    missing = np.max(size) - np.min(size)
    if size[0] > size[1]:
        img = cv2.copyMakeBorder(img, 0, 0, int(np.floor(missing/2)), int(np.ceil(missing/2)), cv2.BORDER_WRAP)
    elif size[0] < size[1]:
        img = cv2.copyMakeBorder(img, int(np.floor(missing/2)), int(np.ceil(missing/2)), 0,0, cv2.BORDER_WRAP)
    img = cv2.resize(img,(resize_to_size, resize_to_size), interpolation = cv2.INTER_CUBIC)
    return img

def load_json_file_paths(folder_path):
    """
    checks all files in a directory for json files

    Args:
        folder_path: string with a folder path in a local directory

    Returns:
        file_paths: python list with file path strings
    """
    folder_files = os.listdir(folder_path)
    file_paths = []
    for file in folder_files:
        if ".json" not in file:
            continue
        else:
            file_paths += [os.path.join(folder_path, file)]
    return file_paths

def read_label_json(folder, line_interpreter_fn):
    """
    returns df with file names and labels for all json files found in folder

    Args:
        folder: string that is a file folder in the local directory
        line_interpreter_fn:  a function that reads a single line in the json
            file. This is important as different json files with be formatted in
            different ways

    Returns:
        label_df: returns a dataframe with all file names in one column called
            "file_names" and the associated labels in another called "labels"
    """
    files = load_json_file_paths(folder)
    label_df = pd.DataFrame()
    for file in files:
        with open(file) as f:
            lines = list(f)
        for line in lines:
            file_name, label = line_interpreter_fn(line)
            df_temp = pd.DataFrame.from_dict({"file_name": [file_name], "label": [label]})
            label_df = label_df.append(df_temp, ignore_index=True)
    # drop duplicates
    label_df = label_df.drop_duplicates("file_name")
    return label_df

def analyse_line_dataturk_format(line):
    """
    scans a json file line and returns the file name and label

    the format is specifically for the one returned by the dataturks labelling
    platform

    Args:
        line: string read from json file

    Returns:
        file_name: sting with name of the file in the json
        label: string with the identified value of label identified
    """
    line_dict = json.loads(line)
    file_name = line_dict["content"]
    file_name = "___".join(file_name.split("___")[1:])
    label = None
    if "annotation" in line_dict.keys():
        if line_dict["annotation"] is not None:
            if "labels" in line_dict["annotation"].keys():
                labels = line_dict["annotation"]["labels"]
                if len(labels) > 1:
                    print("more than one label contained in file {}".format(file_name))
                    #raise ValueError("more than one label contained in file {}".format(file_name))
                elif len(labels) == 0:
                    print("No label contained in file {}".format(file_name))
                    #raise ValueError("No label contained in file {}".format(file_name))
                else:
                    label =  labels[0]
            elif "label" in line_dict["annotation"].keys():
                labels = line_dict["annotation"]["label"]
                if len(labels) > 1:
                    print("more than one label contained in file {}".format(file_name))
                    #raise ValueError("more than one label contained in file {}".format(file_name))
                elif len(labels) == 0:
                    print("No label contained in file {}".format(file_name))
                    #raise ValueError("No label contained in file {}".format(file_name))
                label = labels[0]

    return(file_name, label)

def return_labelled_images(labels, file_names, files):
    """
    aligns labels and files to return a labelled dataset

    if no label is found for an image, then it is omitted in the files returned
    to ensure that the dataset is 100% labelled. File names in file_names and
    files themselves have to be aligned

    Args:
        labels: pd.DataFrame containg a column called "file_name" and a column
            called "label"
        file_names: pd.DataFrame containg a column called "file_name", the order
            of the names has to correspond to the images in the files variable
        files: np.array with the pixel values of images

    Returns:
        file_dataframe: pd.DataFrame with the most
        files: np.array but with the files omitted that dont have a label
    """
    len1 = len(file_names)
    file_dataframe = file_names.reset_index().set_index("file_name").join(labels.set_index("file_name")).reset_index()
    file_dataframe = file_dataframe.loc[file_dataframe["label"].notna(),:]
    file_dataframe = file_dataframe.reset_index(drop=True)
    files = files[file_dataframe["index"]]
    print("{} out of {} files dropped".format(len1-len(file_dataframe), len1))
    return(file_dataframe, files)

def factorize_labels(labels):
    """
    Turns string labels into numeric labels

    Args:
        labels:

    Returns:
        labels: np.array with numeric label values
        original_labels: np.array which shows the translation of string labels
            to numeric labels and vice versa
    """
    labels, original_labels = pd.factorize(labels)
    return(labels, original_labels)

def break_up_array(array_to_split, nbuckets):
    """
    splits an array into equally sized smaller arrays

    Args:
        array_to_split: np.array that should be splitted into smaller arrays
        nbuckets: (int) the amount of buckets the array should be split into

    Returns:
        array_split: list of arrays
    """
    if  nbuckets == 1:
        return [array_to_split]
    size = len(array_to_split)
    step_size = np.round(size/nbuckets, 0)
    steps = [[int(step_size*i), int(step_size*(1+i))] for i in range(nbuckets-1)]
    steps = steps + [[np.max(steps),size]]
    array_split = [array_to_split[i:j] for i,j in steps]
    return array_split

def split_arrays_unequally(splits, arrays):
    """
    splits arrays in a list into fractions defined in splits variable

    An example would be the split [0.5, 0.3, 0.2] on the array [1,2,3,4,5,6,
    7,8,9,10] resulting in [[1,2,3,4,5],[6,7,8],[9,10]]. This is very useful,
    when you want to split several arrays into the same unequally sized splits

    Args:
        splits: list of decimals that add up to 1 and indicate the desired split
        arrays: list of the arrays that are to be split

    Returns:
        arrays_new: array containing newly split arrays
    """
    assert sum(splits) <= 1
    if isinstance(arrays, np.ndarray) is False:
        arrays = np.array(arrays)[:-1]
    if isinstance(splits, np.ndarray) is False:
        splits = np.array(splits)
    total_length = len(arrays[0])
    p = np.random.permutation(total_length)
    splits=splits*total_length
    splits = np.floor(splits).cumsum().astype(int)
    splits[-1]=total_length
    splits = np.insert(splits, 0, 0)

    list_of_arrays = []
    for array in arrays:
        if len(array) == 0:
            list_of_arrays.extend([None]*(len(splits)-1))
        else:
            array = array[p]
            array_list = []
            for i in range(len(splits)-1):
                array_list.extend([array[splits[i]:splits[i+1]]])
            list_of_arrays.extend([array_list])
    arrays_new = np.array(list_of_arrays)
    return arrays_new

def balance_data(x_data, y_data):
    """
    rebalances an unequal dataset (in terms of available data)

    replaces the underrepresented sample to force equal amount of samples. So
    far it only works when two labels are given. So far the balance_data option
    is the only option that does not work with more than two classes (non-binary
    classification)

    Args:
        x_data: np.array with image values
        y_data: np.array with two label tags

    Returns:
        x_new: np.array with newly arranged images
        y_new: np.array with newly arranged labels
    """
    a,b = np.bincount(y_data)
    diff = np.abs(a-b)
    if a > b:
        undersampled = x_data[np.where(y_data==1,True,False)]
        label_num = 1
    else:
        undersampled = x_data[np.where(y_data==0,True,False)]
        label_num = 0

    new_images = []
    max_idx = len(undersampled)
    for i in range(diff):
        idx = np.random.randint(0, max_idx)
        new_images.extend([undersampled[idx]])
    x2 = np.array(new_images)
    y2 = np.array([label_num]*diff)
    x_new = np.concatenate([x_data, x2])
    y_new = np.concatenate([y_data, y2])

    random_order = np.random.permutation(len(x_new))
    x_new = x_new[random_order]
    y_new = y_new[random_order]

    return(x_new, y_new)

def save_to_numpy(folder_path, files, img_names, object):
    """
    Saves a file as one or several numpy pickle format depending on file size

    the function whether the file surpasses a maximum file size. If so, then it
    splits the data up into several file packages. It prints progress when
    saving but doesnt return any value

    Args:
        folder_path: sting of the path on the local file system to which the
            files should be saved to
        files: the np.array of files to be saved
        img_names: the names of the images being saved, they are added to the
            files being saved so that they can be reloaded later to create an
            overview of predictions
        object: (string) name of the object being saved. It is used for the name
            of the saved file
    """
    approx_file_size = files[0].nbytes * len(files)
    gb = approx_file_size / 1.08e+9 # approximate conversion rate of byte to GB
    splits = int(np.ceil(gb)) # two gb is the max size for saving a np array to disk on mac as of python y 3.6.8, so we choose a convenient cap of approx one gb
    print("splits needed: {}".format(splits))

    files = break_up_array(files, splits)
    file_names = break_up_array([img_names], splits)

    for i in range(len(files)):
        file_bundle_out = np.array([files[i], file_names[i]])
        path = os.path.join(folder_path, object + "_image_package_no_labels_" + str(i))
        np.save(path, file_bundle_out)
        print("File package {} out of {} saved successfully".format(i+1, len(files)), end="\r")

def save_to_numpy_with_labels(folder_path, files, labels, object, balance_dataset, augment_training_data, train_val_test_split, file_names=None):
    """
    saves a bundle of files as a numpy pickle with some smart functionalities

    the function takes files and labels, optionally splits them into a train,
    validation, and test split. If selected, the function also augments the
    training dataset in various ways before saving it. Optionally, file_names
    are saved together with the files and file labels as well. the function also
    automatically checks whether the total file size is to big to be saved to a
    single file and if necessary splits the files into several chunks and saves
    them individually.
    The test data and labels are saved separately
    The function doesnt return any values but prints updates on the saving
    progress

    Args:
        folder_path: (string) the local file path to which the files should be
            saved
        files: (np.array) the files to be saved
        labels: (np.array) the labels of the files to be saved
        object: (string) the name of the object depiced in the image, this is
            relevant for the file name
        balance_dataset: (boolean) if True then the underreprsented image
            classes are resampled until the balance between the different images
            is equal
        augment_training_data: (boolean) if True the images are augmented in
            various ways
        train_val_test_split: (boolean) if True the all the data is split into a
            training, validation and test split. The test values are saved under
            a different file name.
        file_names: (np.array) the list of file names associated with the images
    """
    labels, label_index = factorize_labels(labels)
    if file_names is None:
        file_names = np.array([None]*len(files))
    if train_val_test_split:
        [x,y,names] = split_arrays_unequally([0.7,0.2,0.1], [files,labels,file_names,label_index])
        [x_train, x_val, x_test] = x
        [y_train, y_val, y_test] = y
        [names_train, names_val, names_test] = names
    else:
        x_train=files
        y_train=labels
        names_train=file_names

    if balance_dataset:
        x_train, y_train = balance_data(x_train, y_train)

    if augment_training_data:
        x_train, y_train = augment_data(x_train, y_train, shuffle=True)

    # next steps are for saving only
    # checking whether the file size is larger than 4 gigabytes (*1.3 for margin of error)
    if train_val_test_split:
        approx_file_size = (x_train[0].nbytes + y_train[0].nbytes) * (len(x_train) + len(x_val)) * 1.3
    else:
        approx_file_size = (x_train[0].nbytes + y_train[0].nbytes) * len(x_train) * 1.3
    gb = approx_file_size / 1.08e+9 # approximate conversion rate of byte to GB
    splits = int(np.ceil(gb)) # two gb is the max size for saving a np array to disk on mac as of python y 3.6.8, so we choose a convenient cap of approx one gb
    print("splits needed: {}".format(splits))

    x_train_splits = break_up_array(x_train, splits)

    y_train_splits = break_up_array(y_train, splits)
    names_train_splits = break_up_array(names_train, splits)
    if train_val_test_split:
        x_val_splits = break_up_array(x_val, splits)
        y_val_splits = break_up_array(y_val, splits)
        names_val_splits = break_up_array(names_val, splits)

    if train_val_test_split:
        path_testing_set = os.path.join(folder_path, object + "_final_testing_dataset")
        final_testing_set = np.array([x_test, y_test, names_test, label_index])
        np.save(path_testing_set, final_testing_set)

    augmented_str = "augmented_" if augment_training_data else ""

    for i in range(len(x_train_splits)):
        if train_val_test_split:
            file_bundle_out = np.array([x_train_splits[i], y_train_splits[i], x_val_splits[i], y_val_splits[i], label_index])
            path = os.path.join(folder_path, object + "_image_package_train_val_split_" + augmented_str + str(i))
        else:
            if labels is not None:
                file_bundle_out = np.array([x_train_splits[i], y_train_splits[i], label_index])
            else:
                file_bundle_out = np.array([x_train_splits[i], label_index])
            path = os.path.join(folder_path, object + "_image_package_" + augmented_str + str(i))
        np.save(path, file_bundle_out)
        print("File package {} out of {} saved successfully".format(i+1, len(x_train_splits)), end="\r")

def translate_image(img, right, down):
    """
    A function that moves an image along the x,y direction

    The parts of the image pushed "off" the image are added to the part of the
        image that has received free space

    Args:
        img: the image file to apply the translation to (numpy array)
        right: number of pixels to shift image to the right (negative value shifts left)
        down: number of pixels to shift image down (negative value shifts up)

    Returns:
        img: translated image (numpy array)
    """
    rows,cols = img.shape[:2]
    m = np.float32([[1,0,right],[0,1,down]])
    img = cv2.warpAffine(img, m, (cols,rows), borderMode=cv2.BORDER_WRAP)
    return(img)

def change_brightness(img, value=30, mode="lighten"):
    """
    Changes the brightness of an image

    Assumes images are saved in BGR format with values between 0 and 255. If a
        pixel value would be bigger (or smaller) than 255 (0) then it is clamped
        at a value of 255 (0)

    Args:
        img: image to be used (numpy array)
        value: amount to shift change the brightness by
        mode: set to 'lighten' or 'darken'
    Returns:
        img: new image (numpy array)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if mode == "lighten":
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    elif mode == "darken":
        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def augment_single_image(img, label):
    """
    Applies numerous changes to an image to expand a dataset

    We 'augment' an image with five different methods and thereby produce 11
    version of our original image (counting the original version). The methods
    applied are: (A) horizontal flipping (B) blurring (C) Translation (4x) (D)
    change of brightness (2x) (E) conversion to black and white.

    Args:
        img: (np.array) pixel values of an image
        label: (int or string) the label of the associated image

    Returns:
        files: (np.array) of the pixel values of the new files
        labels: (np.array) array of the old label adjusted to the new amount of
            images
    """
    flipped = cv2.flip(img, 1)
    blurred = cv2.GaussianBlur(img,(3,3),0)
    median_filtering = cv2.medianBlur(img,3)
    translated1 = translate_image(img, 200, 0)
    translated2 = translate_image(img, -200, 0)
    translated3 = translate_image(img, 0, 200)
    translated4 = translate_image(img, 0, -200)
    lightened_image = change_brightness(img, value=75, mode="lighten")
    darkened_image = change_brightness(img, value=75, mode="darken")
    black_and_white = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    files = np.array([img, flipped, blurred, median_filtering, translated1,
                    translated2, translated3, translated4, lightened_image,
                    darkened_image, black_and_white,])
    labels = np.array([label]*len(files))

    return(files, labels)

def augment_data(image_files, labels, shuffle=True):
    """
    Takes an array of images and labels and applies augmentations

    The function applies a set of image augmentation functions to extend the
    size of an image set which in turn allows better training of algorithms,
    especially when the number of training data is quite low.

    Args:
        image_files: an numpy array containing images
        labels: the corresponding labels to the images
        shuffle: whether or not to shuffle the final image set (could be useful
            for later training as the algorithm will otherwise train on a similar
            image several times in a row)

    Returns:
        images: a numpy file of the image
        labels: a numpy array with the corresponding labels
    """
    images_v2 = []
    labels_v2 = []
    for i,l in tqdm(zip(image_files, labels)):
        imgs, labs = augment_single_image(i,l)
        images_v2.extend(imgs)
        labels_v2.extend(labs)
    images_v2 = np.array(images_v2)
    labels_v2 = np.array(labels_v2)

    if shuffle:
         p = np.random.permutation(len(labels_v2))
         images_v2 = images_v2[p]
         labels_v2 = labels_v2[p]

    return(images_v2, labels_v2)

def get_children_synset_ids(synset_id):
    """
    Takes a synset id from wordnet and returns all children synsets

    For more information on ImageNet, Synset, and the ImageNet API have a look
    at http://image-net.org/download-API

    Args:
        synset_id: (string) a ID for an ImageNet synset

    Returns:
        synsets: list of synset IDs
    """
    b = requests.get("http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=" + synset_id)
    synsets = b.text.split("\r\n-")
    synsets[-1] = synsets[-1][:-2]
    return synsets

def get_synset_image_urls(synset_id):
    """
    Takes a synset id and fetches urls of associated images

    The function uses the synset ID to open a website from ImageNet
    corresponding to the synset ID that displays nothing but a long list of
    links of images associated with the object of the synset

    Args:
        synset_id: (string) a ID for an ImageNet synset

    Returns:
        website_lines: a list of URLs to images
    """
    website = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=" + synset_id)
    website_content = BeautifulSoup(website.content, 'html.parser')
    website_content_string = str(website_content)
    website_lines = website_content_string.split('\r\n')
    return (website_lines)

def timeout_handler(num, stack):
    """
    A function that does nothing else to raise an exception. It is needed for
    another function that requires a function to be passed in case of a timeout.
    See the function 'download_image' for more information
    """
    raise Exception()

def download_image(url):
    """
    Downloads and resizes an image from a URL and aborts after 5 seconds waiting

    This function is used to load images from a ImageNet URL list. As the list
    is quite old, a number of the links are dead. If we try to load these links
    it takes quite a bit of time until python notices, that the links are dead
    and aborts the process. Therefore we use the signals module that auto-
    matically aborty the process if it takes too long. This significantly
    improves the waiting time of the function.

    NOTE:
    A part of the code was adapted from the following question on stackoverflow:
    https://stackoverflow.com/questions/54160208/how-to-use-opencv-in-python3-to-read-file-from-file-buffer/54162776#54162776
    A different buffer function was used as the older version was causing a
    deprecation warning

    Args:
        url: the url from which the image should be loaded

    Returns:
        img: (np.array) pixel value of the image loaded. If no image is found at
            the URL, then a None value is returned
    """
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)

        response = requests.get(url)
        img_pre = BytesIO(response.content)
        img = cv2.imdecode(np.frombuffer(img_pre.read(), np.uint8), 1)
        img = resize_image(img)
    except:
        return None
    finally:
        signal.alarm(0)
    return img

def create_imagenet_dataset(synset_id, size, use_child_synsets):
    """
    Gathers images from the internet associated with a ImageNet synset ID

    Args:
        synset_id: (string) a ID for an ImageNet synset
        size: the amount of images to collect
        use_child_synsets: (boolean) whether or not to use the children synsets
            of the synset (this is generally advised when collecting larger
            datasets as there might not be enough images assigned to single
            synset and the children synsets also contain versions of the image
            (e.g. beach waggon would be a child for car)
    Returns:
        images: np.array with downloaded images
    """
    if use_child_synsets:
        synset_ids = [synset_id] + get_children_synset_ids(synset_id)
    else:
        synset_ids = [synset_id]
    image_urls = []
    for id in synset_ids:
        urls = get_synset_image_urls(id)
        image_urls.extend(urls)
    random.shuffle(image_urls)
    image_list = []
    counter = 0
    print("ready for search")
    for i, url in enumerate(image_urls):
        img = download_image(url)
        if img is not None:
            counter += 1
            image_list.extend([img])
            if counter == size:
                break
        print("URLs tried: {} --- images_found: {}".format(i, counter), end="\r")
    images = np.array(image_list)
    return images

def create_imagenet_dataset_random(size, max_synset_imgs, forbidden_synset, exclude_synset_children):
    """
    Gathers random images from the internet with the help of ImageNet lists

    The method chooses a random synset from all synset IDs existent and starts
    downloading images from this list until a parameter defined threshold is
    reached. Then, the function continues downloading images from another
    synset. The user can specify whether to exclude a specific synset and its
    children

    Args:
        size: (int) the amount of images to collect
        max_synset_imgs: (int) the maximal number of images taken from one
            synset. A higher number will increase speed, but will result in a
            less diverse synset and vice versa. Generally a lower number is
            recommended
        forbidden_synset: (string) a Synset ID which should definitely be
            excluded when gathering the images
        exclude_synset_children: (boolean) if True then all children synsets of
            the forbidden synsets are excluded as well

    Returns:
        images: np.array of images gathered
    """
    if exclude_synset_children:
        forbidden_synsets = [forbidden_synset] + get_children_synset_ids(forbidden_synset)
    else:
        forbidden_synsets = [forbidden_synset]
    website = requests.get("http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list")
    synset_ids = website.text.split("\n")
    synset_ids = synset_ids[:-3] # the last three lines are empty

    random.shuffle(synset_ids)
    image_list = []
    counter = 0
    print("starting new url list")
    for i, synset_id in enumerate(synset_ids):
        if synset_id not in forbidden_synsets:
            urls = get_synset_image_urls(synset_id)
            for j, url in enumerate(urls):
                img = download_image(url)
                print("Synsets searched: {} --- URLs tried: {} --- images_found: {}".format(i, i*max_synset_imgs+j, counter), end="\r")
                if img is not None:
                    counter += 1
                    image_list.extend([img])
                    if j >= max_synset_imgs or counter >= size:
                        break
            if counter >= size:
                break
    images = np.array(image_list)
    return images

def load_from_gcp(url):
    """
    loads a file from a GCP storage bucket location (if being run in GCP code)

    This function can be used when running code on google cloud (e.g. compute
    engine). It could just as well be used, when running from a local
    environment, you would only have to add the credentials to access the google
    cloud storage bucket. As changes over time can cause unforeseen issues with
    such a method, it is recommended to use this function when running a script
    in the cloud

    Args:
        url: (string) the link to the storage bucket from which the data should
        be loaded

    Returns:
        data: the raw data loaded from the URL and which has to be processed
            further
    """

    f = BytesIO(file_io.read_file_to_string(url, binary_mode=True))
    data = np.load(f)
    return data

def join_npy_data(file_path_list, training_data_only):
    """
    loads data that was saved using the save_to_numpy_with_labels function

    This function is not necessary to load the data which was previously saved
    using the save_to_numpy_with_labels function, but can save a significant
    amount of time.
    As the save_to_numpy_with_labels sometimes splits the data due to size con-
    straints, this function automatically joins the data back together if a list
    of several file paths is passed
    The last file in the list returned is the file indicating how the original
    labels were translated into numbers

    Args:
        file_path_list: (list of strings) a list of file_paths from which to
            load and merge the data
        training_data_only: (boolean) if True, then the function expects to
            unpack only two three parameters, else it will expect five, as
            validation files would also be included

    Returns:
        data_out: (list of np.arrays) a list of either three or five values
            corresponding to the files extracted. In the case of three files
            this is the training data and labels plus the label translation file
            (mentioned above). In case of five values it is the training and
            validation data plus the label translation file (mentioned above)
    """
    x_train_list = []
    x_val_list = []
    y_train_list = []
    y_val_list = []
    for element in tqdm(file_path_list):
        if "gs://" in element:
            data_package = load_from_gcp(element)
        else:
            data_package = np.load(element)
        if training_data_only is True:
            x_train, y_train, conversion = data_package
        else:
            x_train, y_train, x_val, y_val, conversion = data_package
            y_train_list.extend(y_train)
            y_val_list.extend(y_val)
        x_train_list.extend(x_train)
        x_val_list.extend(x_val)

    if training_data_only is True:
        data_out = [np.array(x_train_list), np.array(y_train_list), conversion]
    else:
        data_out = [np.array(x_train_list), np.array(y_train_list), np.array(x_val_list), np.array(y_val_list), conversion]
    return data_out


GLOBAL_RANDOM_STATE = 42
resize_to_size = 299 # the length and height an image is resized to, if set to None no resizing happens
allow_skewing = False # if set to True then the image is skewed, if False, the image is padded

if __name__ == "__main__":

    np.random.seed(GLOBAL_RANDOM_STATE)


    ########################
    ####### for cars #######
    ########################
    #
    # # all folders to search for image files
    # car_image_folders = ["../Data/car/car",
    #                     "../Data/car/no_car"]
    # # the folder in which to search for json files to be loaded
    # car_json_folder = "../Data/json_files/cars"
    #
    # # loading the images and thei rnames from the designated folders
    # car_names_raw, car_images = load_images(folders=car_image_folders)
    # # reading in the image labels with a specified parsing function
    # car_labels = read_label_json(folder=car_json_folder,
    #                              line_interpreter_fn=analyse_line_dataturk_format)
    # # merging the labels and images together
    # car_names, car_files = return_labelled_images(labels=car_labels,
    #                                               file_names=car_names_raw,
    #                                               files=car_images)
    #
    # target_output_folder = "../Data/np_files_final"
    #
    # save_to_numpy_with_labels(folder_path=target_output_folder,
    #                           files=car_files,
    #                           labels=car_names["label"],
    #                           object="car",
    #                           balance_dataset=True,
    #                           augment_training_data=False,
    #                           train_val_test_split=True,
    #                           file_names=car_names["file_name"])
    #
    # save_to_numpy_with_labels(folder_path=target_output_folder,
    #                           files=car_files,
    #                           labels=car_names["label"],
    #                           object="car",
    #                           balance_dataset=True,
    #                           augment_training_data=True,
    #                           train_val_test_split=True,
    #                           file_names=car_names["file_name"])
    #
    #
    # save_to_numpy(folder_path=target_np_folder,
    #               img_names=car_names_raw["file_name"],
    #               files=car_images,
    #               object="testing_data")


    ########################
    ####### for food #######
    ########################
    #
    # # all folders to search for image files
    # food_image_folders = ["../Data/food/food",
    #                     "../Data/food/no_food"]
    # # the folder in which to search for json files to be loaded
    # food_json_folder = "../Data/json_files/food"
    #
    # # loading the images and thei rnames from the designated folders
    # food_names_raw, food_images = load_images(folders=food_image_folders)
    # # reading in the image labels with a specified parsing function
    # food_labels = read_label_json(folder=food_json_folder,
    #                              line_interpreter_fn=analyse_line_dataturk_format)
    # # merging the labels and images together
    # food_names, food_files = return_labelled_images(labels=food_labels,
    #                                               file_names=food_names_raw,
    #                                               files=food_images)
    #
    # target_output_folder = "../Data/np_files_final"
    #
    #
    # save_to_numpy_with_labels(folder_path=target_output_folder,
    #                           files=food_files,
    #                           labels=food_names["label"],
    #                           object="food",
    #                           balance_dataset=True,
    #                           augment_training_data=False,
    #                           train_val_test_split=True,
    #                           file_names=food_names["file_name"])
    #
    # save_to_numpy_with_labels(folder_path=target_output_folder,
    #                           files=food_files,
    #                           labels=food_names["label"],
    #                           object="food",
    #                           balance_dataset=True,
    #                           augment_training_data=True,
    #                           train_val_test_split=True,
    #                           file_names=food_names["file_name"])
    #
    #
    # save_to_numpy(folder_path=target_output_folder,
    #               img_names=food_names_raw["file_name"],
    #               files=food_images,
    #               object="testing_data")



    ###########################
    ####### for apparel #######
    ###########################

    # all folders to search for image files
    apparel_image_folders = ["../Data/apparel/apparel",
                        "../Data/apparel/no_apparel"]
    # the folder in which to search for json files to be loaded
    apparel_json_folder = "../Data/json_files/apparel"

    # loading the images and thei rnames from the designated folders
    apparel_names_raw, apparel_images = load_images(folders=apparel_image_folders)
    # reading in the image labels with a specified parsing function
    apparel_labels = read_label_json(folder=apparel_json_folder,
                                 line_interpreter_fn=analyse_line_dataturk_format)
    # merging the labels and images together
    apparel_names, apparel_files = return_labelled_images(labels=apparel_labels,
                                                  file_names=apparel_names_raw,
                                                  files=apparel_images)

    target_output_folder = "../Data/np_files_final"


    save_to_numpy_with_labels(folder_path=target_output_folder,
                              files=apparel_files,
                              labels=apparel_names["label"],
                              object="apparel",
                              balance_dataset=True,
                              augment_training_data=False,
                              train_val_test_split=True,
                              file_names=apparel_names["file_name"])

    save_to_numpy_with_labels(folder_path=target_output_folder,
                              files=apparel_files,
                              labels=apparel_names["label"],
                              object="apparel",
                              balance_dataset=True,
                              augment_training_data=True,
                              train_val_test_split=True,
                              file_names=apparel_names["file_name"])


    save_to_numpy(folder_path=target_output_folder,
                  img_names=apparel_names_raw["file_name"],
                  files=apparel_images,
                  object="testing_data")











"""

from PIL import Image
img = load_single_image("/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/apparel/apparel/RalphLauren_6797246895_10153439142626896_.jpg")

files, labs = augment_single_image(img, 1)
img2 = Image.fromarray(translated, 'RGB')
img2.show()

blurring:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
edges: https://github.com/aleju/imgaug/issues/79

"""
