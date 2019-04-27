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
from sklearn.model_selection import train_test_split

"""
Further to do ideas:
- show_image_function
- load images from disc without labels (do it with an example)
- label images by folder from which they are loaded


getting images from imagenet:
    https://medium.com/coinmonks/how-to-get-images-from-imagenet-with-python-in-google-colaboratory-aeef5c1c45e5
    https://colab.research.google.com/drive/1MALKxRqmNdjBUXJ-6V4PFYU6inPWq7Qe#scrollTo=vVPl9aGooPC9

for data augmentation:
    https://github.com/aleju/imgaug
    https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
"""


resize_to_size = 299 # if set to None no resizing happens
allow_skewing = False # if false then image is padded

def load_images(folders):
    """
    loads images from a list of folders and returns a list with the names
    and a numpy array of the images
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
    """Takes the path of an image and returns it at a fixed square size
    """
    if ".jpg" not in path:
        print("Not an image: {}".format(path))
    if is_corrupt(path):
        print("Skipping corrupt image: {}".format(path))
        return None
    else:
        try:
            img = cv2.imread(path)
            if img is None:
                print("Skipping corrupt image {}".format(path))
            else:
                img = resize_image(img)
        except:
            print(path)
            return None
        return img

def is_corrupt(path):
    """Checks whether an image is damaged and cannot be loaded"""
    try:
        cv2.imread(path)
    except:
        return True
    return False

def resize_image(img):
    if allow_skewing:
        img = skew_image(img)
    else:
        img = pad_image(img)
    return img

def skew_image(img):
    img = cv2.resize(img,(resize_to_size, resize_to_size), interpolation = cv2.INTER_CUBIC)
    return img

def pad_image(img):
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
    returns a list of json files from a folder
    """
    folder_files = os.listdir(folder_path)
    file_paths = []
    for file in folder_files:
        if ".json" not in file:
            continue
        else:
            file_paths += [os.path.join(folder_path, file)]
    return file_paths

def read_label_json(folder):
    files = load_json_file_paths(folder)
    label_df = pd.DataFrame()
    for file in files:
        with open(file) as f:
            lines = list(f)
        for line in lines:
            line_dict = json.loads(line)
            file_name = line_dict["content"]
            file_name = "___".join(file_name.split("___")[1:])
            # file_name = "_".join(file_name.split("_")[4:])
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
            else:
                labels = None
            df_temp = pd.DataFrame.from_dict({"file_name": [file_name], "label": [label]})
            label_df = label_df.append(df_temp, ignore_index=True)
    # drop duplicates
    label_df = label_df.drop_duplicates("file_name")
    return label_df

def return_labelled_images(labels, file_names, files):
    """function that takes a names df, file_array, and label_df to return a reduced version of labelled items

    labels and file_names have to contain the column 'file_name' and file_dataframe has to contain the column 'label'
    """
    len1 = len(file_names)
    file_dataframe = file_names.reset_index().set_index("file_name").join(labels.set_index("file_name")).reset_index()
    file_dataframe = file_dataframe.loc[file_dataframe["label"].notna(),:]
    file_dataframe = file_dataframe.reset_index(drop=True)
    files = files[file_dataframe["index"]]
    print("{} out of {} files dropped".format(len1-len(file_dataframe), len1))
    return(file_dataframe, files)

def factorize_labels(labels):
    """A function that turns string labels into numeric labels

    file_name_labels must contain a column called "label"
    """
    labels, original_labels = pd.factorize(labels)
    return(labels, original_labels)

def split_array(array1, nbuckets):
    if  nbuckets == 1:
        return [array1]
    size = len(array1)
    step_size = np.round(size/nbuckets, 0)
    steps = [[int(step_size*i), int(step_size*(1+i))] for i in range(nbuckets-1)]
    steps = steps + [[np.max(steps),size]]
    ranges = [array1[i:j] for i,j in steps]
    return ranges

def save_to_numpy(folder_path, files, img_names, object):
    """Saves a file as numpy pickle and checks whether max saving size is surpassed
    """
    approx_file_size = files[0].nbytes * len(files)
    gb = approx_file_size / 1.08e+9 # approximate conversion rate of byte to GB
    splits = int(np.ceil(gb)) # two gb is the max size for saving a np array to disk on mac as of python y 3.6.8, so we choose a convenient cap of approx one gb
    print("splits needed: {}".format(splits))

    files = split_array(files, splits)
    file_names = split_array([img_names], splits)

    for i in range(len(files)):
        file_bundle_out = np.array([files[i], file_names[i]])
        path = os.path.join(folder_path, object + "_image_package_no_labels_" + str(i))
        np.save(path, file_bundle_out)
        print("File package {} out of {} saved successfully".format(i+1, len(files)), end="\r")

def save_to_numpy_with_labels(folder_path, files, labels, object, augment_training_data, split_into_train_test):
    labels, label_index = factorize_labels(labels)
    if split_into_train_test:
        x_train, x_test, y_train, y_test = train_test_split(files, labels, test_size=0.2)
    else:
        x_train=files
        y_train=labels

    if augment_training_data:
        x_train, y_train = augment_data(x_train, y_train, shuffle=True)

    # checking whether the file size is larger than 4 gigabytes (*1.2 for margin of error)
    if split_into_train_test:
        approx_file_size = (x_train[0].nbytes + y_train[0].nbytes) * (len(x_train) + len(x_test)) * 1.3
    else:
        approx_file_size = (x_train[0].nbytes + y_train[0].nbytes) * len(x_train)
    gb = approx_file_size / 1.08e+9 # approximate conversion rate of byte to GB
    splits = int(np.ceil(gb)) # two gb is the max size for saving a np array to disk on mac as of python y 3.6.8, so we choose a convenient cap of approx one gb
    print("splits needed: {}".format(splits))

    x_train = split_array(x_train, splits)
    y_train = split_array(y_train, splits)
    if split_into_train_test:
        x_test = split_array(x_test, splits)
        y_test = split_array(y_test, splits)

    for i in range(len(x_train)):
        if split_into_train_test:
            file_bundle_out = np.array([x_train[i], y_train[i], x_test[i], y_test[i], label_index])
        else:
            file_bundle_out = np.array([x_train[i], y_train[i], label_index])
        if split_into_train_test:
            path = os.path.join(folder_path, object + "_image_package_train_test_split" + str(i))
        else:
            path = os.path.join(folder_path, object + "_image_package_" + str(i))
        np.save(path, file_bundle_out)
        print("File package {} out of {} saved successfully".format(i+1, len(x_train)), end="\r")

def translate_image(img, right, down):
    """
    A function that moves an image along the x,y direction

    The parts of the image pushed "off" the image are added to the part of the
        image that has received free space

    args:
        img: the image file to apply the translation to (numpy array)
        right: number of pixels to shift image to the right (negative value shifts left)
        down: number of pixels to shift image down (negative value shifts up)

    returns:
        img: translated image (numpy array)
    """
    rows,cols = img.shape[:2]
    m = np.float32([[1,0,right],[0,1,down]])
    img = cv2.warpAffine(img, m, (cols,rows), borderMode=cv2.BORDER_WRAP)
    return(img)

def change_brightness(img, value=30, mode="lighten"):
    """
    changes the brightness of an image

    Assumes images are saved in BGR format with values between 0 and 255. If a
        pixel value would be bigger (or smaller) than 255 (0) then it is clamped
        at a value of 255 (0)

    args:
        img: image to be used (numpy array)
        value: amount to shift change the brightness by
        mode: set to 'lighten' or 'darken'
    returns:
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
    applies numerous changes to an image to expand a dataset

    We 'augment' an image with five different methods and thereby produce 11
    version of our original image (counting the original version). The methods
    applied are: (A) horizontal flipping (B) blurring (C) Translation (4x) (D)
    change of brightness (2x) (E) conversion to black and white.
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
    """takes an array of images and labels and applies augmentations

    The function applies a set of image augmentation functions to extend the
    size of an image set which in turn allows better training of algorithms,
    especially when the number of training data is quite low.

    args:
        image_files: an numpy array containing images
        labels: the corresponding labels to the images
        shuffle: whether or not to shuffle the final image set (could be useful
            for later training as the algorithm will otherwise train on a similar
            image several times in a row)

    returns:
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
    """takes a synset id from wordnet and returns all direct children synsets
    """
    b = requests.get("http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=" + synset_id)
    synsets = b.text.split("\r\n-")
    synsets[-1] = synsets[-1][:-2]
    return synsets

def get_synset_image_urls(synset_id):
    """takes a synset id and fetches urls of associated images
    """
    website = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=" + synset_id)
    website_content = BeautifulSoup(website.content, 'html.parser')
    website_content_string = str(website_content)
    website_lines = website_content_string.split('\r\n')
    return (website_lines)

def timeout_handler(num, stack):
    raise Exception()

def download_image(url):
    """
    downloads an image from a link and resizes it to desired size

    https://stackoverflow.com/questions/54160208/how-to-use-opencv-in-python3-to-read-file-from-file-buffer/54162776#54162776
    updated np.frombuffer as older version caused a deprecation warning
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
    return np.array(image_list)

def create_imagenet_dataset_random(size, max_synset_imgs, forbidden_synset, exclude_synset_children):
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
    return np.array(image_list)

"""
visualisation:
- show images_v2
- show transfer vals
- balance training set

Then:
* AutoML
* Fetch Wordnet Datasets
* Write on pre-processing
* Hyperopt transfer learn
"""

if __name__ == "__main__":

    target_np_folder = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files"

    ##############################
    ### load images with label ###
    ##############################

    # for cars
    car_image_folders = ["/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/car/car",
                        "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/car/no_car"]
    car_json_folder = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/json_files/cars"
    car_names_raw, car_images = load_images(car_image_folders)
    car_labels = read_label_json(car_json_folder)
    car_names, car_files = return_labelled_images(car_labels, car_names_raw, car_files)
    save_to_numpy_with_labels(target_np_folder, car_files, car_names["label"], "car", augment_training_data=True, split_into_train_test=True)


    #################################
    ### load images without label ###
    #################################
    save_to_numpy(folder_path=target_np_folder,
                img_names=car_names_raw["file_name"],
                files=car_images,
                object="testing_data")




    # # for apparel
    # apparel_image_folders = ["/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/apparel/apparel",
    #                     "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/apparel/no_apparel"]
    # apparel_json_folder = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/json_files/apparel"
    # apparel_names_raw, apparel_files = load_images(apparel_image_folders)
    # apparel_labels = read_label_json(apparel_json_folder)
    # apparel_names, apparel_files = return_labelled_images(apparel_labels, apparel_names_raw, apparel_files)
    # save_to_numpy_with_labels(target_np_folder, apparel_files, apparel_names["label"], "apparel", augment_training_data=True, split_into_train_test=True)
    #
    # # for food
    # food_image_folders = ["/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/food/food",
    #                     "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/food/no_food"]
    # food_json_folder = "/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/json_files/food"
    # food_names_raw, food_files = load_images(food_image_folders)
    # food_labels = read_label_json(food_json_folder)
    # food_names, food_files2 = return_labelled_images(food_labels, food_names_raw, food_files)
    # save_to_numpy_with_labels(target_np_folder, food_files2, food_names["label"], "food", augment_training_data=True, split_into_train_test=True)
    #










"""
from PIL import Image
img = load_single_image("/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/apparel/apparel/RalphLauren_6797246895_10153439142626896_.jpg")

files, labs = augment_single_image(img, 1)
img2 = Image.fromarray(translated, 'RGB')
img2.show()

blurring:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
edges: https://github.com/aleju/imgaug/issues/79


# get words of the synsets
a = requests.get("http://www.image-net.org/api/text/wordnet.synset.getwords?wnid=n02958343")
a.text.split("\n")


TOOLS:

### show images and their labels
for j in range(10):
    i = np.random.randint(1,949)
    print(i, food_names.reset_index().loc[i,"file_name"], food_names.reset_index().loc[i,"label"])
    img2 = Image.fromarray(food_files2[i], 'RGB')
    time.sleep(1)
    img2.show()


### searching for empty files
c = []
d = []
for i in range(len(food_files)):
    if food_files[i] is None:
        d.extend([i])
    elif food_files[i].shape != (299, 299, 3):
        c.extend([i])



### create a basic dataframe for testing purposes
pd.DataFrame(data=[[1,2],[3,4]], columns=["hello", "estragon"]).reset_index()



    #################################
    ### load images from imagenet ###
    #################################

    # create an own dataset via image-net
    synset_id = "n02958343" # synset for "auto" (check pls)

    imgs_object = create_imagenet_dataset(synset_id=synset_id, size=10, use_child_synsets=True)
    imgs_random = create_imagenet_dataset_random(size=10, max_synset_imgs=10, forbidden_synset=synset_id, exclude_synset_children=True)

    imgnet_imgs = np.concatenate((imgs_object, imgs_random))
    imgnet_labels = np.array([1]*len(imgs_object) + [0]*len(imgs_random))

    random_order = np.random.permutation(len(imgnet_imgs))
    imgnet_imgs = imgnet_imgs[random_order]
    imgnet_labels = imgnet_labels[random_order]

    path_imgnet_imgs = os.path.join(target_np_folder, "imgnet_automobile_x")
    path_imgnet_labels = os.path.join(target_np_folder, "imgnet_automobile_y")
    np.save(path_imgnet_imgs, imgnet_imgs)
    np.save(path_imgnet_labels, imgnet_labels)

# x = np.load("/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/imgnet_automobile_x.npy")
# y = np.load("/Users/dominiquepaul/xBachelorArbeit/Spring19/Data/np_files/imgnet_automobile_y.npy")


"""









#
