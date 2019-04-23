#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 15:37:36 2018

@author: dominiquepaul

version: 1.1
"""


#import modules
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import cv2
import keras
from IPython.display import Image, display
from tqdm import tqdm
from sklearn.model_selection import train_test_split
#from datetime import timedelta
# import time




        
######################################################
# Definition of relevant parameters
######################################################
   
# The following parameters MUST be set for each new image group

os.chdir("/Users/dominiquepaul/xBachelorArbeit/Daten/domiwom") # your working directory
# The working directory must contain the following: 
# The folder called "inception"; 
# The files: cache.py; download.py; inception.py;
# A folder created by yourself called "Data" which contains a subfolder with the 
# the name of the object you are trying to identify. Please assign the name of this
# folder to the following variable "object_":
object_ = "cleaned_car" 
# This folder must in turn contain two folders with the respective images placed 
# in them. Please assign the following variable "labels" the two names of the folders
# as strings in the order: [no_object_folder_name, object_folder_name]:
labels = ["no_car", "car"]


# The setting of the following parameters is optional 
visualisation_on = False # Determines whether you want visualisations turned on or off (True = on)

# hyper parameters for training the model
train_frac = 0.7 # determines the fraction of training data in the train-test-split
steps = 1000 # Number of training steps (batches) when training the model
display_every = 200 # After how many training steps to present progress. 200 is recommended


num_classes = 2

######################################################
# End of parameter definition
######################################################




# Functions and classes for loading and using the Inception model are imported
# This can only be done after the directory was set
import inception
from inception import transfer_values_cache


# several subpaths are defined with a system which relies on the folder structure 
# described in the parameter description part

path_gen = os.getcwd()
path_data = os.path.join(path_gen,"Data",object_)
# sets the two data paths,
path_raw_data_class0 = os.path.join(path_data,labels[0])
path_raw_data_class1 = os.path.join(path_data,labels[1])

# the paths for the processed data
path_prepared_images0 = path_data + "/cropped_resized_" + str(labels[0]) + "/"
path_prepared_images1 = path_data + "/cropped_resized_" + str(labels[1]) + "/"



######################################################
# Defining Helper Functions
######################################################

def create_image_df(path, label):
    """Exclude images from df out of image aspect ratio arange
    
    Input:
        df: a data frame with a columns called "aspect_ratio"
        min: minimal aspect ratio of image
        max: maximal aspect ratio of image
    
    Output:
        df: a data frame similar to the inputted data frame with all unsuiting aspect ratios removed
        
        image_df: the data frame with file_path in first column, pixel width in the second
            column, pixel height in the third column. 
    """
    
    
    image_df = pd.DataFrame()
    for subdir, dirs, files in os.walk(path):
        for i,file in enumerate(tqdm(files)):
            try:
                name = file
                file = os.path.join(subdir, file)
                img = cv2.imread(file)
                height, width = img.shape[:2]
                aspect_ratio = width / height
                image_df = image_df.append([[file,width,height,aspect_ratio, label]], ignore_index=True)
            except:
                print("Following file was excluded due to non-readability: {0}".format(name))
    # renames the columns
    image_df.columns = ["image","width","height","aspect_ratio", "is_object"]
    return(image_df)

def pick_aspects(df, min, max):
    """Exclude images from df out of image aspect ratio arange
    
    Input:
        df: a data frame with a columns called "aspect_ratio"
        min: minimal aspect ratio of image
        max: maximal aspect ratio of image
    
    Output:
        df: a data frame similar to the inputted data frame with all unsuiting aspect ratios removed
    """
    initial_length = len(df)
    df = df.loc[df.loc[:,"aspect_ratio"] > min,:]
    df = df.loc[df.loc[:,"aspect_ratio"] < max,:]
    final_length = len(df)
    removed = initial_length - final_length
    print(str(removed) + " out of " + str(initial_length) + " images were removed")
    return(df)

def resize_image(img, size):
    """Resizes square images to a square of a different size
    
    Input:
        img: a square image         MISSING FORMAT
        size: the size to which the image shall be resized
    
    Output:
        img: a square image         MISSING FORMAT
    """
    # hier könnte man theoretisch nur nach breite, oder nur nach länge gehen,
    # da eine der annahmen ist, dass nur bereits quadratische bilder eingegeben werde

    output_height = output_width = size
    height, width = img.shape[:2]

    # Shrinking?
    if output_height < height or output_width < width:
        scaling_factor = output_height / float(height)
        if output_width/float(width) < scaling_factor:
            scaling_factor = output_width / float(width)
        img = cv2.resize(img, None, fx= scaling_factor, fy = scaling_factor, interpolation = cv2.INTER_AREA)

    # Expanding? # to-self: not validated
    if output_height > height or output_width > width:
        scaling_factor = output_height / float(height)
        if output_width/float(width) > scaling_factor:
            scaling_factor = output_width / float(width)
        img = cv2.resize(img, None, fx= scaling_factor, fy = scaling_factor, interpolation = cv2.INTER_AREA)

    return(img)

def crop_and_resize(image_df, output_folder, labels, resize_to = None):
    """Three square crops of an image are taken, these are resized to a certain size,
    and these image are saved to a folder on the local hard drive
    
    Input:
        image_df: the data frame with file_path in first column, pixel width in the second
            column, pixel height in the third column. 
        output_folder: the file_path of the folder to which the crops shall be saved
        labels: the labels of the two types of images, which were entered in the very 
            beginning of the script. Are necessary to name two new subfolders
        resize_to: Optional parameter. Determines the pixel size to which new images are 
            resized. If no value is entered the images are not resized (not recommended)
    Output:
        None
    """
    omitted = 0
    for i in tqdm(range(len(image_df))):
        try:
            width = image_df.iloc[i,1]
            height = image_df.iloc[i,2]
            location = image_df.iloc[i,0]
            file_name = location.split("/")[-1][0:-4]
            image_original = cv2.imread(location)
            label = image_df.iloc[i,4]
            width_list = [int(round((width/2 - height/2),0)),int(round((width/2 + height/2),0))]
            height_list = [int(round((height/2 - width/2),0)),int(round((height/2 + width/2),0))]
            # we take three sqaure crops of the image, this requires the image to have an
            # aspect ratio of <2 and >0.5
            if width > height:
                crop1 = image_original[0:height, 0:height]
                crop2 = image_original[0:height, width_list[0]:width_list[1]]
                crop3 = image_original[0:height, (width-height):width]
            elif width < height:
                crop1 = image_original[0:width, 0:width]
                crop2 = image_original[height_list[0]:height_list[1], 0:width]
                crop3 = image_original[(height-width):height, 0:width]
            else: # case where image already was square
                # If an image is already square we save it three times so that
                # it is used equally often in training and the training data isnt skewed
                crop1 = crop2 = crop3 = image_original
            # resize image
            if resize_to is not None:
                crop1 = resize_image(crop1, resize_to)
                crop2 = resize_image(crop2, resize_to)
                crop3 = resize_image(crop3, resize_to)

            path = output_folder + "/cropped_resized_" + str(labels[label]) + "/"
            if not os.path.exists(path):
                os.makedirs(path)

            #cv2.imwrite(path + file_name  +"crop1.jpg", crop1)
            cv2.imwrite(path + file_name  +"crop2.jpg", crop2)
            #cv2.imwrite(path + file_name  +"crop3.jpg", crop3)
        except:
            omitted += 1
    print("{0} out of {1} images had to be skipped due to non-readability".format(omitted, len(image_df)))
    #print(str(omitted) + " out of " + str(len(image_df)) + " images had to be skipped due to non-readability")


#helper function to plot images
def show_images_v2(image_inputs, label_inputs, predictions_image_class = None):
    """A helper function which plots images to a 3x3 grid, annotated by their labels and 
    optionally annotated by the predictions of the own model.
    
    Input:
        image_inputs: An array containing the pixel values of 9 images
        label_inputs: The labels of the images, either as 0 or 1 (0= is not object, 1= is object)
        predictions_image_class: A numpy array of predictions made by the model either as 0 or 1 
        
    Output:
        A 3x3 plot of images annotated with labels
    """
    fig, axes = plt.subplots(3,3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.6, wspace=0.3)
    for i, ax in enumerate(axes.flat):

        img = image_inputs[i]
        b,g,r = cv2.split(img)       # get b,g,r
        rgb_img = cv2.merge([r,g,b])
        ax.imshow(rgb_img)


        if predictions_image_class is None:
            xlabel = "Label: {}".format(labels[label_inputs[i]])
        else:
            validity =  str(label_inputs[i] == predictions_image_class[i])

            xlabel = "{0} \n Label: {1}\n Model Prediction: {2}".format(validity,
                                                                       labels[label_inputs[i]],
                                                                       labels[predictions_image_class[i]] )

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_transfer_vals(images2, transfer_images):
    """A helper function which plots the three images and their respective transfer values 
    to a 3x2 grid and visualizes the transfer values with colour. The plots have no direct 
    visual meaning but serve to show how the the transfer values differ between images and 
    might be an indication of features contained in the images. The transfer values are 
    transformed from a one-dimensional space into a two dimensional space to improve 
    visualisation
    
    Input:
        images2: the pixel values of three images as nested numpy arrays
        transfer_images: the numeric numbers of the transfer values
        
    Output:
        A 3x2 plot of visualised transfer values
    """
    
    print("\n\n Images and their respective transfer values:")
    fig, axes = plt.subplots(2,3,figsize = (14,7))
    for i,ax in enumerate(axes.flat):
        if i <= 2:
            b,g,r = cv2.split(images2[i])       # get b,g,r
            rgb_img = cv2.merge([r,g,b])
            ax.imshow(rgb_img)
        else:
            img = transfer_images[i-3].reshape((32, 64))
            ax.imshow(img, cmap = "PuBuGn") # Other CMAPs: hot_r, seismic, gnuplot_r, PuBuGn
    fig.tight_layout()


def next_batch(size = 64, data_type = "train"):
    """A helper function which serves to fetch a number of random images and their 
    respective labels from either the training or the test set to train or evaluate 
    the model.
    
    Input:
        size: The number of images which are to be fed as one batch, default is 64
        data_type: determines the source of the images and labels. This can either be from 
            the training set ("train") which is set as default, or from the test set ("eval")
        
    Output:
        transfer_values: returns the transfer values of random images 
        one_hot_labels: returns the labels of these images in the one hot encoding format
    """
    if data_type == "train":
        data = transfer_values_train
        labels = y_train
    if data_type == "eval":
        data = transfer_values_test
        labels = y_test

    indeces = np.arange(0,len(data))
    indeces = np.random.choice(indeces, size = size, replace = False)

    transfer_values = data[indeces]
    one_hot_labels = labels[indeces]

    return(transfer_values, one_hot_labels)

def delete_corrupted_images(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
            input_amount = len(files)
            deleted = 0
            for file in tqdm(files):
                path = os.path.join(subdir, file)
                img = cv2.imread(path)
                try:
                    if img.mean() == 0:
                        deleted += 1
                        os.remove(path)
                except: 
                    print("file {} was skipped".format(path)) 
            print("{0} out of {1} files had to be deleted due to non-readability".format(deleted, input_amount)) # technically not quite accurate due to line above

        
######################################################
# End of helper functions
######################################################
   

   
######################################################
# Create CSV where untrained and trained model accuracies are displayed
######################################################
   



# create a pandas dataframe with the summarizing info of all images in the folders
# this includes the image path, their width, height, aspect ratio and a label of the object
no_object_df = create_image_df(os.path.join(path_gen, path_raw_data_class0), 0)
object_df = create_image_df(os.path.join(path_gen, path_raw_data_class1), 1)


# Only shows the distribution of aspect ratios if visualisation option is on
if(visualisation_on):
    # plot distribution of the aspect ratios
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    aspect_array_no_object = np.array(no_object_df["aspect_ratio"])
    aspect_array_object = np.array(object_df["aspect_ratio"])
    ax1.hist(aspect_array_object, bins = 15)
    ax2.hist(aspect_array_no_object, bins = 15)
    ax1.set_xlabel("width / height")
    ax1.set_ylabel("# of occurences")
    ax2.set_xlabel("width / height")
    ax2.set_ylabel("# of occurences")
    ax1.set_title('Object: {} images'.format(len(aspect_array_object)))
    ax2.set_title('No Object: {} images'.format(len(aspect_array_no_object)))
    plt.suptitle('Distribution of the aspect ratio')
    plt.show()


# exclude all images with unsuiting aspect ratio
# by default all images with an aspect ratio below 0.5 or above 2 are excluded
no_object_df = pick_aspects(no_object_df,0.5,2.0)
object_df = pick_aspects(object_df,0.5,2.0)

# resize each image according to size and make three square crops of it
crop_and_resize(no_object_df, path_data, labels, resize_to = 128)
crop_and_resize(object_df, path_data, labels, resize_to = 128)



# Some images are "broken" and are simply black pixels. This function deletes these images
delete_corrupted_images(path_prepared_images0)
delete_corrupted_images(path_prepared_images1)


# re-read images
no_object_set = create_image_df(path_prepared_images0, 0)
object_set = create_image_df(path_prepared_images1, 1)

# Merge both image data frames to one data frame
image_df = no_object_set.append(object_set, ignore_index = True)
image_df.head()

before = len(image_df)
image_df = image_df.loc[image_df.loc[:,"aspect_ratio"] == 1.0,:] # omit images without perfect aspect ratio of 1
print("{0} out of {1} rows were omitted".format(before - len(image_df),before))

test1 = image_df.copy()
test1.reset_index(inplace = True, drop = True)
test1.head()



######################################################
# Create numpy arrays with image pixel data
######################################################
   

# index
image_df.reset_index(inplace = True,drop = True)
index = np.array(image_df.index)

# images
# we initialize the empty array first to increase speed
image_shape = cv2.imread(image_df["image"][0]).shape
# default dtype would be float64 but this causes our images to be displayed differently later on
image_array = np.zeros([len(image_df), image_shape[0],image_shape[1] ,image_shape[2]], dtype = "uint8")

for i,file in enumerate(tqdm(image_df["image"])):
    image_array[i] = cv2.imread(file)

# sanity check
print(image_array.shape)
print(image_array.dtype)


# split into training and test

x_train, x_test, index_train, index_test = train_test_split(image_array, index, train_size=train_frac, random_state = 2 ) # random state to be deleted once finsihed

df_train = image_df.iloc[index_train].copy()
df_test = image_df.iloc[index_test].copy()
# sanity check
df_train["original_index"] = index_train
df_test["original_index"] = index_test
df_train.reset_index(inplace = True, drop = True)
df_test.reset_index(inplace = True, drop = True)
df_train.head()

class_labels_train = np.array(df_train["is_object"])
class_labels_test = np.array(df_test["is_object"])

# conversion to one hot encoding
y_train = keras.utils.to_categorical(class_labels_train, num_classes )
y_test = keras.utils.to_categorical(class_labels_test, num_classes )

if visualisation_on:
    images1 = x_train[18:27]
    labels1 = class_labels_train[18:27]
    show_images_v2(images1,labels1)


# Download inception model
print("downloading inception v3 model if not already done")
inception.data_dir = 'inception/'
inception.maybe_download()
model = inception.Inception()


#calculate transfer values
file_path_cache_train = os.path.join(path_data, 'inception_train.pkl')
file_path_cache_test = os.path.join(path_data, 'inception_test.pkl')

print("Processing Inception transfer-values for training-images ...")

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=x_train,
                                              model=model)

print("Processing Inception transfer-values for test-images ...")

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             images=x_test,
                                             model=model)

# Only plot transfer values if visualisation is on
if visualisation_on:
    images1 = x_train[0:3]
    transfer_images = transfer_values_train[0:3]
    plot_transfer_vals(images1,transfer_images)






######################################################
# Building a New Classifier with Tensorflow
######################################################

print("\n Building neural net...")



tf.reset_default_graph()
keep_prob = 0.5
with tf.name_scope('input'):
    x_ = tf.placeholder(tf.float32, shape = [None, transfer_values_train.shape[1]], name = "input1")
    y_ = tf.placeholder(tf.float32, shape = [None, num_classes], name = "labels")

# Dense Layer one
with tf.name_scope('inner_layers'):
    first_layer = tf.layers.dense(x_, units = 1024, activation=tf.nn.relu, name = "first_dense")
    dropout_layer = tf.nn.dropout(first_layer, keep_prob, name = "kinda_dropout")
#    second_layer = tf.layers.dense(dropout_layer, units = 2048, activation=tf.nn.relu, name = "second_dense")
#    dropout_layer2 = tf.nn.dropout(second_layer, keep_prob, name = "kinda_dropout2")
    
    y_conv = tf.layers.dense(dropout_layer, units = 2, activation=tf.nn.relu, name = "wut")

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_ , logits = y_conv),
                                   name = "cross_entropy_layer")

with tf.name_scope('trainer'):
    train_step = tf.train.AdamOptimizer(0.00001).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    prediction = tf.argmax(y_conv, axis = 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y_, axis = 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


######################################################
# Run model
######################################################

print("\n Training neural net...")

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

num_images = len(x_test)

# Accuracy on test data
batch_test_x, batch_test_y = next_batch(size = num_images, data_type ="eval")
print("Initial test accuracy is {0:.3f}%".format(accuracy.eval(feed_dict= {x_: batch_test_x, y_: batch_test_y}) * 100))

steps = 10000

for i in tqdm(range(steps)):
    batch_x, batch_y = next_batch(size = 64)
    sess.run(train_step, feed_dict = {x_: batch_x, y_: batch_y})

    if i % display_every == 0:
        train_accuracy = accuracy.eval(feed_dict = {x_: batch_x, y_: batch_y}) * 100
        print("\nStep: {0}, training accuracy = {1:.3f}%".format(i,train_accuracy))
        
        batch_test_x, batch_test_y = next_batch(size = num_images, data_type ="eval")
        print("\nTest accuracy is {0:.3f}%".format(accuracy.eval(feed_dict= {x_: batch_test_x, y_: batch_test_y}) * 100))

# Accuracy on test data
batch_test_x, batch_test_y = next_batch(size = num_images, data_type ="eval")
print("\nTest accuracy is {0:.3f}%".format(accuracy.eval(feed_dict= {x_: batch_test_x, y_: batch_test_y}) * 100))


sess.close()






######################################################
# Mark classification in test and train set
######################################################

print("\n Classifying images...")

for i, transfer_val in enumerate(tqdm(transfer_values_train)):
    transfer_val_transformed = np.array([transfer_val])
    true_label_one_hot = [y_train[i]]
    true_label = np.argmax(y_train[i])
    model_prediction = prediction.eval(feed_dict = {x_: transfer_val_transformed, y_: true_label_one_hot})
    if true_label == model_prediction:
        match = "Correct"
    else:
        match = "False"

    df_train.loc[i,"model_prediction"] = int(model_prediction)
    df_train.loc[i,"original_label"] = int(true_label)
    df_train.loc[i,"correct"] = match

# test set
for i, transfer_val in enumerate(tqdm(transfer_values_test)):
    transfer_val_transformed = np.array([transfer_val])
    true_label_one_hot = [y_test[i]]
    true_label = np.argmax(y_test[i])
    model_prediction = prediction.eval(feed_dict = {x_: transfer_val_transformed, y_: true_label_one_hot})
    if true_label == model_prediction:
        match = "Correct"
    else:
        match = "False"

    df_test.loc[i,"model_prediction"] = int(model_prediction)
    df_test.loc[i,"original_label"] = int(true_label)
    df_test.loc[i,"correct"] = match


######################################################
# Merge df with train and test data
######################################################

print("\n Saving classifications as csv...")

joined_df = df_train.append(df_test,ignore_index=True)
joined_df.head()



original_order = np.array(joined_df["original_index"])
output_df = joined_df.loc[original_order,["image", "model_prediction","original_label", "correct"]].copy()
output_df.reset_index(inplace = True, drop = True)
output_df.head()


output_df2 = output_df.copy()

for i,element in enumerate(output_df["image"]):
    file_name_solo = element.split("/")[-1]
    no_crop_in_name = file_name_solo[0:-9] + file_name_solo[-4:]
    output_df2.loc[i,"image"] = no_crop_in_name


output_df2.to_csv(os.path.join(path_data, object_ +"_readout.csv"))


# Visualisation of Model Predictions if visualisation option is on
if visualisation_on:
    prediction_array = np.array(df_train["model_prediction"], dtype = "uint8")
    idx = np.random.randint(0, len(x_train) - 9)
    images1 = x_train[idx:idx+9]
    labels1 = class_labels_train[idx:idx+9]
    predictions1 = prediction_array[idx:idx+9]
    
    show_images_v2(images1,labels1, predictions1)



######################################################
# create function to run model and return test accuracy
######################################################
    
#check for misclassification

orig_length = len(output_df2)
# car but classified as no_car
misclas1 = len(output_df2.loc[output_df2["correct"] == "False"].loc[output_df2["original_label"] == 1]["image"])
print("{} out of {} images were classified as no_car despite  being a car".format(misclas1,orig_length))
# no_car but classified as car
misclas2 = len(output_df2.loc[output_df2["correct"] == "False"].loc[output_df2["original_label"] == 0])
print("{} out of {} images were classified as car despite not being a car".format(misclas2,orig_length))














######################################################
# create function to run model and return test accuracy
######################################################

def run_model(steps_ = 2000):
    
    # determine number of images to be tested 
    num_images = len(x_test)
    

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        #tbWriter = tf.summary.FileWriter(log_path,sess.graph)

        # Accuracy on test data
        batch_test_x, batch_test_y = next_batch(size = num_images, data_type ="eval")

        for i in range(steps_):
            batch_x, batch_y = next_batch(size = 64)
            sess.run(train_step, feed_dict = {x_: batch_x, y_: batch_y})

        # Accuracy on test data
        batch_test_x, batch_test_y = next_batch(size = num_images, data_type ="eval")
        test_accuracy_ = accuracy.eval(feed_dict= {x_: batch_test_x, y_: batch_test_y}) * 100

        return(test_accuracy_)
        
        
######################################################
# Create CSV where untrained and trained model accuracies are displayed
######################################################
      
print("\n Testing accuracy of several untrained and trained models. \n This might take some time...")

without_training = []
accuracy_results = []

for i in tqdm(range(10)):
    without_train = round(run_model(steps_=0), 2)
    without_training.append(without_train)

for i in tqdm(range(10)):
    accuracy_result = round(run_model(steps_=3000), 2)
    accuracy_results.append(accuracy_result)

result_df = pd.DataFrame()
result_df["without_training"] = without_training
result_df["After_training"] = accuracy_results
# save as csv file
result_df.to_csv(os.path.join(path_data, object_ +"_accuracy_results.csv"))


result_df.head()
