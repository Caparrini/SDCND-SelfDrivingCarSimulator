import csv
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

import cv2 as cv
import matplotlib as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True
set_session(tf.Session(config=config))

def get_base_path():
    return os.path.dirname(os.path.realpath(__file__))

def get_dataset(folders_data=["data","data2"]):
    main_dir = get_base_path()
    data_dirs = [os.path.join(main_dir, folder_data) for folder_data in folders_data]

    # Read the csv file to collect the image and steering data
    images = []
    measurements = []
    for data_dir in data_dirs:
        lines = []
        with open(os.path.join(data_dir, "driving_log.csv")) as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                lines.append(line)
        del lines[0]
        # Collect the images and measurements

        for line in lines:
            source_path = line[0]
            filename = source_path.split("/")[-1]
            current_path = os.path.join(data_dir, "IMG", filename)
            image = cv2.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
            #Augmented data
            images.append(np.fliplr(image))
            measurements.append(-measurement)

    # Transform to np.array
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train

def get_nmodel():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Conv2D(24, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    return model

if __name__ == "__main__":

    X_train, y_train = get_dataset()

    model = get_nmodel()

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=6)

    model.save(os.path.join(get_base_path(), "model.h5"))
