import csv
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.convolutional import Conv2D
import matplotlib.pyplot as plt


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def init_gpu_conf():
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
            imageBGR = cv2.imread(current_path)
            image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)
            #Augmented data
            images.append(cv2.flip(image,1))
            measurements.append(-measurement)

    # Transform to np.array
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train

def get_dataset2(folders_data=["data","data1","data2"]):

    main_dir = get_base_path()
    data_dirs = [os.path.join(main_dir, folder_data) for folder_data in folders_data]
    correct_factor = 0.11

    # Read the csv file to collect the image and steering data
    images = []
    steer_angles = []
    for data_dir in data_dirs:
        lines = []
        with open(os.path.join(data_dir, "driving_log.csv")) as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                lines.append(line)
        del lines[0]
        # Collect the images and measurements

        for line in lines:
            print("Processing {0}".format(line))
            filename_center = line[0].split("/")[-1]
            filename_left = line[1].split("/")[-1]
            filename_right = line[2].split("/")[-1]
            current_path_center = os.path.join(data_dir, "IMG", filename_center)
            current_path_left = os.path.join(data_dir, "IMG", filename_left)
            current_path_right = os.path.join(data_dir, "IMG", filename_right)

            #original image
            image_centerBGR = cv2.imread(current_path_center)
            image_center = cv2.cvtColor(image_centerBGR, cv2.COLOR_BGR2RGB)

            image_leftBGR = cv2.imread(current_path_left)
            image_left = cv2.cvtColor(image_leftBGR, cv2.COLOR_BGR2RGB)

            image_rightBGR = cv2.imread(current_path_right)
            image_right = cv2.cvtColor(image_rightBGR, cv2.COLOR_BGR2RGB)

            aux_images = [image_center, image_left, image_right]
            images.extend(aux_images)

            steer_center = float(line[3])
            steer_left = float(line[4]) + correct_factor
            steer_right = float(line[5]) - correct_factor

            aux_angles = [steer_center, steer_left, steer_right]
            steer_angles.extend(aux_angles)
            #Augmented data (flipped image)
            for image in aux_images:
                images.append(cv2.flip(image,1))
            for angle in aux_angles:
                steer_angles.append(-angle)

    # Transform to np.array
    X_train = np.array(images)
    y_train = np.array(steer_angles)
    return X_train, y_train

def generator(samples, batch_size=32):
    samples_size = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, samples_size, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                filename = source_path.split("/")[-1]
                current_path = os.path.join(get_base_path(), "data", "IMG", filename)
                imageBGR = cv2.imread(current_path)
                image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)

                images.append(image)
                measurement = float(batch_sample[3])
                measurements.append(measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

def get_dataset_generators(folders_data=["data","data1","data2"]):
    main_dir = get_base_path()
    data_dirs = [os.path.join(main_dir, folder_data) for folder_data in folders_data]

    samples = []
    for data_dir in data_dirs:

        data_log = os.path.join(get_base_path(),data_dir,"driving_log.csv")
        with open(data_log) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)
            del samples[0]
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    return train_generator, validation_generator, len(train_samples), len(validation_samples)


def get_dummymodel():
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model

def get_nVidiaModel():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    model.add(Conv2D(24, kernel_size=(5, 5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), subsample=(2,2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))


    return model

if __name__ == "__main__":
    init_gpu_conf()

    #X_train, y_train = get_dataset2()
    train_gen, validation_gen, n_train, n_valid = get_dataset_generators()
    model = get_nVidiaModel()
    model.compile(loss='mse', optimizer='adam')
    ##history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)
    history_object = model.fit_generator(generator=train_gen,samples_per_epoch=n_train,validation_data=validation_gen, nb_val_samples=n_valid, nb_epoch=3 )

    model.save(os.path.join(get_base_path(), "model.h5"))
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
