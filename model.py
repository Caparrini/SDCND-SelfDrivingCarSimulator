import csv
import os
import cv2
import numpy as np

main_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(main_dir, "data")

lines = []
with open(os.path.join(data_dir, "driving_log.csv")) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)
del lines[0]

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split("/")[-1]
    current_path = os.path.join(data_dir, "IMG", filename)
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D



model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(160,320,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(180))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=20)

model.save(os.path.join(main_dir, "model.h5"))
