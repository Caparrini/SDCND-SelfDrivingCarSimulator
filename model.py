import csv
import os
import cv2
import numpy as np

data_dir = os.path.join(os.getcwd(), "data")

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
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=100)

model.save("model.h5")
