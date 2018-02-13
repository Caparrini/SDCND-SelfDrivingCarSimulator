# **Behavioral Cloning**

## Writeup


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./examples/data.jpg
[image1]: ./examples/data1.jpg
[image2]: ./examples/data2.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Included files in the project

My project includes the following files:
* model.py containing the script to create and train the model among other methods finally not used
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results

#### 2. Example of use
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
    python drive.py model.h5
```
I applied 2 modification to drive.py code:
* Add the function init_gpu_conf() to be able to use my GPU
* Change in the speed parameter (a little bit higher)

#### 3. Code of the model

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model contained in the main function.
* get_dataset2() to retrieve the X_train, y_train numpy arrays
* get_nmodel() to generate the convolutional neural network


### Model Architecture and Training Strategy

The model use is mostly the used by the NVIDIA team that has the following components:

* Creation and preprocessing layers. First the cropping layer to cut the parts less interesting of the image to choose a steer angle and second a lambda layer to normalize the input image.
```python
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)),input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
```

* Convolutional layers
```python
    model.add(Conv2D(24, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
```
* Fully conected layers with dropout layers aiming to reduce the overfitting in the model
```python
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Dense(1))
```
* Optimizer and loss function. I diminished the learning rate due to the great amount of data used in the training
```python
    adam = Adam(lr=0.0005)
    model.compile(loss='mse', optimizer=adam)
```

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The first attempt was a fully conected layer trained with the example data which was not really good. Then I started adding a more complex model similar to LeNet and collecting more data using the simulator and augmented data (flipped). This approach had a better behaviour but still failed when the car pass near a curve or it got deviated from the center of the road so I added simulation data moving from sides of the road to the center. That way the model has training examples to learn the behaviour of returning to the center in case of missdirection.

Even with the back to center examples the data was skewed, much of the samples had a near 0 angle so I added the left and right images with the steering correction of 0.3.

To combat overfitting I added too a lap on the second circuit of the simulator. Giving distinct data to train reduce overfitting.

Finally I changed the model for the NVIDIA model described before and use all the training data.

#### 3. Creation of the Training Set & Training Process

The dataset used to train the model contains 3 parts:
* The data example from Udacity
![data][image0]
* Data from the first circuit. Two laps driving by the center of the road to get good driving example, one in each direction. And little maniuvers recentering the car when it was moving far from center.
![data1][image1]
* Data from the second circuit, one lap, to reduce overfitting.
![data2][image2]


I used the left and right images recorder with a steer angle correction of 0.3 to aument the dataset. Also I added a flipped image for each set of images (center, left, right) adding up 69995 samples in total. I tried the generation of samples on the fly (using yielding function) but due to my computer was able to handle the full dataset on memory I did not use the generators in the final training.

