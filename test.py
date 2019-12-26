from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import cv2
from os import path

# initialize the model
model = Sequential()
inputShape = (28, 28, 1)

# define the first set of CONV => ACTIVATION => POOL layers
model.add(Conv2D(32, 5, padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# define the second set of CONV => ACTIVATION => POOL layers
model.add(Conv2D(64, 5, padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# define the first FC => ACTIVATION layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# define the second FC layer
model.add(Dense(10))

# lastly, define the soft-max classifier
model.add(Activation("softmax"))

# if a weights path is supplied (inicating that the model was
# pre-trained), then load the weights
model.load_weights("weights.h5")

# randomly select a few testing digits
for i in range(1,28):
	image = cv2.imread("images2/"+str(i)+".jpg",0)
	ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
	image1 = image.copy()
	testData = image.reshape((1, 28, 28, 1))
	# classify the digit
	probs = model.predict(testData)
	prediction = probs.argmax(axis=1)
	print("Recognised Digit: {}".format(prediction[0]))
	cv2.imshow("Digit", image1)
	cv2.waitKey(0)
