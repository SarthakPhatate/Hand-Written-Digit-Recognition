# import the necessary packages
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
if path.exists("weights.h5"):
	model.load_weights("weights.h5")


# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("downloading MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# matrix shape should be: num_samples x rows x columns x depth
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# initialize the optimizer and model
print("compiling model...")
opt = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

print("training...")
model.fit(trainData, trainLabels, batch_size=128, epochs=10,verbose=1)
#accuracy on the testing set
print("evaluating...")
(loss, accuracy) = model.evaluate(testData, testLabels,batch_size=128, verbose=1)
print("accuracy: {:.2f}%".format(accuracy * 100))


print("saving weights to current directory...")
model.save_weights("weights.h5", overwrite=True)
