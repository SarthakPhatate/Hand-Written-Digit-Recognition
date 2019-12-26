# Hand-Written-Digit-Recognition

Recognition of digits from Convolutional Neural Network in Python

I have uset the MNIST dataset of Hand Written Numbers The MNIST dataset is arguably the most well-studied, most understood dataset in the computer vision and machine learning literature.
The goal of this dataset is to classify the handwritten digits 0-9. We’re given a total of 70,000 images, with (normally) 60,000 images used for training and 10,000 used for evaluation; however, we’re free to split this data as we see fit. Common splits include the standard 60,000/10,000, 75%/25%, and 66.6%/33.3%. I’ll be using 2/3 of the data for training and 1/3 of the data for testing later in the blog post.

Each digit is represented as a 28 x 28 grayscale image (examples from the MNIST dataset can be seen in the figure above). These grayscale pixel intensities are unsigned integers, with the values of the pixels falling in the range [0, 255]. All digits are placed on a black background with a light foreground (i.e., the digit itself) being white and various shades of gray.

It’s worth noting that many libraries (such as scikit-learn) have built-in helper methods to download the MNIST dataset, cache it locally to disk, and then load it. These helper methods normally represent each image as a 784-d vector.


Traning the dadaset:
  The train.py file is used to train the dataset with 60000 images and also validate on 10000 images.
  The weights of trained network is stored in weights.h5 file
  
Testing :
  The test.py file is used to test the trained network on different images as in images folder
