import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from time import sleep
print("Tensorflow Version:", tf.__version__)

#Load data set into train and test variables
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalize and convert data to floats
x_train, x_test = x_train/255.0, x_test / 255.0

MNIST_model = tf.keras.models.load_model("MNIST_Seq_model.h5")

#classify a single input
def classify_single(input):
    if(input.ndim != 3):
        input = tuplify_input_image(input)
    
    if(input.shape[1] != 28 or input.shape[2] != 28):
        print(input.shape)
        print("Invalid Image Size. Resize image to 28x28")
        return

    y = MNIST_model.predict(input, verbose=2)[0]
    #print("Prediction array: ",y)
    y = int(np.where(y == np.amax(y))[0])
    #print("The number is", y)
    return y

#Our model requires a tuple so convert input image to tuple if not already
def tuplify_input_image(img):
    if(img.ndim != 3):
        prep_img = np.array([img])
        return prep_img

# TODO: Turn this image processing into a function -> likely in another file
#img = cv.imread('seven.png', cv.IMREAD_GRAYSCALE)
#img = img/255.0
#th,img = cv.threshold(img, 0.5, 1, cv.THRESH_BINARY_INV)
#img = cv.resize(img,(20,20),interpolation=cv.INTER_LINEAR)
#img = cv.copyMakeBorder(img, 4,4,4,4, cv.BORDER_CONSTANT, None, [0,0,0])
#img = x_test[8:9]
#img = img[0]
#implot = plt.imshow(img, cmap="gray")
#plt.show()
#classify_single(img)