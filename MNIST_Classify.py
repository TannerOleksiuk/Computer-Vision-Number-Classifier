import tensorflow as tf
import numpy as np
import cv2 as cv
#import matplotlib.pyplot as plt # Not required unless you want to use it for displaying images
from time import sleep
print("Tensorflow Version:", tf.__version__)

# Load data set into train and test variables (Not required for normal use, only testing)
#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and convert data to floats
#x_train, x_test = x_train/255.0, x_test / 255.0

# Load model
MNIST_model = tf.keras.models.load_model("MNIST_Seq_model.h5")
# Generate probabillity model
probability_model = tf.keras.Sequential([MNIST_model, tf.keras.layers.Softmax()])

"""
 Convert Single Image to 3D Numpy Array
 Our model requires a 3D input so convert input image to tuple if not already
 img: Image of shape (x, y) where x,y are the dimensions of the imension.
 Therefore image must be grayscale or single colour channel.
"""
def convert3D_input_image(img):
    if(img.ndim != 3):
        prep_img = np.array([img])
        return prep_img

"""
 Classify a single input
 input: Image of shape (28,28) or (1,28,28)
 Image should ideally match formatting of MNIST dataset, binarized 20x20 image padded with 4 pixels
 on each side to make
"""
def classify_single(input):

    # Is the input a 3d? If not convert it.
    if(input.ndim != 3):
        input = convert3D_input_image(input)
    
    if(input.shape[1] != 28 or input.shape[2] != 28):
        print(input.shape)
        print("Invalid Image Size. Resize image to 28x28")
        return

    #y = MNIST_model.predict(input, verbose=2)[0] # If wanted can use this instead of probabillity model
    #print("Prediction array: ",y) # Debug print to see prediction scores

    y_prob = probability_model(input)[0]
    y = int(np.where(y_prob == np.amax(y_prob))[0])

    # Grab probabillity of that number
    y_prob = y_prob[y]
    y_prob = y_prob.numpy()

    #y = int(np.where(y == np.amax(y))[0]) # Not needed now that using prob model

    #print("The number is", y, y_prob) # Debug print

    return y, y_prob


# Below was used for testing. Uncomment to import a single image and use it in classify function

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