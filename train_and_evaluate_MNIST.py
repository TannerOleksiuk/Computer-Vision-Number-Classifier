import tensorflow as tf 
print("Tensorflow Version:", tf.__version__)

#Load data set into train and test variables
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalize and convert data to floats
x_train, x_test = x_train/255.0, x_test / 255.0

#Build a sequential model by stacking layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

#create vector of logits
predictions = model(x_train[:1]).numpy()
predictions

#Convert logits to probabillities using the softmax
tf.nn.softmax(predictions).numpy()

#Define a loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

#Compile model
model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])

#Train and evaluate model
model.fit(x_train,y_train,epochs=5)

model.evaluate(x_test,y_test,verbose=2)

#Return probabillities
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

probability_model(x_test[:5])

#Save model for use later
model.save('MNIST_Seq_model.h5')