# Regan Willis  2020

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PIXEL_SCALE_DIVISOR = 255.0
NUM_OF_NODES = 128
NUM_OF_CLASSES = 10
ACTIVATION = 'relu'
OPTIMIZER = 'adam'
METRIC = 'accuracy'
NUM_OF_EPOCHS = 10  # num of passes through training dataset
VERBOSE = 2  # output format
CORRECT_COLOR = 'blue'
INCORRECT_COLOR = 'magenta'
NUM_DIS_ROWS = 4
NUM_DIS_COLS = 4

# import dataset
mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
(train_imgs, train_labels), (test_imgs, test_labels) = mnist

# get dimensions of training data
dim_1 = train_imgs.shape[1]
dim_2 = train_imgs.shape[2]
input_shape = (dim_1, dim_2)

# preprocess data
train_imgs = train_imgs / PIXEL_SCALE_DIVISOR
test_imgs = test_imgs / PIXEL_SCALE_DIVISOR

# configure layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(NUM_OF_NODES, activation=ACTIVATION),
    tf.keras.layers.Dense(NUM_OF_CLASSES)
])

# compile model
model.compile(optimizer=OPTIMIZER,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                   from_logits=True),
              metrics=[METRIC])

# train model - fit model to training data
model.fit(train_imgs, train_labels, epochs=NUM_OF_EPOCHS, verbose=VERBOSE)

# evaluate accuracy on test dataset
test_loss, test_acc = model.evaluate(test_imgs, test_labels, verbose=VERBOSE)

# add layer that converts logits to probabilities
model.add(tf.keras.layers.Softmax())

# create prediction array of test images
predictions = model.predict(test_imgs)


# function to plot images
def plot_image(predicted_label, predictions_arr, true_label, img):
    # remove plot labels
    plt.xticks([])
    plt.yticks([])
    # change images to grayscale
    plt.imshow(img, cmap=plt.cm.binary)
    x_label = plt.xlabel(f'{predicted_label} -'
                         f'{100*np.max(predictions_arr):2.0f}%')
    # display data accorrding to if net guessed correctly
    if predicted_label == true_label:
        x_label.set_color(CORRECT_COLOR)
    else:
        x_label.set_color(INCORRECT_COLOR)


# function to plot value array
def plot_value_array(predicted_label, predictions_arr, true_label):
    # change plot labels
    plt.xticks(range(NUM_OF_CLASSES))
    plt.yticks([])
    # display plot bar
    thisplot = plt.bar(range(NUM_OF_CLASSES), predictions_arr, color='#777777')
    thisplot[predicted_label].set_color(INCORRECT_COLOR)
    thisplot[true_label].set_color(CORRECT_COLOR)


# plot results
num_imgs = NUM_DIS_ROWS*NUM_DIS_COLS
plt.figure(figsize=(2*2*NUM_DIS_COLS, 2*NUM_DIS_ROWS))
for i in range(num_imgs):
    plt.subplot(NUM_DIS_ROWS, 2*NUM_DIS_COLS, 2*i+1)
    # get highest probable answer
    predicted_label = np.argmax(predictions[i])
    plot_image(predicted_label, predictions[i], test_labels[i], test_imgs[i])
    plt.subplot(NUM_DIS_ROWS, 2*NUM_DIS_COLS, 2*i+2)
    plot_value_array(predicted_label, predictions[i], test_labels[i])
plt.tight_layout()
plt.show()
