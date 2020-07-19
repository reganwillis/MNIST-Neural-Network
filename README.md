# MNIST-Neural-Network
This program was made with the help of this [Tensorflow tutorial](https://www.tensorflow.org/tutorials/keras/classification).

# Dataset
I used the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/), the version that is [built in](https://keras.io/api/datasets/mnist/) to the Tensorflow Keras API. The training dataset consists of 60,000 images and labels, while the testing dataset consists of 10,000 images and labels. Each image is 28x28 pixels.

# Preprocessing
Since the Keras API has datasets that are meant for ML practice, the only preprocessing necessary was to scale the pixel value range from 0-255 to 0-1.

# The Net
I used a Multilayer Perceptron with one hidden layer. I used a sequential model, which is the most common form of model used in neural networks. The input layer reformats the images from a two-dimensional to a one-dimensional array of pixels. The hidden layer uses 128 nodes and the rectified linear unit activation function. The output is in the form of a logit array.

# Compiling the Model
I used an adaptive moment estimation optimizer, one of the most popular gradient descent optimization algorithms, and an accuracy metric which calculates how often the prediction equals the label.

# Model Training
The current version of the program uses 10 epochs, which means the model will label all the test images correctly from what I've seen so far. Lowering the number of epochs to 1 means the model will be less sure about most of the labels, and will get a few wrong. Each iteration of the program will look a little different, and it is fun to experiment with changing the number of epochs.

# The Results
The program plots the first few results from the test dataset. Here are the results with 1 epoch and 10 epochs, respectively:
