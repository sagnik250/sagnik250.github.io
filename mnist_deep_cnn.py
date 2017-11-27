
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from datetime import timedelta
import tensorflow as tf



#Import MNIST labelled images dataset
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot = True)
print("Training set size is ", len(data.train.labels))
print("Testing set size is ", len(data.test.labels))



#Set up the architecture of the network

#The first convolutional layer will consist of 20 3x3 filters
filter_size_1 = 3     #Filter size is 3 pixels x 3 pixels.
num_filters_1 = 20

#The second convolutional layer will consist of 36 5x5 filters
filter_size_2 = 5
num_filters_2 = 36

#132 neurons in the fully connected layer
fc_size = 132


#Setting the dimensions of the input data

img_size = 28                           #MNIST images are of pixel dimension 28x28
img_area = img_size * img_size          #In order to flatten the 28x28 matrix to a 1D array
img_shape = (img_size, img_size)        #Tuple with height and width of images to reshape arrays
channels = 1                            #Number of colour channels for the image; 1 for greyscale
num_classes = 10                        #Number of classes for each of the 10 digits



#We build a function first that  plots 9 images in a 3x3 grid,
#and describes the true and predicted classes for each image.
def plot_images(images, true_class, predicted_class = None):
    assert len(images) == len(true_class) == 9
    
    #Create area with 3x3 subplots
    fig, axes = plt.subplots(3, 3)
    #Reserve 3/10 length units of the average axis width for the blank-space between the subplots
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3) 
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap = 'binary') #Plot the image
        
        if predicted_class is None : #Show true and predicted classes
            x_label = "True : {0}".format(true_class[i])
            
        else :
            x_label = "True : {0}, Predicted : {1}".format(true_class[i], predicted_class[i])
            
            
        ax.set_xlabel(x_label) #Show classes as the x-label
        #Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()



#Now let's plot some images
images = data.test.images[0:9]
data.test.cls = np.argmax(data.test.labels, axis = 1)
true_class = data.test.cls[0:9]
plot_images(images = images, true_class = true_class)



#Define initialization of the variables that need to be updated while training.
#This definition is just a broad set of instructions given to the tensorflow graph.
#There is no actual initialization taking place here, i.e. the neurons aren't ready to fire yet.
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape = [length]))


def new_convolutional_layer(input, channels, filter_size, num_filters, use_pooling = True):
    
    #Arguments of this function are
    #1. The output from the previous layer
    #2. Number of channels in the previous layer
    #3. Filter size (length x breadth)
    #4. Number of filters to be used
    #5. Permission to use max-pooling
    
    #Set the shape of the weights in the filter
    shape = [filter_size, filter_size, channels, num_filters]
    
    #Create new weights
    weights = new_weights(shape = shape)
    
    #Create new biases; set one bias value for each filter
    biases = new_biases(length = num_filters)
    
    #Create the tensorflow operation for convolution
    layer = tf.nn.conv2d(input = input, filter = weights, strides = [1, 1, 1, 1], padding = 'SAME')
    #The stride vector [1, 1, 1, 1] should always have its first and last elements as 1
    #since the first element represents striding through the images
    #and the last element represents striding through the input-channels.
    #The elements in between represent the lengthwise and breadthwise strides while convolving
    #through a given image.
    #For example, a [1,4,2,1] stride vector means that for a given image, lengthwise stride is 4
    #and breadthwise stride is 2.
    #Padding is set to 'SAME' which implies that the input images are padded with sufficient amount of zeroes
    #such that the size of the output image is the same
    
    layer += biases #Adds a bias to each filter channel
    
    if use_pooling == True :
        layer = tf.nn.max_pool(value = layer, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    #Creates a pooling layer with 2x2 pooling-filter size and a stride of 2 units along the length and breadth
    
    #ReLU activation function is used
    layer = tf.nn.relu(layer)
    
    return layer, weights



#The convolution creates a 4D tensor. In order to connect the convolutional layer to a fully connected
#layer, we will need to reshape the 4D tensor to a 2D tensor
def flatten_layer(layer):
    
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    #The number of features is given by
    #image length * image breadth * number of channels
    
    flat_layer = tf.reshape(layer, [-1, num_features])
    #Reshaped the layer to [number of images, number of features]
    #The size of the second dimension is set to the number of features.
    #The size of the first dimension is set to -1 which means
    #that the first dimension of the 'flat_layer tensor' is a flattened 1-D tensor whose
    #elements are from the 'layer' tensor.
    
    return flat_layer, num_features


#Now let's set the rules for building a fully connected layer.
def new_fully_connected_layer(input, num_inputs, num_outputs, use_relu = True):
    #Arguments include:
    #1. The previous layer
    #2. Number of inputs from the previous layer
    #3. Number of outputs
    #4. Permission to use ReLU
    
    weights = new_weights(shape = [num_inputs, num_outputs])
    biases = new_biases(length = num_outputs)
    
    layer = tf.matmul(input, weights) + biases
    
    if use_relu == True :
        layer = tf.nn.relu(layer)
        
    return layer



x = tf.placeholder(tf.float32, shape = [None, img_area], name = 'x')
#Placeholder variable for input images, 'None' in the first argument of shape means
#that the x tensor can hold an arbitrary number of images with each image being a vector of length
#img_area.
x_image = tf.reshape(x, [-1, img_size, img_size, channels])
#Encode the image as a 4D tensor

#True labels placeholder
y_true = tf.placeholder(tf.float32, shape = [None, num_classes], name = 'y_true')
y_true_class = tf.argmax(y_true, dimension = 1)


#Let's create the first convolutional layer
conv_layer_1, weights_conv_layer_1 = new_convolutional_layer(input = x_image, channels = channels, filter_size = filter_size_1, num_filters = num_filters_1, use_pooling = True)
conv_layer_1


#The second convolutional layer which takes the first convolutional layer's output as its input.
conv_layer_2, weights_conv_layer_2 = new_convolutional_layer(input = conv_layer_1, channels = num_filters_1, filter_size = filter_size_2, num_filters = num_filters_2, use_pooling = True)
conv_layer_2

#The output of the 2nd convolutional layer is a 4D tensor. In order to use this output to
#feed it to the fully connected layer, we need to flatten it into a 2D tensor.

layer_flat, num_features = flatten_layer(conv_layer_2)
layer_flat, num_features

#First fully connected layer
fc1_layer = new_fully_connected_layer(input = layer_flat, num_inputs = num_features, num_outputs = fc_size, use_relu = True)
fc1_layer

#First fully connected layer
fc2_layer = new_fully_connected_layer(input = fc1_layer, num_inputs = fc_size, num_outputs = num_classes, use_relu = False)
fc2_layer


y_predicted = tf.nn.softmax(fc2_layer)
#Class number is the index having the largest softmax score
y_predicted_class = tf.argmax(y_predicted, dimension = 1)


#Logits implies that the softmax scores are logarithmically scaled before finding the cost.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fc2_layer, labels = y_true)

#Since we need a single scalar value to guide the variables during optimization process
#we average the softmax scores for all classifications in a single epoch.
cost = tf.reduce_mean(cross_entropy)


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-4).minimize(cost)
#The optimizer has been added to the Tensorflow graph
#Optimization process is the stochastic gradient descent with learning rate of 0.0001

#Performance measures
correct_prediction = tf.equal(y_predicted_class, y_true_class)
#This returns a vector of booleans, which states whether the predicted class equals the true class (True)
#or does not equal the true class (False).

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#This converts the booleans to float values. True becomes 1.0 and False becomes 0.0
#The average value of the correct_prediction vector then becomes a measure of the accuracy of the classifier

#Now that the tensorflow graph has been created, we will create the tensorflow session to execute the graph.
session = tf.Session()

#Initialize the weights and the biases
session.run(tf.global_variables_initializer())

#Set batch-size
training_batch_size = 45


iterations = 0         #Counter for the total number of iterations

def optimize(num_epochs):
    
    global iterations  #Update the global counter rather than a local copy
    start_time = time.time() #Start the counter
    
    for i in range(iterations, iterations + num_epochs) :
        #Fetch a batch of training examples.
        #x_batch holds a batch of images and y_true_batch are the true labels of those images
        x_batch, y_true_batch = data.train.next_batch(training_batch_size)
        
        #Set up a dictionary linking the batch with its placeholder variable names in the tensorflow graph.
        feed_training_dict = {x : x_batch, y_true : y_true_batch}

        #Run the optimizer using this batch of training data.
        session.run(optimizer, feed_dict = feed_training_dict)
        
        #Print loss in every 1000th iteration
        if i % 1000 == 0 :
            loss = session.run(cost, feed_dict = feed_training_dict)
            print("Loss at epoch ", i, " is ", float(loss))
            
    iterations += num_epochs
    end_time = time.time()
    delta_t = end_time - start_time
    print("Time usage: " + str(timedelta(seconds = int(round(delta_t)))))


def plot_erroneous(predicted_class, correct):
    #This function plots images from the test set that have been misclassified
    #The second argument of the function i.e. 'correct' is a boolean array comparing
    #whether the predicted class is equal to the true class for each image in the test-set
    
    #Negate the 'correct' array
    incorrect = (correct == False)
    
    #Retrieve the incorrectly classified images from the test-set
    images = data.test.images[incorrect]
    #Get the predicted classes of those images
    predicted_class = predicted_class[incorrect]
    #Get the true classes of those images
    true_class = data.test.cls[incorrect]
    
    #Plot the first 9 images
    plot_images(images = images[0:9], true_class = true_class[0:9], predicted_class = predicted_class[0:9])


#Now let's print the confusion matrix.
def print_confusion_matrix(predicted_class):
    #First, let's get the true classifications for the test-set
    true_class = data.test.cls
    
    #Get the confusion matrix
    cm = confusion_matrix(y_true = true_class, y_pred = predicted_class)
    
    #Display it nicely
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    
test_batch_size = 120
def print_test_accuracy(show_erroneous = False, show_confusion_matrix = False):
    #This prints the classification accuracy on the test-set
    #It takes quite a while to compute the classification of all the images in the test-set
    #and hence the results have to be re-used for computational efficiency.
    
    num_test = len(data.test.images)  #Number of images in the test set
    #Allocate an array for the predicted classes, initialize it to zeros
    #Fill the zeros up as the batches are tested for their accuracy
    predicted_class = np.zeros(shape = num_test, dtype = np.int)
    
    i = 0   #Indexing over the batches, the starting index for the next batch is denoted by i
    
    while i < num_test :
        
        j = min(i + test_batch_size, num_test)  #The ending index for the next batch is j
        #Get the images from the test-set between index i and j
        
        images = data.test.images[i:j,:]
        #Get the associated labels
        labels = data.test.labels[i:j,:]
        
        #Create a feed-dictionary with these images and labels
        feed_dict = {x : images, y_true : labels}
        
        #Calculate the predicted class using tensorflow
        predicted_class[i:j] = session.run(y_predicted_class, feed_dict = feed_dict)
        
        #Set the starting index of the next batch to the ending of the current batch
        i = j
        
    true_class = data.test.cls           #Get the true class of the test images
    correct = (true_class == predicted_class)   #Create a boolean array of correctly classified images
    correct_sum = correct.sum()  #When summing over a Boolean array, False means 0 and True means 1
    accuracy = float(correct_sum)/num_test
    
    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(accuracy, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_erroneous:
        print("Example errors:")
        plot_erroneous(predicted_class = predicted_class, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(predicted_class = predicted_class)


optimize(num_epochs = 10000)
print_test_accuracy(show_erroneous = True)

