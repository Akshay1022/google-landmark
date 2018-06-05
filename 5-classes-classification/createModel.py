import loadData
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


b_size = 32

#Prepare input data
classes = ['2061', '6051', '6599', '9633', '9779']
class_size = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.15
img_size = 256
no_of_channel = 3
training_set_path=r'C:\Users\Arun\Desktop\MachineLearning\Project\CNN\5-classes-classification\5-class-dataset\training-data'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = loadData.read_train_data(training_set_path, img_size, classes, validation_size=validation_size)


print("Completed reading input data. Going to print a snippet of it")
print("Size of Training-set:\t\t{}".format(len(data.train.labels)))
print("Size of Validation-set:\t{}".format(len(data.valid.labels)))



session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,no_of_channel], name='x')

# labels
y_true = tf.placeholder(tf.float32, shape=[None, class_size], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



##Network graph params
convolution_filter_size_1 = 3 
no_of_convolution_filter_1 = 32

convolution_filter_size_2 = 3
no_of_convolution_filter_2 = 32

convolution_filter_size_3 = 3
no_of_convolution_filter_3 = 64
    
fully_connected_layer_size = 128

def assign_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def make_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,
               no_of_channel, 
               filter_size,        
               num_filters):  
    
    ## the weights that will be trained using assign_weights function.
    weights = assign_weights(shape=[filter_size, filter_size, no_of_channel, num_filters])
    
    biases = make_biases(num_filters)

    ## convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer

    

def flatten_layer(layer):
    
    layer_shape = layer.get_shape()

    ## Features are img_height * img_width* no_of_channel. But we will calculate it.
    num_features = layer_shape[1:4].num_elements()

    ##  reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def make_fc_layer(input,          
             inputsize,    
             outputsize,
             use_relu=True):
    
    # define the weights and biases of training data
    weights = assign_weights(shape=[inputsize, outputsize])
    biases = make_biases(outputsize)

    # We are going to use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


convolution_layer_1 = create_convolutional_layer(input=x,
               no_of_channel=no_of_channel,
               filter_size=convolution_filter_size_1,
               num_filters=no_of_convolution_filter_1)
convolution_layer_2 = create_convolutional_layer(input=convolution_layer_1,
               no_of_channel=no_of_convolution_filter_1,
               filter_size=convolution_filter_size_2,
               num_filters=no_of_convolution_filter_2)

convolution_layer_3= create_convolutional_layer(input=convolution_layer_2,
               no_of_channel=no_of_convolution_filter_2,
               filter_size=convolution_filter_size_3,
               num_filters=no_of_convolution_filter_3)
          
layer_flat = flatten_layer(convolution_layer_3)

fully_connected_layer_1 = make_fc_layer(input=layer_flat,
                     inputsize=layer_flat.get_shape()[1:4].num_elements(),
                     outputsize=fully_connected_layer_size,
                     use_relu=True)

fully_connected_layer_2 = make_fc_layer(input=fully_connected_layer_1,
                     inputsize=fully_connected_layer_size,
                     outputsize=class_size,
                     use_relu=False) 

y_pred = tf.nn.softmax(fully_connected_layer_2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fully_connected_layer_2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 


def print_accuracy(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Accuracy: {0:>6.1%}, Validation Accuracy: {1:>6.1%},  Validation Loss: {2:.3f}"
    print(msg.format( acc, val_acc, val_loss))

total_no_of_iter = 0

saver = tf.train.Saver()
def train(num_iteration):
    global total_no_of_iter
    
    for i in range(total_no_of_iter,
                   total_no_of_iter + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(b_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(b_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/b_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/b_size))    
            
            print_accuracy(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session,r'C:\Users\Arun\Desktop\MachineLearning\Project\CNN\5-classes-classification\model') 


    total_no_of_iter += num_iteration

train(num_iteration=3000)