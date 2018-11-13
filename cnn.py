# Basic tensorflow 3-layer convolutional neural network for use on fashion-mnist dataset

import numpy as np
# import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# %matplotlib inline

# Loading dataset, reshaping for training

data = input_data.read_data_sets('data/fashion',one_hot=True)

print('Shape of training images: {shape}'.format(shape=data.train.images.shape))
print('Shape of training labels: {shape}'.format(shape=data.train.labels.shape))

print('Shape of test images: {shape}'.format(shape=data.test.images.shape))
print('Shape of test labels: {shape}'.format(shape=data.test.labels.shape))

# Dictionary of class labels
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

X_train = data.train.images.reshape(-1,28,28,1)
X_test = data.test.images.reshape(-1,28,28,1)
Y_train = data.train.labels
Y_test = data.test.labels

# print(X_train.shape)
# print(Y_train.shape)

# # Selecting subset of data
# n_samples = 100
# X_train = X_train[:n_samples,:,:,:]
# X_test = X_test[:n_samples,:,:,:]
# Y_train = Y_train[:n_samples,:]
# Y_test = Y_test[:n_samples,:]
# print(X_train.shape)
# ##

# Set up CNN model

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

input_size = 784
n_classes = 10

x = tf.placeholder('float',[None,28,28,1])
y = tf.placeholder('float',[None,n_classes])

def conv2d(x, W, b, stride=1):
	# Wrapper function for conv layer - perform convolution, add bias and ReLU activate

	x = tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	x = tf.nn.relu(x)

	return x

def maxpool2d(x, k=2):
	# Wrapper function for maxpooling layer - maxpool with a kernel size of k, and stride of k

	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

# Define weight and bias matrices
weights = {
	'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
	'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
	'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
	'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()),
	'out': tf.get_variable('W4', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer())
}

biases = {
	'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
	'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
	'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
	'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
	'out': tf.get_variable('B4', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer())
}

def conv_net(x, weights, biases):
	# Build cnn model. 3 conv layers, 3 maxpool layers, 1 fully connected layer
	# Each input has shape 28x28

	# Convolve with 32 3x3 filters, maxpool down to 14x14x32
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	pool1 = maxpool2d(conv1, k=2)

	# Convolve with 64 3x3 filters, maxpool down to 7x7x64
	conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
	pool2 = maxpool2d(conv2, k=2)

	# Convolve with 128 3x3 filters, maxpool down to 4x4x128
	conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
	pool3 = maxpool2d(conv3, k=2)

	# Fully connected layer
	fc1 = tf.reshape(pool3, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.matmul(fc1, weights['wd1'])
	fc1 = tf.add(fc1, biases['bd1'])

	# Output layer
	out = tf.matmul(fc1, weights['out'])
	out = tf.add(out, biases['out'])

	return out

# Obtain predictions from forward pass, calculate loss, optimize
prediction = conv_net(x, weights, biases)
output = tf.argmax(prediction,1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate model during training
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialise tf variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	training_losses = []
	test_losses = []
	training_accuracy = []
	test_accuracy = []
	summary_writer = tf.summary.FileWriter('./Ouput', sess.graph)

	print('Starting training')
	for i in range(EPOCHS):
		for batch in range(int(len(X_train)/BATCH_SIZE)):
			batch_x = X_train[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(X_train))]
			batch_y = Y_train[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(Y_train))]

			opt = sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
			loss, acc = sess.run([cost,accuracy], feed_dict={x:batch_x, y:batch_y})

		print('Iter {0:}, loss = {1:.6f}, train accuracy = {2:.6f}'.format(i,loss,acc))

		test_loss, test_acc = sess.run([cost,accuracy], feed_dict={x:X_test,y:Y_test})
		print('Test accuracy = {0:.6f}'.format(test_acc))

		training_losses.append(loss)
		test_losses.append(test_loss)
		training_accuracy.append(acc)
		test_accuracy.append(test_acc)

	ex_predictions = sess.run(output, feed_dict={x:X_test[-10:,:,:,:], y:Y_test[-10:,:]})
	print('Example prediction: ', ex_predictions)
	print('Example true value: ', np.argmax(Y_test[-10:,:],1))

	summary_writer.close()
