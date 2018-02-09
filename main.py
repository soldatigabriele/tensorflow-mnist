import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request


import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from mnist import model

import os
# from random import shuffle
# from tqdm import tqdm

# import dataset as data

#import the cnn graphs
# import graphs.cnn_2layers as conv2
# import graphs.cnn_6layers as conv6
# import graphs.cnn_8layers as conv8
#import alexnet
# import graphs.alexnet as alexnet
# import graphs.resnext as resnext
# import graphs.inception as inception
# import graphs.googlenet as googlenet
# import graphs.convnet as conv


""" Linear Regression Example """

# from __future__ import absolute_import, division, print_function

# Linear Regression graph
input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.01)
m = tflearn.DNN(regression)
# m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)
m.load('model/test.model')
# m.predict([3.2, 3.3, 3.4])


# print("\nRegression result:")
# print("Y = " + str(m.get_weights(linear.W)) +
       # "*X + " + str(m.get_weights(linear.b)))

# print("\nTest prediction for x = 3.2, 3.3, 3.4:")


def network(img_shape, name, LR):
      
    network = input_data(shape=img_shape, name=name )

    network = conv_2d(network, 32, 2, activation='relu')
    network = max_pool_2d(network,2)

    network = conv_2d(network, 64, 2, activation='relu')
    network = max_pool_2d(network,2)

    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.8)
    # 2 is the number of classes
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    return network





x = tf.placeholder("float", [None, 784])
sess = tf.Session()

# restore trained data
with tf.variable_scope("regression"):
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/regression.ckpt")


with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


# webapp
app = Flask(__name__)


@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    return jsonify(results=[output1, output2])


@app.route('/predict')
def main():
    data = request.args.get('data')    
    # return render_template('index.html')
    # m.load('model/test.model')
    return jsonify( (m.predict([data])).tolist() )
    # return user


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
