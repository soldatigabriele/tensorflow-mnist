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

max_amount_request = 0
max_months = 0
max_monthly_revenue = 0

# Preprocessing function
def preprocess(leads, columns_to_delete, max_amount_request, max_months, max_monthly_revenue):
    # Sort by descending id and delete columns
    for column_to_delete in sorted(columns_to_delete, reverse=True):
        leads.pop(column_to_delete)
        # [lead.pop(column_to_delete) for lead in leads]
    return leads


""" Linear Regression Example """
data_size = [None, 3]
LR = 0.001

tf.reset_default_graph()

def titanic_network(shape, LR):

    # # Build neural network
    net = tflearn.input_data(shape=shape)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, learning_rate=LR, name='targets')
    return net

net = titanic_network(data_size, LR)

# # Define model
m = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='log')

m.load('model/leads/leads-new.model')


# webapp
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def mnist():
    lista = []
    data = request.form
    for variables in data:
        lista.append(float(request.form[variables]))
    to_ignore=[0,1,2,3,7]
    lista = preprocess(lista, to_ignore, max_amount_request, max_months, max_monthly_revenue)
    result = m.predict([lista])
    res = result[0]
    return res
    # return( jsonify(result[0] ))
    # [1] is silver [0] is unrated
    # return jsonify( result[0] )
    # return( result[0][1] )

@app.route('/predict')
def main():
    data = request.args.get('data')    
    # return render_template('index.html')
    # m.load('model/test.model')
    return jsonify( (m.predict([data])).tolist() )
    # return user


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
