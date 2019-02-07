from __future__ import print_function
import tensorflow as tf
import numpy as np
import copy
from network.PFNNparameters import *

class PFNNnet(object):
    def __init__(self, sess, input_dim, output_dim, layer_dim, config, name=None, **kwargs):
        self.config = config
        self.sess = sess
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        self.name = name

        self.all_layer_dim = np.concatenate([[self.input_dim], layer_dim, [self.output_dim]], axis=0).astype(int)
        self.if_bias = kwargs.get('if_bias', ([True] * len(layer_dim)) + [False])
        self.activation_fns = kwargs.get('activation', (['tanh'] * len(layer_dim)) + ['None'])
        self.initialize_weight = kwargs.get('init_weight', None)
        self.initialize_bias = kwargs.get('init_bias', None)
        self.trainable = kwargs.get('trainable', True)
        self.reusable = kwargs.get('reusable', False)
        self.nslices = kwargs.get('slices', 4)

        if len(self.if_bias) == 1:
            self.if_bias *= len(layer_dim) + 1
        if len(self.activation_fns) == 1:
            self.activation_fns *= len(layer_dim) + 1

        act_funcs_dict = {'tanh': tf.nn.tanh, 'relu': tf.nn.relu, 'leaky_relu':tf.nn.leaky_relu, 'elu': tf.nn.elu,'None': lambda x: x}
        self.activation_fns_call = [act_funcs_dict[_] for _ in self.activation_fns]

        #with tf.variable_scope(self.name):
        self.input = kwargs.get('input_tf', tf.placeholder(tf.float32, [None, self.input_dim], name='input'))

        assert (self.input.get_shape()[-1].value == input_dim)
        assert (len(self.activation_fns_call) == len(self.layer_dim) + 1)
        assert (len(self.if_bias) == len(self.layer_dim) + 1)

        self.output = self.build(self.input, self.name)

    def build(self, input_tf, name):
        """parameter of nn"""
        rng = np.random.RandomState(23456)
        nslices = self.nslices  # number of control points in phase function
        phase = input_tf[:, -1]  # phase
        state = input_tf[:,:-1] # state

        self.all_layer_dim[0] = self.all_layer_dim[0] - 1 # remove phase out of input

        net = tf.expand_dims(state, -1) #input layer
        # print(net)
        layers = []

        for i, (dim_1, dim_2) in enumerate(zip(self.all_layer_dim[:-1], self.all_layer_dim[1:])):
            #net = tf.nn.dropout(net, keep_prob=keep_prob)
            layer = PFNNParameter((nslices, dim_2, dim_1), rng, phase, self.config, 'wb_%i' % i)
            layers.append(layer)
            # b = layers[i].bias
            b = tf.expand_dims(layers[i].bias, -1)
            # print(b)
            w = layers[i].weight
            # print(w)
            net = tf.matmul(w, net) + b
            net = self.activation_fns_call[i](net)

        net = tf.squeeze(net, -1)
        # print(net)
        # """parameter of nn"""
        # rng = np.random.RandomState(23456)
        # nslices = 4  # number of control points in phase function
        # X_nn = input_tf
        # Xdim = self.input_dim
        # Ydim = self.output_dim
        # phase = X_nn[:, -1]  # phase
        # P0 = PFNNParameter((nslices, 128, Xdim - 1), rng, phase, 'wb0')
        # P1 = PFNNParameter((nslices, 128, 128), rng, phase, 'wb1')
        # P2 = PFNNParameter((nslices, Ydim, 128), rng, phase, 'wb2')
        #
        # # keep_prob = tf.placeholder(tf.float32)  # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
        #
        # H0 = X_nn[:, :-1]
        # H0 = tf.expand_dims(H0, -1)
        # # H0 = tf.nn.dropout(H0, keep_prob=keep_prob)
        #
        # b0 = tf.expand_dims(P0.bias, -1)
        # H1 = tf.matmul(P0.weight, H0) + b0
        # H1 = tf.nn.relu(H1)
        # # H1 = tf.nn.dropout(H1, keep_prob=keep_prob)
        #
        # b1 = tf.expand_dims(P1.bias, -1)
        # H2 = tf.matmul(P1.weight, H1) + b1
        # H2 = tf.nn.relu(H2)
        # # H2 = tf.nn.dropout(H2, keep_prob=keep_prob)
        #
        # b2 = tf.expand_dims(P2.bias, -1)
        # H3 = tf.matmul(P2.weight, H2) + b2
        # H3 = tf.squeeze(H3, -1)
        #
        # net = H3

        return net

    def getInput(self, data, index_joint):
        index_joint = copy.deepcopy(index_joint)#copy list to prevent overwriting
        gating_input = data[..., index_joint[0]:index_joint[0] + 1]
        index_joint.remove(index_joint[0])
        for i in index_joint:
            gating_input = tf.concat([gating_input, data[..., i:i + 1]], axis=-1)
        return gating_input