from __future__ import print_function
import tensorflow as tf
import numpy as np
import copy

import network.Gating as GT
from network.Gating import Gating

import network.ExpertWeights as EW
from network.ExpertWeights import ExpertWeights

class MANNnet(object):
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


        self.num_experts = kwargs.get('num_experts', 4)
        # self.hidden_size = kwargs.get('hidden_size', 512)
        # self.hidden_size_gt = kwargs.get('hidden_size_gt', 32)
        self.index_gating = kwargs.get('index_gating', [10, 15, 19, 23])
        self.index_expert = kwargs.get('index_expert', [10, 15, 19, 23])
        self.batch_size = kwargs.get('batch_size', 32)
        self.rng = kwargs.get('rng', np.random.RandomState(23456))

        self.output = self.build(self.input, self.name)

    def build(self, input_tf, name):
        self.input_size = self.input_dim
        self.output_size = self.output_dim

        #Placeholders
        # self.nn_keep_prob = tf.placeholder(tf.float32, name = 'nn_keep_prob')


        """BUILD gatingNN"""
        # input of gatingNN
        print(self.index_gating)
        self.input_dim_gt = len(self.index_gating)
        print(self.input_dim_gt)
        self.output_dim_gt = self.num_experts
        self.layer_dim_gt = self.config.conf['gating-layer-size']
        self.activation_fns_gt = self.config.conf['gating-activation-fn']
        self.gating_input = tf.transpose(GT.getInput(input_tf, self.index_gating))#create new list to prevent overwriting
        print(self.gating_input)
        print(self.index_gating)
        self.gatingNN = Gating(self.rng, self.gating_input, self.input_dim_gt, self.output_dim_gt, self.layer_dim_gt, self.config, activation_fns = self.activation_fns_gt)

        # bleding coefficients
        self.BC = self.gatingNN.BC
        self.input_dim_ex = len(self.index_expert)
        self.all_layer_dim[0] = self.input_dim_ex#self.all_layer_dim[0]-1#remove last phase from input
        self.expert_input = self.getInput(input_tf, self.index_expert)#remove last pahse input
        #self.expert_input = input_tf[:,:-1]
        net = tf.expand_dims(self.expert_input, -1)  # ?*in -> ?*in*1
        layers = []
        for i, (dim_1, dim_2) in enumerate(zip(self.all_layer_dim[:-1], self.all_layer_dim[1:])):
            #net = tf.nn.dropout(net, keep_prob=keep_prob)

            layer = ExpertWeights(self.rng, (self.num_experts, dim_2, dim_1), 'layer%i'%i)
            layers.append(layer)
            w = layers[i].get_NNweight(self.BC, self.batch_size)
            b = layers[i].get_NNbias(self.BC, self.batch_size)

            net = tf.matmul(w, net) + b
            net = self.activation_fns_call[i](net)

        net = tf.squeeze(net, -1)

        # # initialize experts
        # self.layer0 = ExpertWeights(self.rng, (self.num_experts, self.hidden_size, self.input_size),
        #                             'layer0')  # alpha: 4/8*hid*in, beta: 4/8*hid*1
        # self.layer1 = ExpertWeights(self.rng, (self.num_experts, self.hidden_size, self.hidden_size),
        #                             'layer1')  # alpha: 4/8*hid*hid,beta: 4/8*hid*1
        # self.layer2 = ExpertWeights(self.rng, (self.num_experts, self.output_size, self.hidden_size),
        #                             'layer2')  # alpha: 4/8*out*hid,beta: 4/8*out*1
        #
        # # initialize parameters in main NN
        # """
        # dimension of w: ?* out* in
        # dimension of b: ?* out* 1
        # """
        # w0 = self.layer0.get_NNweight(self.BC, self.batch_size)
        # w1 = self.layer1.get_NNweight(self.BC, self.batch_size)
        # w2 = self.layer2.get_NNweight(self.BC, self.batch_size)
        #
        # b0 = self.layer0.get_NNbias(self.BC, self.batch_size)
        # b1 = self.layer1.get_NNbias(self.BC, self.batch_size)
        # b2 = self.layer2.get_NNbias(self.BC, self.batch_size)
        #
        # # build main NN
        # H0 = tf.expand_dims(self.expert_input, -1)  # ?*in -> ?*in*1
        # # H0 = tf.nn.dropout(H0, keep_prob=self.nn_keep_prob)
        #
        # H1 = tf.matmul(w0, H0) + b0  # ?*out*in mul ?*in*1 + ?*out*1 = ?*out*1
        # H1 = tf.nn.elu(H1)
        # # H1 = tf.nn.dropout(H1, keep_prob=self.nn_keep_prob)
        #
        # H2 = tf.matmul(w1, H1) + b1
        # H2 = tf.nn.elu(H2)
        # # H2 = tf.nn.dropout(H2, keep_prob=self.nn_keep_prob)
        #
        # H3 = tf.matmul(w2, H2) + b2
        # self.H3 = tf.squeeze(H3, -1)  # ?*out*1 ->?*out
        # net = H3

        return net

    def getInput(self,data, index_joint):
        index_joint = copy.deepcopy(index_joint)#copy list to prevent overwriting
        gating_input = data[..., index_joint[0]:index_joint[0] + 1]
        index_joint.remove(index_joint[0])
        for i in index_joint:
            gating_input = tf.concat([gating_input, data[..., i:i + 1]], axis=-1)
        return gating_input