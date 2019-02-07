"""
Class of Gating NN
"""
import numpy as np
import tensorflow as tf
import copy

class Gating(object):
    def __init__(self, rng, input_tf, input_dim, output_dim, layer_dim, config, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config

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

        """rng"""
        self.initialRNG = rng

        """input"""
        self.input = input_tf

        # """dropout"""
        # self.keep_prob = keep_prob

        # """size"""
        # self.input_size = input_size
        # self.output_size = output_size
        # self.hidden_size = hidden_size
        #
        # """parameters"""
        #
        # self.w0 = tf.get_variable(
        #     initializer=self.initial_weight([hidden_size, input_size]), trainable=self.trainable, name='wc0_w'
        # )
        # self.w1 = tf.get_variable(
        #     initializer=self.initial_weight([hidden_size, hidden_size]), trainable=self.trainable, name='wc1_w'
        # )
        # self.w2 = tf.get_variable(
        #     initializer=self.initial_weight([output_size, hidden_size]), trainable=self.trainable, name='wc2_w'
        # )
        # self.b0 = tf.get_variable(
        #     initializer=self.initial_bias([hidden_size, 1]), trainable=self.trainable, name='wc0_b'
        # )
        # self.b1 = tf.get_variable(
        #     initializer=self.initial_bias([hidden_size, 1]), trainable=self.trainable, name='wc1_b'
        # )
        # self.b2 = tf.get_variable(
        #     initializer=self.initial_bias([output_size, 1]), trainable=self.trainable, name='wc2_b'
        # )

        """"output blending coefficients"""
        self.BC = self.fp()

    """initialize parameters """

    def initial_weight(self, shape):
        rng = self.initialRNG
        weight_bound = np.sqrt(6. / np.sum(shape[-2:]))
        """Understanding difficulty of training deep feedforward neural networks"""
        # weight_bound = np.sqrt(6./ (shape[-2]+shape[-1])) #Xavier initialization
        """Delving deep into rectifiers: Surpassing human-level performance on ImageNet Classification"""
        # alpha_bound = np.sqrt(6. / shape[-1])
        weight = np.asarray(
            rng.uniform(low=-weight_bound, high=weight_bound, size=shape),
            dtype=np.float32)
        return tf.convert_to_tensor(weight, dtype=tf.float32)

    def initial_bias(self, shape):
        return tf.zeros(shape, tf.float32)

    """forward propogation"""

    def fp(self):
        """parameter of nn"""

        net = self.input #input layer
        # print(net)
        weights = []
        biases = []
        print(self.all_layer_dim)
        for i, (dim_1, dim_2) in enumerate(zip(self.all_layer_dim[:-1], self.all_layer_dim[1:])):
            #net = tf.nn.dropout(net, keep_prob=keep_prob)
            w = tf.get_variable(
            initializer=self.initial_weight([dim_2, dim_1]), trainable=self.trainable, name='wc%i_w' % i
            )
            weights.append(w)
            b = tf.get_variable(
                initializer=self.initial_bias([dim_2, 1]), trainable=self.trainable, name='wc%i_b' % i
            )
            biases.append(b)
            print(w)
            print(b)
            net = tf.matmul(w, net) + b
            net = self.activation_fns_call[i](net)

        net = tf.nn.softmax(net, dim=0)  # out*batch
        return net
        # # H0 = tf.nn.dropout(self.input, keep_prob=self.keep_prob)  # input*batch
        #
        # H1 = tf.matmul(self.w0, H0) + self.b0  # hidden*input mul input*batch
        # H1 = tf.nn.elu(H1)
        # # H1 = tf.nn.dropout(H1, keep_prob=self.keep_prob)
        #
        # H2 = tf.matmul(self.w1, H1) + self.b1
        # H2 = tf.nn.elu(H2)
        # # H2 = tf.nn.dropout(H2, keep_prob=self.keep_prob)
        #
        # H3 = tf.matmul(self.w2, H2) + self.b2  # out*hidden   mul hidden*batch
        # H3 = tf.nn.softmax(H3, dim=0)  # out*batch
        # return H3


# --------------------------------------get the input for the Gating network---------------------------------
"""global parameters"""
num_trajPoints = 12  # number of trajectory points
num_trajUnit_noSpeed = 6  # number of trajectory units: Position X,Z; Direction X,Z; Velocity X,Z;
num_trajUnit_speed = 7  # number of trajectory units: Position X,Z; Direction X,Z; Velocity X,Z; Speed
num_jointUnit = 12  # number of joint units: PositionXYZ Rotation VelocityXYZ


# get the velocity of joints, desired velocity and style
def getInput(data, index_joint):
    index_joint = copy.deepcopy(index_joint)
    gating_input = data[..., index_joint[0]:index_joint[0] + 1]
    index_joint.remove(index_joint[0])
    for i in index_joint:
        gating_input = tf.concat([gating_input, data[..., i:i + 1]], axis=-1)
    return gating_input


def save_GT(weight, bias, filename):
    for i in range(len(weight)):
        a = weight[i]
        b = bias[i]
        a.tofile(filename + '/wc%0i_w.bin' % i)
        b.tofile(filename + '/wc%0i_b.bin' % i)
