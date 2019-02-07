import tensorflow as tf
import numpy as np

class grad_inverter:
    def __init__(self, action_bounds, sess): #[lower, upper]
        self.sess = sess
        self.action_size = len(action_bounds[0])
        self.action_bounds = action_bounds

        self.action_input = tf.placeholder(tf.float32, [None, self.action_size])
        self.pmax = tf.constant(action_bounds[1], dtype=tf.float32)
        self.pmin = tf.constant(action_bounds[0], dtype=tf.float32)
        self.prange = tf.constant([upper - lower for lower, upper in zip(action_bounds[0], action_bounds[1])], dtype=tf.float32)
        self.pdiff_max = tf.div(-self.action_input + self.pmax, self.prange)
        self.pdiff_min = tf.div(self.action_input - self.pmin, self.prange)
        self.zeros_act_grad_filter = tf.zeros([self.action_size])
        self.act_grad = tf.placeholder(tf.float32, [None, self.action_size])
        # self.grad_inverter = tf.select(tf.greater(self.act_grad, self.zeros_act_grad_filter), tf.mul(self.act_grad, self.pdiff_max), tf.mul(self.act_grad, self.pdiff_min))
        self.grad_inverter = tf.where(tf.greater(self.act_grad, self.zeros_act_grad_filter),
                                      tf.multiply(self.act_grad, self.pdiff_max),
                                      tf.multiply(self.act_grad, self.pdiff_min))

    def invert(self, grad, action):
        return self.sess.run(self.grad_inverter, feed_dict={self.action_input: action, self.act_grad: grad})

    def invert_op(self, grad, action):

        pmax = tf.constant(self.action_bounds[1], dtype=tf.float32)
        pmin = tf.constant(self.action_bounds[0], dtype=tf.float32)
        prange = tf.constant([upper - lower for lower, upper in zip(self.action_bounds[0], self.action_bounds[1])], dtype=tf.float32)
        pdiff_max = tf.div(-action + pmax, prange)
        pdiff_min = tf.div(action - pmin, prange)
        zeros_act_grad_filter = tf.zeros([self.action_size])
        # self.grad_inverter = tf.select(tf.greater(self.act_grad, self.zeros_act_grad_filter), tf.mul(self.act_grad, self.pdiff_max), tf.mul(self.act_grad, self.pdiff_min))
        grad_inverter_op = tf.where(tf.greater(grad, zeros_act_grad_filter),
                                      tf.multiply(grad, pdiff_max),
                                      tf.multiply(grad, pdiff_min))
        return grad_inverter_op

    # def invert_loss(self, action):
    #     pmax = tf.constant(self.action_bounds[1], dtype=tf.float32)
    #     pmin = tf.constant(self.action_bounds[0], dtype=tf.float32)
    #     zeros = tf.zeros([self.action_size])
    #     # self.grad_inverter = tf.select(tf.greater(self.act_grad, self.zeros_act_grad_filter), tf.mul(self.act_grad, self.pdiff_max), tf.mul(self.act_grad, self.pdiff_min))
    #     upper_bound = tf.where(tf.greater(action, pmax),
    #                                   tf.subtract(action, pmax),
    #                                   tf.multiply(action, zeros))
    #     lower_bound = tf.where(tf.less(action, pmin),
    #                                   tf.subtract(pmin, action),
    #                                   tf.multiply(action, zeros))
    #     return tf.reduce_mean(upper_bound+lower_bound)

    def bound_loss(self, action):
        # penalty for violating bounds
        violation_min = tf.minimum(action - self.action_bounds[0], 0)#lower bound
        violation_max = tf.maximum(action - self.action_bounds[1], 0)#upper bound
        violation = tf.reduce_sum(tf.square(violation_min), axis=-1) + tf.reduce_sum(tf.square(violation_max), axis=-1) #
        loss = 0.5 * tf.reduce_mean(violation)
        return loss

class clip_action:
    def __init__(self, action_bounds, sess):
        self.sess = sess
        self.action_size = len(action_bounds[0])
        self.action_input = tf.placeholder(tf.float32, [None, self.action_size])
        self.clip_action = tf.clip_by_value(self.action_input, action_bounds[0], action_bounds[1])

    def clip(self, action):
        if action.ndim < 2:  # no batch
            action = action[np.newaxis, :]
            return self.sess.run(self.clip_action, feed_dict={self.action_input: action})[0]
        else:
            action_batch = action
            return self.sess.run(self.clip_action, feed_dict={self.action_input: action_batch})

class rerange_action:
    def __init__(self, network_bounds, action_bounds, sess):
        self.sess = sess
        self.action_size = len(action_bounds[0])
        self.network_bounds = network_bounds
        self.action_bounds = action_bounds
        network_range = (self.network_bounds[1][:] - self.network_bounds[0][:])#upper bound - lower bound
        action_range = (self.action_bounds[1][:] - self.action_bounds[0][:])
        self.action_input = tf.placeholder(tf.float32, [None, self.action_size])
        self.rerange_action = tf.add(tf.subtract(self.action_input,self.network_bounds[0])/network_range*action_range, self.action_bounds[0])

    def rerange(self, action):
        if action.ndim < 2:  # no batch
            action = action[np.newaxis, :]
            return self.sess.run(self.rerange_action, feed_dict={self.action_input: action})[0]
        else:
            action_batch = action

            return self.sess.run(self.rerange_action, feed_dict={self.action_input: action_batch})
    def rerange_op(self, action):
        network_range = (self.network_bounds[1][:] - self.network_bounds[0][:])#upper bound - lower bound
        action_range = (self.action_bounds[1][:] - self.action_bounds[0][:])
        return tf.add(tf.subtract(action,self.network_bounds[0])/network_range*action_range, self.action_bounds[0])