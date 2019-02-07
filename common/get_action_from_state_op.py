import tensorflow as tf
import numpy as np

class get_action_from_state_op:
    def __init__(self, state_dim, action_dim):
        self.init_mat(state_dim, action_dim)

    def init_mat(self, state_dim, action_dim):
        if action_dim == 4:
            mat = np.zeros((state_dim, action_dim))
            mat[11][0] = 1 #torso pitch
            mat[15][1] = 1 #right hip pitch
            mat[17][2] = 1 #right knee pitch
            mat[19][3] = 1 #right ankle pitch

            init_mat = tf.constant(value=mat, dtype=tf.float32)
            self.mat = tf.get_variable(name='mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[action_dim, action_dim]
        elif action_dim == 11:
            mat = np.zeros((state_dim, action_dim))
            mat[11][0] = 1  # torso pitch

            mat[13][1] = 1  # rightHipRoll = leftHipRoll
            mat[15][2] = 1  # rightHipPitch = leftHipPitch
            mat[17][3] = 1  # rightKneePitch = leftKneePitch
            mat[19][4] = 1  # rightAnklePitch = leftAnklePitch
            mat[21][5] = 1  # rightAnkleRoll = leftAnkleRoll

            mat[26][6] = 1  # rightHipRoll = leftHipRoll
            mat[28][7] = 1  # rightHipPitch = leftHipPitch
            mat[30][8] = 1  # rightKneePitch = leftKneePitch
            mat[32][9] = 1  # rightAnklePitch = leftAnklePitch
            mat[34][10] = 1  # rightAnkleRoll = leftAnkleRoll
            init_mat = tf.constant(value=mat, dtype=tf.float32)
            self.mat = tf.get_variable(name='mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[action_dim, action_dim]
        elif action_dim == 12:
            mat = np.zeros((state_dim, action_dim))
            mat[11][0] = 1  # torso pitch
            mat[1][1] = 1  # torso roll

            mat[13][2] = 1  # rightHipRoll = leftHipRoll
            mat[15][3] = 1  # rightHipPitch = leftHipPitch
            mat[17][4] = 1  # rightKneePitch = leftKneePitch
            mat[19][5] = 1  # rightAnklePitch = leftAnklePitch
            mat[21][6] = 1  # rightAnkleRoll = leftAnkleRoll

            mat[26][7] = 1  # rightHipRoll = leftHipRoll
            mat[28][8] = 1  # rightHipPitch = leftHipPitch
            mat[30][9] = 1  # rightKneePitch = leftKneePitch
            mat[32][10] = 1  # rightAnklePitch = leftAnklePitch
            mat[34][11] = 1  # rightAnkleRoll = leftAnkleRoll
            init_mat = tf.constant(value=mat, dtype=tf.float32)
            self.mat = tf.get_variable(name='mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[action_dim, action_dim]

        elif action_dim == 13:
            mat = np.zeros((state_dim, action_dim))
            mat[11][0] = 1  # torso pitch

            mat[7][1] = 1  # rightHipYaw = leftHipYaw
            mat[13][2] = 1  # rightHipRoll = leftHipRoll
            mat[15][3] = 1  # rightHipPitch = leftHipPitch
            mat[17][4] = 1  # rightKneePitch = leftKneePitch
            mat[19][5] = 1  # rightAnklePitch = leftAnklePitch
            mat[21][6] = 1  # rightAnkleRoll = leftAnkleRoll

            mat[7][7] = 1  # rightHipYaw = leftHipYaw
            mat[26][8] = 1  # rightHipRoll = leftHipRoll
            mat[28][9] = 1  # rightHipPitch = leftHipPitch
            mat[30][10] = 1  # rightKneePitch = leftKneePitch
            mat[32][11] = 1  # rightAnklePitch = leftAnklePitch
            mat[34][12] = 1  # rightAnkleRoll = leftAnkleRoll
            init_mat = tf.constant(value=mat, dtype=tf.float32)
            self.mat = tf.get_variable(name='mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[action_dim, action_dim]

        elif action_dim == 15:
            mat = np.zeros((state_dim, action_dim))
            mat[11][0] = 1  # torso yaw
            mat[1][1] = 1  # torso pitch
            mat[2][2] = 1  # torso roll

            mat[9][3] = 1  # rightHipYaw = leftHipYaw
            mat[13][4] = 1  # rightHipRoll = leftHipRoll
            mat[15][5] = 1  # rightHipPitch = leftHipPitch
            mat[17][6] = 1  # rightKneePitch = leftKneePitch
            mat[19][7] = 1  # rightAnklePitch = leftAnklePitch
            mat[21][8] = 1  # rightAnkleRoll = leftAnkleRoll

            mat[3][9] = 1  # rightHipYaw = leftHipYaw
            mat[26][10] = 1  # rightHipRoll = leftHipRoll
            mat[28][11] = 1  # rightHipPitch = leftHipPitch
            mat[30][12] = 1  # rightKneePitch = leftKneePitch
            mat[32][13] = 1  # rightAnklePitch = leftAnklePitch
            mat[34][14] = 1  # rightAnkleRoll = leftAnkleRoll
            init_mat = tf.constant(value=mat, dtype=tf.float32)
            self.mat = tf.get_variable(name='mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[action_dim, action_dim]

    def get_action_op(self, state):
        action = tf.matmul(state, self.mat)
        return action