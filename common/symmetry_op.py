import tensorflow as tf
import numpy as np

class Symmetric_op:
    def __init__(self, state_dim, action_dim):
        self.init_action_mat(action_dim)
        self.init_state_mat(state_dim)

    def init_action_mat(self, action_dim):
        if action_dim == 4:
            mat = np.zeros((action_dim, action_dim))
            mat[0][0] = 1
            mat[1][1] = 1
            mat[2][2] = 1
            mat[3][3] = 1

            init_mat = tf.constant(value=mat, dtype=tf.float32)
            self.sym_mat_action = tf.get_variable(name='action_mirror_mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[action_dim, action_dim]
        elif action_dim == 11:
            mat = np.zeros((action_dim, action_dim))
            mat[0][0] = 1  # torso pitch

            mat[6][1] = -1  # rightHipRoll = leftHipRoll
            mat[7][2] = 1  # rightHipPitch = leftHipPitch
            mat[8][3] = 1  # rightKneePitch = leftKneePitch
            mat[9][4] = 1  # rightAnklePitch = leftAnklePitch
            mat[10][5] = -1  # rightAnkleRoll = leftAnkleRoll

            mat[1][6] = -1  # rightHipRoll = leftHipRoll
            mat[2][7] = 1  # rightHipPitch = leftHipPitch
            mat[3][8] = 1  # rightKneePitch = leftKneePitch
            mat[4][9] = 1  # rightAnklePitch = leftAnklePitch
            mat[5][10] = -1  # rightAnkleRoll = leftAnkleRoll
            init_mat = tf.constant(value=mat, dtype=tf.float32)
            self.sym_mat_action = tf.get_variable(name='action_mirror_mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[action_dim, action_dim]
        elif action_dim == 12:
            mat = np.zeros((action_dim, action_dim))
            mat[0][0] = 1  # torso pitch
            mat[1][1] = -1  # torso roll

            mat[7][2] = -1  # rightHipRoll = leftHipRoll
            mat[8][3] = 1  # rightHipPitch = leftHipPitch
            mat[9][4] = 1  # rightKneePitch = leftKneePitch
            mat[10][5] = 1  # rightAnklePitch = leftAnklePitch
            mat[11][6] = -1  # rightAnkleRoll = leftAnkleRoll

            mat[2][7] = -1  # rightHipRoll = leftHipRoll
            mat[3][8] = 1  # rightHipPitch = leftHipPitch
            mat[4][9] = 1  # rightKneePitch = leftKneePitch
            mat[5][10] = 1  # rightAnklePitch = leftAnklePitch
            mat[6][11] = -1  # rightAnkleRoll = leftAnkleRoll
            init_mat = tf.constant(value=mat, dtype=tf.float32)
            self.sym_mat_action = tf.get_variable(name='action_mirror_mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[action_dim, action_dim]

        elif action_dim == 13:
            mat = np.zeros((action_dim, action_dim))
            mat[0][0] = 1  # torso pitch

            mat[7][1] = -1  # rightHipYaw = leftHipYaw
            mat[8][2] = -1  # rightHipRoll = leftHipRoll
            mat[9][3] = 1  # rightHipPitch = leftHipPitch
            mat[10][4] = 1  # rightKneePitch = leftKneePitch
            mat[11][5] = 1  # rightAnklePitch = leftAnklePitch
            mat[12][6] = -1  # rightAnkleRoll = leftAnkleRoll

            mat[7][1] = -1  # rightHipYaw = leftHipYaw
            mat[8][2] = -1  # rightHipRoll = leftHipRoll
            mat[9][3] = 1  # rightHipPitch = leftHipPitch
            mat[10][4] = 1  # rightKneePitch = leftKneePitch
            mat[11][5] = 1  # rightAnklePitch = leftAnklePitch
            mat[12][6] = -1  # rightAnkleRoll = leftAnkleRoll
            init_mat = tf.constant(value=mat, dtype=tf.float32)
            self.sym_mat_action = tf.get_variable(name='action_mirror_mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[action_dim, action_dim]

        elif action_dim == 15:
            mat = np.zeros((action_dim, action_dim))
            mat[0][0] = -1  # torso yaw
            mat[1][1] = 1  # torso pitch
            mat[2][2] = -1  # torso roll

            mat[9][3] = -1  # rightHipYaw = leftHipYaw
            mat[10][4] = -1  # rightHipRoll = leftHipRoll
            mat[11][5] = 1  # rightHipPitch = leftHipPitch
            mat[12][6] = 1  # rightKneePitch = leftKneePitch
            mat[13][7] = 1  # rightAnklePitch = leftAnklePitch
            mat[14][8] = -1  # rightAnkleRoll = leftAnkleRoll

            mat[3][9] = -1  # rightHipYaw = leftHipYaw
            mat[4][10] = -1  # rightHipRoll = leftHipRoll
            mat[5][11] = 1  # rightHipPitch = leftHipPitch
            mat[6][12] = 1  # rightKneePitch = leftKneePitch
            mat[7][13] = 1  # rightAnklePitch = leftAnklePitch
            mat[8][14] = -1  # rightAnkleRoll = leftAnkleRoll
            init_mat = tf.constant(value=mat, dtype=tf.float32)
            self.sym_mat_action = tf.get_variable(name='action_mirror_mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[action_dim, action_dim]

        else:
            mat = np.eye((action_dim))
            init_mat = tf.constant(value=mat, dtype=tf.float32)
            self.sym_mat_action = tf.get_variable(name='action_mirror_mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[action_dim, action_dim]

    def init_state_mat(self, state_dim):
        mat = np.zeros((state_dim, state_dim))

        mat[0][0] = 1  # pelvis_x_dot
        mat[1][1] = -1  # pelvis_y_dot
        mat[2][2] = 1  # pelvis_z_dot
        # pelvis
        mat[3][3] = -1  # pelvis_roll
        mat[4][4] = 1  # pelvis_pitch

        mat[5][5] = -1  # pelvis_pitch_dot
        mat[6][6] = 1  # pelvis_roll_dot
        mat[7][7] = -1  # pelvis_yaw_dot

        # chest_link_dis_yaw = chest_link_dis
        mat[8][8] = 1  # chest_com_position_x - pelvis_com_position_x
        mat[9][9] = -1  # chest_com_position_y - pelvis_com_position_y
        mat[10][10] = 1  # chest_com_position_z - pelvis_com_position_z

        mat[11][11] = 1  # torso pitch angle
        mat[12][12] = 1  # torso pitch velocity

        mat[26][13] = 1  # right hip pitch position = right hip
        mat[27][14] = 1  # velocity

        mat[28][15] = -1  # hip roll position
        mat[29][16] = -1  # velocity

        mat[30][17] = 1  # knee pitch position
        mat[31][18] = 1  # velocity

        mat[32][19] = 1  # ankle pitch
        mat[33][20] = 1

        mat[34][21] = -1  # ankle roll
        mat[35][22] = -1

        # right_foot_link_dis_yaw = left_foot_link_dis
        mat[36][23] = 1  # foot_com_position_x - pelvis_com_position_x
        mat[37][24] = -1  # foot_com_position_y - pelvis_com_position_y
        mat[38][25] = 1  # foot_com_position_z - pelvis_com_position_z

        mat[13][26] = 1  # left hip pitch position
        mat[14][27] = 1  # velocity

        mat[15][28] = -1  # hip roll position
        mat[16][29] = -1  # velocity

        mat[17][30] = 1  # knee pitch position
        mat[18][31] = 1  # velocity

        mat[19][32] = 1  # ankle pitch
        mat[20][33] = 1

        mat[21][34] = -1  # ankle roll
        mat[22][35] = -1

        # left_foot_link_dis_yaw = left_foot_link_dis
        mat[23][36] = 1  # foot_com_position_x - pelvis_com_position_x
        mat[24][37] = -1  # foot_com_position_y - pelvis_com_position_y
        mat[25][38] = 1  # foot_com_position_z - pelvis_com_position_z

        # COM_dis_yaw = COM_dis
        mat[39][39] = 1
        mat[40][40] = -1
        mat[41][41] = 1

        # COM_vel_yaw = COM_vel
        mat[42][42] = 1
        mat[43][43] = -1
        mat[44][44] = 1

        mat[46][45] = 1  # right contact force = left_contact_force
        mat[45][46] = 1

        init_mat = tf.constant(value=mat, dtype=tf.float32)
        self.sym_mat_state = tf.get_variable(name='state_mirror_mat', initializer=init_mat, dtype=tf.float32, trainable=False)#, shape=[state_dim, state_dim]

    def mirror_action_op(self, action):
        mirror_action = tf.matmul(action, self.sym_mat_action)
        return mirror_action


    def mirror_state_op(self, state):
        mirror_state = tf.matmul(state, self.sym_mat_state)
        return mirror_state