import numpy as np
import pickle
import math
import copy
import numpy
import matplotlib.pyplot as plt
import scipy
from scipy import signal

class Motion():
    def __init__(self, config=None, dsr_gait_freq=0.7):
        self.config = config
        self.joint_list = list([
                "torsoYaw", "torsoPitch", "torsoRoll",
                "rightHipYaw", "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipYaw", "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])
        self.dsr_data_freq = self.config.conf['HLC-frequency'] #desired frequency

        self.dsr_gait_freq = dsr_gait_freq
        self.dsr_length = int(self.dsr_data_freq/self.dsr_gait_freq)#length of array for single gait cycle

        self.motion_data_list = []
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s1ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s1ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s1ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s2ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s2ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s2ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s3ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s3ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s3ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s4ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s4ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s4ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s5ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s5ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s5ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s6ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s6ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s6ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s7ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s7ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot('/home/chuanyu/Valkyrie_TRPO_PPO_PFNN_MANN/ik_files/s7ik3_processed.obj'))

        self.src_joint_angle_list = []
        self.src_joint_vel_list = []
        for motion_data in self.motion_data_list:
            joint_angle = self.getJointAngle(motion_data)
            joint_vel = self.getJointVel(joint_angle, motion_data)
            self.src_joint_angle_list.append(joint_angle)
            self.src_joint_vel_list.append(joint_vel)

        self.dsr_joint_angle_list = []
        self.dsr_joint_vel_list = []

        for i in range(len(self.motion_data_list)):
            joint_angle = self.src_joint_angle_list[i]
            joint_vel = self.src_joint_vel_list[i]
            motion_data = self.motion_data_list[i]

            dsr_joint_angle, dsr_joint_vel = self.process_data(joint_angle, joint_vel, motion_data)
            self.dsr_joint_angle_list.append(dsr_joint_angle)
            self.dsr_joint_vel_list.append(dsr_joint_vel)

        self.count = 0
        self.start = 0
        self.index = 0

    def load_mot(self, file_name='ik_files/s6ik1_processed.obj'):
        var = {}
        pkl_file = open(file_name + '', 'rb')
        var_temp = pickle.load(pkl_file)
        var.update(var_temp)
        pkl_file.close()
        return var

    def getJointAngle(self, motion_data):
        joint_angle = dict()
        motion_joint_data = motion_data['motion_data']

        surrogateTorsoYaw = np.zeros(np.shape(motion_joint_data['pelvis_rotation']))
        joint_angle.update({'torsoYaw': surrogateTorsoYaw.astype(float)})
        # joint_angle.update({'torsoYaw': -np.array(motion_joint_data['pelvis_rotation']).astype(float)})
        joint_angle.update({'torsoPitch': -np.array(motion_joint_data['pelvis_tilt']).astype(float)})
        surrogateTorsoRoll = np.zeros(np.shape(motion_joint_data['pelvis_list']))
        joint_angle.update({'torsoRoll': surrogateTorsoRoll.astype(float)})
        # joint_angle.update({'torsoRoll': -np.array(motion_joint_data['pelvis_list']).astype(float)})

        surrogateRightHipYaw = np.zeros(np.shape(motion_joint_data['hip_rotation_r']))
        joint_angle.update({'rightHipYaw': surrogateRightHipYaw.astype(float)})
        # joint_angle.update({'rightHipYaw': -np.array(motion_joint_data['hip_rotation_r']).astype(float)})
        surrogateRightHipRoll = np.zeros(np.shape(motion_joint_data['hip_adduction_r']))
        joint_angle.update({'rightHipRoll': surrogateRightHipRoll.astype(float)})
        # joint_angle.update({'rightHipRoll': -np.array(motion_joint_data['hip_adduction_r']).astype(float)})
        joint_angle.update({'rightHipPitch': -np.array(motion_joint_data['hip_flexion_r']).astype(float)})
        joint_angle.update({'rightKneePitch': -np.array(motion_joint_data['knee_angle_r']).astype(float)})
        joint_angle.update({'rightAnklePitch': -np.array(motion_joint_data['ankle_angle_r']).astype(float)})
        joint_angle.update({'rightAnkleRoll': -np.array(motion_joint_data['subtalar_angle_r']).astype(float)})

        surrogateLeftHipYaw = np.zeros(np.shape(motion_joint_data['hip_rotation_l']))
        joint_angle.update({'leftHipYaw': surrogateLeftHipYaw.astype(float)})
        # joint_angle.update({'leftHipYaw': -np.array(motion_joint_data['hip_rotation_l']).astype(float)})
        surrogateLeftHipRoll = np.zeros(np.shape(motion_joint_data['hip_adduction_l']))
        joint_angle.update({'leftHipRoll': surrogateLeftHipRoll.astype(float)})
        # joint_angle.update({'leftHipRoll': -np.array(motion_joint_data['hip_adduction_l']).astype(float)})
        joint_angle.update({'leftHipPitch': -np.array(motion_joint_data['hip_flexion_l']).astype(float)})
        joint_angle.update({'leftKneePitch': -np.array(motion_joint_data['knee_angle_l']).astype(float)})
        joint_angle.update({'leftAnklePitch': -np.array(motion_joint_data['ankle_angle_l']).astype(float)})
        joint_angle.update({'leftAnkleRoll': -np.array(motion_joint_data['subtalar_angle_l']).astype(float)})
        return copy.deepcopy(joint_angle)

    def getJointVel(self, joint_angle, motion_data):

        joint_velocity = dict()
        for key, value in joint_angle.items():
            value = joint_angle[key]
            temp = []
            length = len(value)
            temp.extend(value)
            temp.extend(value)
            temp.extend(value)
            vel = []
            temp = np.array(temp)
            temp = temp.astype(float)
            for i in range(length, 2 * length):
                v = (temp[i + 1] - temp[i]) * motion_data['data_freq']
                vel.append(v)
            vel[-1] = (vel[-2]+vel[0])/2.0
            joint_velocity.update({key: vel})

        # for key, value in joint_velocity.items():
        #     plt.plot(value, label=key)
        #     plt.legend()
        # plt.show()

        filter_joint_vel = dict()
        for key, value in joint_velocity.items():
            temp = []
            length = len(value)
            temp.extend(value)
            temp.extend(value)
            temp.extend(value)
            b, a = signal.butter(1, 0.25)
            y = signal.filtfilt(b, a, temp)
            vel = y[length:2 * length]
            filter_joint_vel.update({key: vel})

        # for key, value in filter_joint_vel.items():
        #     plt.plot(value, label=key)
        #     plt.legend()
        # plt.show()
        return copy.deepcopy(filter_joint_vel)

    def process_data(self, joint_angle, joint_vel, motion_data):
        src_data_freq = motion_data['data_freq']
        src_gait_freq = motion_data['gait_freq']

        dsr_joint_angle = dict()
        for key, value in joint_angle.items():
            array = np.zeros(self.dsr_length)
            src_length = len(motion_data['motion_data']['time'])
            for i in range(self.dsr_length):
                index = min((i*src_length//self.dsr_length), src_length)
                array[i] = value[index]
            dsr_joint_angle.update({key: array})

        dsr_joint_vel = dict()
        for key, value in joint_vel.items():
            array = np.zeros(self.dsr_length)
            src_length = len(motion_data['motion_data']['time'])
            for i in range(self.dsr_length):
                index = min((i*src_length//self.dsr_length), src_length)
                array[i] = value[index] * self.dsr_gait_freq / src_gait_freq # scale velocity using gait cycle frequency
            dsr_joint_vel.update({key: array})

        # for key, value in dsr_joint_vel.items():
        #     plt.plot(value, label=key)
        #     plt.legend()
        # plt.show()

        return copy.deepcopy(dsr_joint_angle), copy.deepcopy(dsr_joint_vel)

    def ref_joint_angle(self):
        joint_angle = self.dsr_joint_angle_list[self.index]
        joint = []
        for i in range(len(self.config.conf['controlled-joints'])):
            key = self.config.conf['controlled-joints'][i]
            angle = joint_angle[key][self.count]#*math.pi/180.0
            # print('joint_angle_std',key, np.std(joint_angle[key]))

            # angle = np.clip(angle, self.config.conf['action-bounds'][0][i], self.config.conf['action-bounds'][1][i])
            joint.append(angle)
        # for key in self.config.conf['controlled-joints']:
        #     joint.append(joint_angles[key][self.count])
        joint = np.array(joint)*math.pi/180.0
        joint = np.clip(joint, self.config.conf['action-bounds'][0], self.config.conf['action-bounds'][1])
        return joint

    def ref_joint_vel(self):
        joint_vel = self.dsr_joint_vel_list[self.index]
        joint = []
        for i in range(len(self.config.conf['controlled-joints'])):
            key = self.config.conf['controlled-joints'][i]
            vel = joint_vel[key][self.count]#*math.pi/180.0
            # print('joint_vel_std',key, np.std(joint_vel[key]))
            # angle = np.clip(angle, self.config.conf['action-bounds'][0][i], self.config.conf['action-bounds'][1][i])
            joint.append(vel)
        # for key in self.config.conf['controlled-joints']:
        #     joint.append(joint_angles[key][self.count])
        joint = np.array(joint)*math.pi/180.0
        # joint = np.clip(joint, self.config.conf['action-bounds'][0], self.config.conf['action-bounds'][1])
        return joint

    def ref_motion(self):
        joint_angle = self.dsr_joint_angle_list[self.index]
        joint_vel = self.dsr_joint_vel_list[self.index]
        joint_angle_array = []
        joint_vel_array = []
        for i in range(len(self.config.conf['controlled-joints'])):
            key = self.config.conf['controlled-joints'][i]
            angle = joint_angle[key][self.count]#*math.pi/180.0
            vel = joint_vel[key][self.count]#*math.pi/180.0
            joint_angle_array.append(angle)
            joint_vel_array.append(vel)
        # for key in self.config.conf['controlled-joints']:
        #     joint.append(joint_angles[key][self.count])
        joint_angle_array = np.array(joint_angle_array)*math.pi/180.0
        joint_angle_array = np.clip(joint_angle_array, self.config.conf['action-bounds'][0], self.config.conf['action-bounds'][1])
        joint_vel_array = np.array(joint_vel_array)*math.pi/180.0
        return joint_angle_array, joint_vel_array

    def ref_motion_dict(self):
        joint_angles = self.dsr_joint_angle_list[self.index]
        joint = dict()
        for i in range(len(self.config.conf['controlled-joints'])):
            key = self.config.conf['controlled-joints'][i]
            angle = joint_angles[key][self.count]*math.pi/180.0
            angle = np.clip(angle, self.config.conf['action-bounds'][0][i], self.config.conf['action-bounds'][1][i])
            joint.update({key:(angle)})

        return joint

    def ref_motion_vel_dict(self):
        joint_vel = self.dsr_joint_vel_list[self.index]
        joint = dict()
        for i in range(len(self.config.conf['controlled-joints'])):
            key = self.config.conf['controlled-joints'][i]
            vel = joint_vel[key][self.count]*math.pi/180.0
            joint.update({key:(vel)})

        return joint

    def ref_motion_avg(self):
        joint = []
        for key in self.config.conf['controlled-joints']:
            temp = 0
            length = len(self.dsr_joint_angle_list)
            for i in range(length):
                temp += self.dsr_joint_angle_list[i][key][self.count]
            temp/=length
            joint.append(temp)
        joint = np.array(joint)*math.pi/180.0
        joint = np.clip(joint, self.config.conf['action-bounds'][0], self.config.conf['action-bounds'][1])
        return joint

    def reset(self, index=None):
        if index is None:
            self.index = numpy.random.randint(len(self.dsr_joint_angle_list))
        else:
            self.index = index
        self.count = 0
        self.start = 0

    def index_count(self):
        self.count += 1
        self.count = self.count%self.dsr_length

    def random_count(self):
        self.count = numpy.random.randint(self.dsr_length)

    def euler_to_quat(self, roll, pitch, yaw): #rad
        cy = np.cos(yaw*0.5)
        sy = np.sin(yaw*0.5)
        cr = np.cos(roll*0.5)
        sr = np.sin(roll*0.5)
        cp = np.cos(pitch*0.5)
        sp = np.sin(pitch*0.5)

        w = cy*cr*cp+sy*sr*sp
        x = cy*sr*cp-sy*cr*sp
        y = cy*cr*sp+sy*sr*cp
        z = sy*cr*cp-cy*sr*sp

        return [x,y,z,w]

    def get_base_orn(self):#calculate base orientation from hip flexation
        left_max = max(self.dsr_joint_angle_list[self.index]['leftHipPitch'])
        left_min = min(self.dsr_joint_angle_list[self.index]['leftHipPitch'])
        right_max = max(self.dsr_joint_angle_list[self.index]['rightHipPitch'])
        right_min = min(self.dsr_joint_angle_list[self.index]['rightHipPitch'])

        offset_angle = -(left_max+right_max+left_min+right_min)/4.0 + 2.0
        # print(offset_angle)
        offset_rad = offset_angle*math.pi/180.0

        quat = self.euler_to_quat(0.0,offset_rad,0.0)
        return quat