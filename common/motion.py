import numpy as np
import pickle
import math
import copy
import numpy
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

        self.src_var_list = []
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s1ik1_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s1ik2_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s1ik3_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s2ik1_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s2ik2_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s2ik3_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s3ik1_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s3ik2_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s3ik3_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s4ik1_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s4ik2_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s4ik3_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s5ik1_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s5ik2_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s5ik3_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s6ik1_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s6ik2_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s6ik3_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s7ik1_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s7ik2_processed.obj'))
        self.src_var_list.append(self.load_mot('/home/chuanyu/Dropbox/Valkyrie_IPG_test/ik_files/s7ik3_processed.obj'))

        self.dsr_motion_data_list = []
        self.joint_angles_list = []
        for src_var in self.src_var_list:
            dsr_motion_data = self.process_motion_data(src_var)
            self.dsr_motion_data_list.append(dsr_motion_data)
            joint_angles = self.map_motion_to_joint(dsr_motion_data)
            self.joint_angles_list.append(joint_angles)

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

    def process_motion_data(self, var):
        dsr_motion_data = dict()
        data_type = var['data_type']
        src_motion_data = var['motion_data']

        src_data_freq = var['data_freq'] #source frequency
        src_gait_freq = var['gait_freq'] #time step for full gait cycle
        src_length = len(src_motion_data['time'])
        for key in data_type:
            array = np.zeros(self.dsr_length)
            for i in range(self.dsr_length):
                index = min((i*src_length//self.dsr_length), src_length)
                array[i] = src_motion_data[key][index]
            dsr_motion_data.update({key: array})

        return copy.deepcopy(dsr_motion_data)

    def map_motion_to_joint(self, dsr_motion_data):
        joint_angles = dict()
        surrogateTorsoYaw = np.zeros(np.shape(dsr_motion_data['pelvis_rotation']))
        joint_angles.update({'torsoYaw': surrogateTorsoYaw})
        #joint_angles.update({'torsoYaw': -dsr_motion_data['pelvis_rotation']})
        joint_angles.update({'torsoPitch': -dsr_motion_data['pelvis_tilt']})
        joint_angles.update({'torsoRoll': -dsr_motion_data['pelvis_list']})

        surrogateRightHipYaw = np.zeros(np.shape(dsr_motion_data['hip_rotation_r']))
        joint_angles.update({'rightHipYaw': surrogateRightHipYaw})
        #joint_angles.update({'rightHipYaw': -dsr_motion_data['hip_rotation_r']})
        joint_angles.update({'rightHipRoll': -dsr_motion_data['hip_adduction_r']})
        joint_angles.update({'rightHipPitch': -dsr_motion_data['hip_flexion_r']})
        joint_angles.update({'rightKneePitch': -dsr_motion_data['knee_angle_r']})
        joint_angles.update({'rightAnklePitch': -dsr_motion_data['ankle_angle_r']})
        joint_angles.update({'rightAnkleRoll': -dsr_motion_data['subtalar_angle_r']})

        surrogateLeftHipYaw = np.zeros(np.shape(dsr_motion_data['hip_rotation_l']))
        joint_angles.update({'leftHipYaw': surrogateLeftHipYaw})
        # joint_angles.update({'leftHipYaw': -dsr_motion_data['hip_rotation_l']})
        joint_angles.update({'leftHipRoll': -dsr_motion_data['hip_adduction_l']})
        joint_angles.update({'leftHipPitch': -dsr_motion_data['hip_flexion_l']})
        joint_angles.update({'leftKneePitch': -dsr_motion_data['knee_angle_l']})
        joint_angles.update({'leftAnklePitch': -dsr_motion_data['ankle_angle_l']})
        joint_angles.update({'leftAnkleRoll': -dsr_motion_data['subtalar_angle_l']})
        return copy.deepcopy(joint_angles)

    def ref_motion(self):
        joint_angles = self.joint_angles_list[self.index]
        joint = []
        for i in range(len(self.config.conf['controlled-joints'])):
            key = self.config.conf['controlled-joints'][i]
            angle = joint_angles[key][self.count]#*math.pi/180.0
            # angle = np.clip(angle, self.config.conf['action-bounds'][0][i], self.config.conf['action-bounds'][1][i])
            joint.append(angle)
        # for key in self.config.conf['controlled-joints']:
        #     joint.append(joint_angles[key][self.count])
        joint = np.array(joint)*math.pi/180.0
        joint = np.clip(joint, self.config.conf['action-bounds'][0], self.config.conf['action-bounds'][1])
        self.count += 1
        self.count = self.count%self.dsr_length
        return joint

    def ref_motion_dict(self):
        joint_angles = self.joint_angles_list[self.index]
        joint = dict()
        for i in range(len(self.config.conf['controlled-joints'])):
            key = self.config.conf['controlled-joints'][i]
            angle = joint_angles[key][self.count]*math.pi/180.0
            angle = np.clip(angle, self.config.conf['action-bounds'][0][i], self.config.conf['action-bounds'][1][i])
            joint.update({key:(angle)})

        return joint

    def ref_motion_avg(self):
        joint = []
        for key in self.config.conf['controlled-joints']:
            temp = 0
            length = len(self.joint_angles_list)
            for i in range(length):
                temp += self.joint_angles_list[i][key][self.count]
            temp/=length
            joint.append(temp)
        joint = np.array(joint)*math.pi/180.0
        joint = np.clip(joint, self.config.conf['action-bounds'][0], self.config.conf['action-bounds'][1])
        self.count += 1
        self.count = self.count%self.dsr_length
        return joint

    def reset(self, index=None):
        if index is None:
            self.index = numpy.random.randint(len(self.joint_angles_list))
        else:
            self.index = index
        self.count = 0
        self.start = 0

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
        left_max = max(self.joint_angles_list[self.index]['leftHipPitch'])
        left_min = min(self.joint_angles_list[self.index]['leftHipPitch'])
        right_max = max(self.joint_angles_list[self.index]['rightHipPitch'])
        right_min = min(self.joint_angles_list[self.index]['rightHipPitch'])

        offset_angle = -(left_max+right_max+left_min+right_min)/4.0 + 2.0
        # print(offset_angle)
        offset_rad = offset_angle*math.pi/180.0

        quat = self.euler_to_quat(0.0,offset_rad,0.0)
        return quat