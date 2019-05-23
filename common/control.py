from common.Interpolate import *
import numpy as np
import time

class Control:
    def __init__(self, config, env):
        self.env = env
        self.config = config

        self.PD_freq = self.config.conf['LLC-frequency']
        self.Physics_freq = self.config.conf['Physics-frequency']
        self.network_freq = self.config.conf['HLC-frequency']
        self.sampling_skip = int(self.PD_freq/self.network_freq)

        self.joint_interpolate = {}
        for joint in self.config.conf['actor-action-joints']:
            interpolate = JointTrajectoryInterpolate()
            # joint_interpolate[joint] = interpolate
            self.joint_interpolate.update({joint: interpolate})

        self.reward_decay = 1.0
        self.reward_scale = config.conf['reward-scale']

        self.action = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.action_actor = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.prev_action = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.action_interpolate = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.action_control = np.zeros((len(self.config.conf['controlled-joints']),))

        # self.actor_bound = np.ones((2,len(self.config.conf['controlled-joints'])))
        # self.actor_bound[0][:] = -1 * np.ones((len(self.config.conf['controlled-joints']),))
        # self.actor_bound[1][:] = 1* np.ones((len(self.config.conf['controlled-joints']),))
        self.actor_bound = np.ones((2,len(self.config.conf['actor-action-joints'])))
        # self.actor_bound[0][:] = -1 * np.ones((len(self.config.conf['actor-action-joints']),))
        # self.actor_bound[1][:] = 1* np.ones((len(self.config.conf['actor-action-joints']),))
        self.actor_bound=self.config.conf['actor-output-bounds']
        self.control_bound = self.config.conf['action-bounds']

        self.info = []

        self.w_imitation = 0.0
        self.w_task = 1.0

        #Different importance weights for different joints
        self.w_joints = list()
        for key in self.config.conf['controlled-joints']:
            # if 'torso' in key: # lower weights for torso joints
            #     self.w_joints.append(0.5)#1
            if 'torsoPitch' in key: # lower weights for torso joints
                self.w_joints.append(1)#1
            elif 'HipPitch' in key: # higher weights for hip pitch joints
                self.w_joints.append(2)#4
            elif 'KneePitch' in key:
                self.w_joints.append(2)#2
            elif 'AnklePitch' in key:
                self.w_joints.append(1)
            else:
                self.w_joints.append(1)

        self.w_joints = np.array(self.w_joints)

        self.gait_phase = 0.0
        self.ref_joint_angle = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.ref_joint_vel = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.ref_end_effector = np.zeros((len(self.config.conf['actor-action-joints']),))

        self.PD_dt = 1.0/float(self.PD_freq)
        self.network_dt = 1.0/float(self.network_freq)

    def control_step(self, action, force = np.array([0,0,0]), gait_phase = 0.0):
        self.gait_phase = gait_phase
        self.force = force

        self.action_actor = action
        # self.action = self.rescale(self.action, self.actor_bound, self.control_bound) #rescaled action
        # self.action = np.clip(self.action, self.config.conf['action-bounds'][0], self.config.conf['action-bounds'][1])
        #self.action = np.array(self.action)

        # map the output from the actor to the actual control configuration
        if len(self.action_control) == 7 and len(self.action_actor) == 4:
            self.action_control[0:4] = self.action_actor[0:4]
            self.action_control[4:7] = self.action_actor[1:4]  # duplicate leg control signals
        # elif len(self.action_control) == 11 and len(self.action_actor) == 4:
        #     self.action_control[0] = self.action_actor[0]  # torso pitch
        #     self.action_control[1] = 0.0  # hip roll
        #     self.action_control[2] = self.action_actor[1]  # hip pitch
        #     self.action_control[3] = self.action_actor[2]  # knee pitch
        #     self.action_control[4] = self.action_actor[3]  # ankle pitch
        #     self.action_control[5] = 0.0  # ankle roll
        #     self.action_control[6:11] = self.action_control[1:6]
        # elif len(self.action_control) == 13 and len(self.action_actor) == 4:
        #     self.action_control[0] = self.action_actor[0]  # torso pitch
        #     self.action_control[1] = 0.0  # hip yaw
        #     self.action_control[2] = 0.0  # hip roll
        #     self.action_control[3] = self.action_actor[1]  # hip pitch
        #     self.action_control[4] = self.action_actor[2]  # knee pitch
        #     self.action_control[5] = self.action_actor[3]  # ankle pitch
        #     self.action_control[6] = 0.0  # ankle roll
        #     self.action_control[7:13] = self.action_control[1:7]
        # elif len(self.action_control) == 11 and len(self.action_actor) == 11:
        #     self.action_control[:] = self.action_actor[:]
        # elif len(self.action_control) == 13 and len(self.action_actor) == 11:
        #     self.action_control[0] = self.action_actor[0]  # torso pitch
        #     self.action_control[1] = 0.0  # hip yaw
        #     self.action_control[2] = self.action_actor[1]  # hip roll
        #     self.action_control[3] = self.action_actor[2]  # hip pitch
        #     self.action_control[4] = self.action_actor[3]  # knee pitch
        #     self.action_control[5] = self.action_actor[4]  # ankle pitch
        #     self.action_control[6] = self.action_actor[5]  # ankle roll
        #     self.action_control[7] = 0.0  # hip yaw
        #     self.action_control[8] = self.action_actor[6]  # hip roll
        #     self.action_control[9] = self.action_actor[7]  # hip pitch
        #     self.action_control[10] = self.action_actor[8]  # knee pitch
        #     self.action_control[11] = self.action_actor[9]  # ankle pitch
        #     self.action_control[12] = self.action_actor[10]  # ankle roll
        elif len(self.action_control) == len(self.action_actor):
            self.action_control[:] = self.action_actor[:]

        #rescale action output
        self.action_control = self.rescale(self.action_control, self.actor_bound, self.control_bound)  # rescaled action
        # t = time.time()
        for n in range(len(self.config.conf['actor-action-joints'])):
            joint_name = self.config.conf['actor-action-joints'][n]
            self.joint_interpolate[joint_name].cubic_interpolation_setup(self.prev_action[n], 0, self.action_control[n], 0,
                                                                         self.network_dt)
        self.prev_action = np.array(self.action_control)
        reward_add = 0
        for i in range(self.sampling_skip):
            if self.config.conf['joint-interpolation']:
                for n in range(len(self.config.conf['actor-action-joints'])):
                    joint_name = self.config.conf['actor-action-joints'][n]
                    self.action_interpolate[n] = self.joint_interpolate[joint_name].interpolate(self.PD_dt)
            else:
                self.action_interpolate = self.action

            # self.action_control = self.rescale(self.action_control, self.actor_bound, self.control_bound)
            # TODO check lipping. clipping action might not be good for PD control
            self.action_control = np.clip(self.action_interpolate, self.config.conf['action-bounds'][0], self.config.conf['action-bounds'][1])
            # t = time.time()
            next_state, reward, done, _ = self.env._step(self.action_control, self.force)
            # print(time.time()-t)
            reward_add = reward + self.reward_decay * reward_add #averaging the reward, might encourage high frequency responses
        #reward *= 0.1
        reward_add/=self.sampling_skip
        reward_task = (reward_add)*self.reward_scale

        reward_joint_angle = self.reward_joint_angle()
        reward_joint_vel = self.reward_joint_vel()
        reward_contact = self.reward_foot_contact()
        # reward_imitation = (10 * reward_joint_angle +reward_contact) * self.reward_scale
        reward_imitation = (10 * (3 * reward_joint_angle + reward_joint_vel) / 4.0 +reward_contact) * self.reward_scale
        reward = self.w_task*reward_task + self.w_imitation*reward_imitation# + 10 * self.reward_action_bound(self.action_control, self.control_bound)
        #reward = reward_task
        # print(time.time()-t)
        self.info = dict([
            ('task_reward', reward_task),
            ('imitation_reward', reward_imitation),
            ('total_reward', reward),
            ('reward_joint_angle', reward_joint_angle),
            ('reward_joint_vel', reward_joint_vel),
            ('reward_contact', reward_contact)
                          ])
        return np.array(next_state), reward, done, self.info

    def update_ref(self, joint_angle, joint_vel, end_effector):
        self.ref_joint_angle = joint_angle
        self.ref_joint_vel = joint_vel
        self.ref_end_effector = end_effector
        return

    def reset(self, w_imitation = 0.0, w_task = 1.0):
        self.info = []
        self.w_imitation = w_imitation
        self.w_task = w_task
        self.action = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.prev_action = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.action_interpolate = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.action_control = np.zeros((len(self.config.conf['controlled-joints']),))

    def rescale(self, value, old_range, new_range):
        value = np.array(value)
        old_range = np.array(old_range)
        new_range = np.array(new_range)

        OldRange = old_range[1][:] - old_range[0][:]
        NewRange = new_range[1][:] - new_range[0][:]
        NewValue = (value - old_range[0][:]) * NewRange / OldRange + new_range[0][:]
        return NewValue

    def clip_action(self, state, action):

        return

    def get_joint_angle(self):
        joint_angle_dict = self.env.getJointAngleDict()
        joint_angle_list = []
        for key in self.config.conf['controlled-joints']:
            joint_angle_list.append(joint_angle_dict[key])
        joint_angle_array = np.array(joint_angle_list)

        return joint_angle_array

    def get_joint_vel(self):
        joint_vel_dict = self.env.getJointVelDict()
        joint_vel_list = []
        for key in self.config.conf['controlled-joints']:
            joint_vel_list.append(joint_vel_dict[key])
        joint_vel_array = np.array(joint_vel_list)

        return joint_vel_array

    def get_joint_torque(self):
        joint_torque_dict = self.env.getJointTorqueDict()
        joint_torque_list = []
        for key in self.config.conf['controlled-joints']:
            joint_torque_list.append(joint_torque_dict[key])
        joint_torque_array = np.array(joint_torque_list)

        return joint_torque_array

    def reward_action_bound(self, action, bound):
        lower_bound = np.maximum(0.0, bound[0]-action)
        upper_bound = np.maximum(0.0, action-bound[1])
        reward = -np.mean(lower_bound+upper_bound)
        #print(reward)
        return reward

    def reward_joint_angle(self):
        alpha = 1e-3  # 1e-1
        err_norm = 20.0/180.0*np.pi#20
        err = (self.ref_joint_angle - self.get_joint_angle()) / err_norm#0.4#0.2
        reward = np.sum(np.exp(np.log(alpha) * (err) ** 2) * self.w_joints) / np.sum(self.w_joints)

        return reward

    def reward_joint_vel(self):
        alpha = 1e-3  # 1e-1
        err_norm = 150.0/180.0*np.pi#150
        err = (self.ref_joint_vel - self.get_joint_vel()) / err_norm#0.4#0.2
        reward = np.sum(np.exp(np.log(alpha) * (err) ** 2) * self.w_joints) / np.sum(self.w_joints)

        return reward

    def penalty_joint_vel(self):
        err = np.power(self.get_joint_vel(),2)
        reward = -10e-2*np.mean(err)
        return reward

    def penalty_joint_torque(self):
        err = np.power(self.get_joint_torque(),2)
        reward = -10e-5*np.mean(err)
        return reward

    def reward_foot_contact(self):
        reward_term = 0
        gap = 0.1 #20% overlap
        # if self.gait_phase>gap and self.gait_phase<(0.5-gap):
        # if self.gait_phase >= gap and self.gait_phase <= 0.5:
        #     if (self.env.checkGroundContact('right') == True) and (self.env.checkGroundContact('left') == False):
        #         reward_term = 2
        #     else:
        #         reward_term = 0
        # elif self.gait_phase >= (0.5+gap) and self.gait_phase <= 1.0:
        # # elif self.gait_phase>(0.5+gap) and self.gait_phase<(1.0-gap):
        #     if (self.env.checkGroundContact('right') == False) and (self.env.checkGroundContact('left') == True):
        #         reward_term = 2
        #     else:
        #         reward_term = 0
        # else:
        #     if (self.env.checkGroundContact('right') == True) and (self.env.checkGroundContact('left') == True):
        #         reward_term = 2
        #     else:
        #         reward_term = 0

        if self.gait_phase >= 0 and self.gait_phase < gap:
            if (self.env.checkGroundContact('right') == True): #allow double contact
                reward_term = 2
            else:
                reward_term = 0
        elif self.gait_phase >= gap and self.gait_phase < 0.5:
            if (self.env.checkGroundContact('right') == True) and (self.env.checkGroundContact('left') == False):
                reward_term = 2
            else:
                reward_term = 0
        elif self.gait_phase >= 0.5 and self.gait_phase < (0.5+gap):
        # elif self.gait_phase>(0.5+gap) and self.gait_phase<(1.0-gap):
            if (self.env.checkGroundContact('left') == True): #allow double contact
                reward_term = 2
            else:
                reward_term = 0
        elif self.gait_phase >= (0.5+gap) and self.gait_phase <= 1.0:
        # elif self.gait_phase>(0.5+gap) and self.gait_phase<(1.0-gap):
            if (self.env.checkGroundContact('right') == False) and (self.env.checkGroundContact('left') == True):
                reward_term = 2
            else:
                reward_term = 0
        else:
            reward_term = 0
        return reward_term

    def set_reward_weights(self, imitation_reward, task_reward):
        imitation_reward = max(1.0, imitation_reward)
        task_reward = max(1.0, task_reward)
        #balance the weight for imitation and task rewards
        #higher imitation rewards, increase the weight for task rewards
        self.w_task = np.clip(imitation_reward/(task_reward+imitation_reward), 0.2,0.8)
        self.w_imitation = 1.0-self.w_task
        return