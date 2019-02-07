import gc
import inspect
import os

from TRPO.configuration import *
from agent.agent_TRPO import *
from common.control import *
from common.logger import logger
from common.value_trace import *
from walkman_gym_env import Walkman
import matplotlib.pyplot as plt
import time
gc.enable()
import moviepy as mpy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from common.motion_new import Motion
import pybullet as p

class Run():
    def __init__(self, config, dir_path):
        self.dir_path = dir_path
        self.config = config
        self.config.print_configuration()

        self.PD_freq = self.config.conf['LLC-frequency']
        self.Physics_freq = self.config.conf['Physics-frequency']
        self.network_freq = self.config.conf['HLC-frequency']
        self.sampling_skip = int(self.PD_freq/self.network_freq)

        self.reward_decay = 1.0
        self.reward_scale = config.conf['reward-scale']
        self.reward_scale = self.reward_scale / float(self.sampling_skip)  # /10.0#normalizing reward to 1
        self.max_time = 10#16
        self.max_step_per_episode = int(self.max_time*self.network_freq)

        self.env = Walkman(
            max_time=self.max_time, renders=True, initial_gap_time=0.1, PD_freq=self.PD_freq,
            Physics_freq=self.Physics_freq, Kp=config.conf['Kp'], Kd=config.conf['Kd'],
            bullet_default_PD=config.conf['bullet-default-PD'], controlled_joints_list=config.conf['controlled-joints'],
            logFileName=dir_path, isEnableSelfCollision=False)

        config.conf['state-dim'] = self.env.stateNumber
        self.agent = Agent(self.env, self.config)
        # self.agent.load_weight(dir_path+'/best_network')

        self.logging = logger(dir_path)

        self.episode_count = 0
        self.step_count = 0
        self.train_iter_count = 0

        self.best_reward = 0
        self.best_episode = 0
        self.best_train_iter = 0

        self.control = Control(self.config, self.env)

        # create new network

        self.force = [0,0,0]
        self.force_chest = [0, 0, 0]  # max(0,force_chest[1]-300*1.0 / EXPLORE)]
        self.force_pelvis = [0, 0, 0]

        self.motion = Motion(config)

        self.image_list = []

    def test(self):
        total_reward = 0
        for i in range(self.config.conf['test-num']):#
            self.control.reset()
            self.motion.reset(index=0)
            self.motion.count = 0
            # self.motion.random_count()
            q_nom = self.motion.ref_motion_dict()

            print(self.motion.get_base_orn())
            # print(q_nom['torsoPitch'])
            # print(self.motion.ref_motion())
            print(q_nom)
            base_orn_nom = self.motion.get_base_orn()#[0.000,0.078,0.000,0.997]#[0,0,0,1]
            _ = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'], base_pos_nom=[0,0,1.5], fixed_base=False, q_nom=q_nom, base_orn_nom=base_orn_nom)
            left_foot_link_state = p.getLinkState(self.env.r, self.env.jointIdx['leftAnkleRoll'], computeLinkVelocity=0)
            left_foot_link_dis = np.array(left_foot_link_state[0])
            right_foot_link_state = p.getLinkState(self.env.r, self.env.jointIdx['rightAnkleRoll'], computeLinkVelocity=0)
            right_foot_link_dis = np.array(right_foot_link_state[0])
            print(left_foot_link_dis-right_foot_link_dis)
            # ref_action = self.motion.ref_motion()
            # for i in range(len(self.config.conf['controlled-joints'])):
            #     q_nom.update({self.config.conf['controlled-joints'][i]:ref_action[i]})
            #
            # # _ = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'])
            # _ = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'], q_nom=q_nom, base_orn_nom=base_orn_nom)
            # self.env._setupCamera()
            self.env.startRendering()
            self.env._startLoggingVideo()

            print(self.motion.index)

            for step in range(self.max_step_per_episode):
                # self.env._setupCamera()
                t = time.time()
                state = self.env.getExtendedObservation()

                # action = self.motion.ref_motion_avg()
                # ref_angle, ref_vel = self.motion.ref_motion()
                ref_angle = self.motion.ref_joint_angle()
                ref_vel = self.motion.ref_joint_vel()
                action = self.control.rescale(ref_angle, self.config.conf['action-bounds'], self.config.conf['actor-output-bounds'])

                # rgb=self.env._render(pitch=0)
                # # print(rgb.shape)
                # self.image_list.append(rgb)

                next_state, reward, done, info = self.control.control_step(action, self.force)
                self.motion.index_count()

                total_reward += reward
                self.logging.add_run('ref_angle', np.squeeze(ref_angle))
                self.logging.add_run('ref_vel', np.squeeze(ref_vel))
                # self.logging.add_run('measured_action', np.squeeze(self.control.get_joint_angles()))
                ob = self.env.getObservation()
                ob_filtered = self.env.getFilteredObservation()
                for l in range(len(ob)):
                    self.logging.add_run('observation' + str(l), ob[l])
                    self.logging.add_run('filtered_observation' + str(l), ob_filtered[l])
                self.logging.add_run('action', action)
                # readings = self.env.getExtendedReading()
                # for key, value in readings.items():
                #     self.logging.add_run(key, value)
                #
                # while 1:
                #     if(time.time()-t)>1.0/self.network_freq:
                #         break

                if done:
                    break
                # print(time.time()-t)
            self.env._stopLoggingVideo()
            self.env.stopRendering()

        ave_reward = total_reward/self.config.conf['test-num']

        clip = ImageSequenceClip(self.image_list, fps=25)
        clip.write_gif('test.gif')
        clip.write_videofile('test.mp4', fps=25, audio=False)

        print(ave_reward)
        self.logging.save_run()


def main():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    os.sys.path.insert(0, parentdir)
    config = Configuration()
    dir_path = '/home/Valkyrie_IPG_test'  # '2017_05_29_18.23.49/with_force'
    test = Run(config, dir_path)
    test.test()

if __name__ == '__main__':
    main()
