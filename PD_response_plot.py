import gc
import inspect
import os

from TRPO.configuration import *
from agent.agent_PPO import *
from common.control import *
from common.logger import logger
from common.value_trace import *
from valkyrie_gym_env import Valkyrie
import matplotlib.pyplot as plt
import time
from common.motion_new import Motion
from datetime import datetime

gc.enable()
import moviepy as mpy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

class Run():
    def __init__(self, config, dir_path):
        self.dir_path = dir_path
        self.config = config

        self.PD_freq = self.config.conf['LLC-frequency']
        self.Physics_freq = self.config.conf['Physics-frequency']
        self.network_freq = self.config.conf['HLC-frequency']
        self.sampling_skip = int(self.PD_freq/self.network_freq)

        self.max_time = 6
        self.max_step_per_episode = int(self.max_time*self.network_freq)

        self.env = Valkyrie(
            max_time=self.max_time, renders=True, initial_gap_time=0.1, PD_freq=self.PD_freq,
            Physics_freq=self.Physics_freq, Kp=config.conf['Kp'], Kd=config.conf['Kd'],
            bullet_default_PD=config.conf['bullet-default-PD'], controlled_joints_list=config.conf['controlled-joints'],
            logFileName=dir_path, isEnableSelfCollision=False)

        config.conf['state-dim'] = self.env.stateNumber+1

        self.logging = logger(dir_path)

        self.episode_count = 0
        self.step_count = 0
        self.train_iter_count = 0

        self.best_reward = 0
        self.best_episode = 0
        self.best_train_iter = 0

        self.control = Control(self.config, self.env)

        # img = [[1,2,3]*50]*100
        img = np.zeros((240,320,3))
        self.image = plt.imshow(img,interpolation='none',animated=True)

        self.ax=plt.gca()
        plt.axis('off')

        self.image_list = []

        self.ref_motion = Motion(config=self.config, dsr_gait_freq=0.6)

    def test(self):
        total_reward = 0
        for i in range(self.config.conf['test-num']):#
            quat = self.ref_motion.euler_to_quat(0,0,0)
            _ = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'], base_pos_nom=[0,0,1.575], base_orn_nom=quat, fixed_base=True)
            # state = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'], base_pos_nom=[0, 0, 1.175], fixed_base=False)
            q_nom = self.ref_motion.ref_motion_dict()
            base_orn_nom = self.ref_motion.get_base_orn()

            # state = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'], q_nom=q_nom, base_orn_nom=base_orn_nom, base_pos_nom=[0, 0, 1.175], fixed_base=False)
            # self.env._setupCamera()
            self.env.startRendering()
            self.env._startLoggingVideo()

            for step in range(self.max_step_per_episode):
                if step>=2*self.network_freq and step<4*self.network_freq:
                    action = [0,0,0,0,0,0,0,0,0.1,0,0]
                else:
                    action = [0,0,0,0,0,0,0,0,0,0,0]
                # action = np.clip(action, self.config.conf['actor-output-bounds'][0],
                #                  self.config.conf['actor-output-bounds'][1])
                action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)

                rgb=self.env._render(roll=0,pitch=0,yaw=90)
                print(rgb.shape)
                self.image_list.append(rgb)
                for i in range(self.sampling_skip):
                    # action = self.control.rescale(ref_action, self.config.conf['action-bounds'],
                    #                               self.config.conf['actor-output-bounds'])
                    _,_,_,_ = self.env._step(action)

                    self.logging.add_run('action', action)

                    joint_angle = self.control.get_joint_angle()
                    self.logging.add_run('joint_angle', joint_angle)
                    readings = self.env.getExtendedReading()
                    ob = self.env.getObservation()
                    for l in range(len(ob)):
                        self.logging.add_run('observation' + str(l), ob[l])
                    # for key, value in readings.items():
                    #     self.logging.add_run(key, value)

            self.env._stopLoggingVideo()
            self.env.stopRendering()

        ave_reward = total_reward/self.config.conf['test-num']
        print(ave_reward)
        self.logging.save_run()


        clip = ImageSequenceClip(self.image_list, fps=25)
        clip.write_gif(self.dir_path+'/test.gif')
        clip.write_videofile(self.dir_path+'/test.mp4', fps=25, audio=False)



def main():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    os.sys.path.insert(0, parentdir)
    config = Configuration()
    path_str = 'PD_response_plot'
    dir_path = path_str + datetime.now().strftime('%Y_%m_%d_%H.%M.%S')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(dir_path + '/saved_actor_networks'):
        os.makedirs(dir_path + '/saved_actor_networks')
    if not os.path.exists(dir_path + '/saved_critic_networks'):
        os.makedirs(dir_path + '/saved_critic_networks')

    test = Run(config, dir_path)
    test.test()

if __name__ == '__main__':
    main()
