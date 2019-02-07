import gc
import inspect
import os

from TRPO.configuration import *
from agent.agent_TRPO import *
from common.control import *
from common.logger import logger
from common.value_trace import *
from valkyrie_gym_env import Valkyrie
import matplotlib.pyplot as plt
import time
gc.enable()
import moviepy as mpy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

class Run():
    def __init__(self, config, dir_path):
        self.dir_path = dir_path
        self.config = config
        self.config.load_configuration(dir_path)
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

        self.env = Valkyrie(
            max_time=self.max_time, renders=False, initial_gap_time=0.5, PD_freq=self.PD_freq,
            Physics_freq=self.Physics_freq, Kp=config.conf['Kp'], Kd=config.conf['Kd'],
            bullet_default_PD=config.conf['bullet-default-PD'], controlled_joints_list=config.conf['controlled-joints'],
            logFileName=dir_path)

        config.conf['state-dim'] = self.env.stateNumber
        self.agent = Agent(self.env, self.config)
        self.agent.load_weight(dir_path+'/best_network')

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

        # img = [[1,2,3]*50]*100
        img = np.zeros((240,320,3))
        self.image = plt.imshow(img,interpolation='none',animated=True)

        self.ax=plt.gca()
        plt.axis('off')

        self.image_list = []

    def test(self):
        total_reward = 0
        for i in range(self.config.conf['test-num']):#
            _ = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'], base_pos_nom=[0,0,1.5], fixed_base=True)
            # self.env._setupCamera()
            self.env.startRendering()
            self.env._startLoggingVideo()
            self.control.reset(w_imitation=0.5, w_task=0.5)

            for step in range(self.max_step_per_episode):
                # self.env._setupCamera()
                t = time.time()
                state = self.env.getExtendedObservation()

                action, actor_info = self.agent.agent.actor.get_action(state)
                mean = actor_info['mean']
                logstd = actor_info['logstd']
                action = mean
                # action = np.clip(action, self.config.conf['actor-output-bounds'][0],
                #                  self.config.conf['actor-output-bounds'][1])
                action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)

                f = self.env.rejectableForce_xy(1.0 / self.network_freq)
                rgb=self.env._render()
                print(rgb.shape)
                self.image_list.append(rgb)
                # self.image.set_data(rgb)
                # self.ax.plot([0])
                # plt.pause(0.005)

                # if step == 5 * self.network_freq:
                #     if f[1] == 0:
                #         self.force = np.array([600.0 * self.network_freq / 10.0, 0.0, 0.0])
                #     else:
                #         self.force = np.array([1.0 * f[1], 0, 0])
                #     print(self.force)
                # elif step == 11 * self.network_freq:
                #     if f[0] == 0:
                #         self.force = np.array([-600.0 * self.network_freq / 10.0, 0.0, 0.0])
                #     else:
                #         self.force = [1.0 * f[0], 0, 0]
                #     print(self.force)
                # elif step == 17 * self.network_freq:
                #     if f[2] == 0:
                #         self.force = np.array([0.0, -800.0 * self.network_freq / 10.0, 0.0])
                #     else:
                #         self.force = [0, 1.0 * f[2], 0]
                #     print(self.force)
                # elif step == 23 * self.network_freq:
                #     if f[3] == 0:
                #         self.force = np.array([0.0, 800.0 * self.network_freq / 10.0, 0.0])
                #     else:
                #         self.force = [0, 1.0 * f[3], 0]
                #     print(self.force)
                # else:
                #     self.force = [0, 0, 0]

                next_state, reward, done, info = self.control.control_step(action, self.force, np.zeros((len(self.config.conf['controlled-joints']),)))

                total_reward += reward

                ob = self.env.getObservation()
                ob_filtered = self.env.getFilteredObservation()
                for l in range(len(ob)):
                    self.logging.add_run('observation' + str(l), ob[l])
                    self.logging.add_run('filtered_observation' + str(l), ob_filtered[l])
                self.logging.add_run('action', action)
                readings = self.env.getExtendedReading()
                for key, value in readings.items():
                    self.logging.add_run(key, value)
                self.logging.add_run('task_reward', info['task_reward'])
                self.logging.add_run('imitation_reward', info['imitation_reward'])
                self.logging.add_run('total_reward', info['total_reward'])
                #
                # while 1:
                #     if(time.time()-t)>1.0/self.network_freq:
                #         break

                if done:
                    break
            self.env._stopLoggingVideo()
            self.env.stopRendering()

        clip = ImageSequenceClip(self.image_list, fps=25)
        clip.write_gif(self.dir_path+'/test.gif')
        clip.write_videofile(self.dir_path+'/test.mp4', fps=25, audio=False)
        ave_reward = total_reward/self.config.conf['test-num']

        print(ave_reward)
        self.logging.save_run()


def main():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    os.sys.path.insert(0, parentdir)
    config = Configuration()
    dir_path = 'TRPO/record/3D_walk_imitation/with_external_force_disturbance/2018_07_04_13.49.47'  # '2017_05_29_18.23.49/with_force'
    test = Run(config, dir_path)
    test.test()

if __name__ == '__main__':
    main()
