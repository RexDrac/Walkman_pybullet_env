import gc
import inspect
import os

from PPO.configuration import *
from agent.agent_PPO import *
from common.control import *
from common.logger import logger
from common.value_trace import *
from valkyrie_gym_env import Valkyrie
import matplotlib.pyplot as plt
import time
from common.motion_new import Motion

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
            max_time=self.max_time, renders=True, initial_gap_time=0.1, PD_freq=self.PD_freq,
            Physics_freq=self.Physics_freq,# Kp=config.conf['Kp'], Kd=config.conf['Kd'],
            bullet_default_PD=config.conf['bullet-default-PD'], controlled_joints_list=config.conf['controlled-joints'],
            logFileName=dir_path, isEnableSelfCollision=True)

        config.conf['state-dim'] = self.env.stateNumber+1
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

        self.ref_motion = Motion(config=self.config, dsr_gait_freq=0.6)

    def test(self):
        total_reward = 0
        for i in range(self.config.conf['test-num']):#
            quat = self.ref_motion.euler_to_quat(0,0,0)
            _ = self.env._reset(#Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'],
                                base_pos_nom=[0,0,1.175], base_orn_nom=quat, fixed_base=False)
            # state = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'], base_pos_nom=[0, 0, 1.175], fixed_base=False)
            q_nom = self.ref_motion.ref_motion_dict()
            base_orn_nom = self.ref_motion.get_base_orn()

            # state = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'], q_nom=q_nom, base_orn_nom=base_orn_nom, base_pos_nom=[0, 0, 1.175], fixed_base=False)
            # self.env._setupCamera()
            self.env.startRendering()
            self.env._startLoggingVideo()
            self.ref_motion.reset(index=0)
            # self.ref_motion.random_count()

            self.control.reset(w_imitation=self.config.conf['imitation-weight'], w_task=self.config.conf['task-weight'])


            for step in range(self.max_step_per_episode):
                # self.env._setupCamera()
                t = time.time()
                gait_phase = self.ref_motion.count / self.ref_motion.dsr_length
                ref_angle = self.ref_motion.ref_joint_angle()
                ref_vel = self.ref_motion.ref_joint_vel()

                state = self.env.getExtendedObservation()
                state = np.squeeze(state)
                state = np.append(state, [gait_phase])
                # state = np.append(state, [np.sin(np.pi * 2 * gait_phase), np.cos(np.pi * 2 * gait_phase)])
                # state = np.append(state,[0,0])

                action, actor_info = self.agent.agent.actor.get_action(state)
                mean = actor_info['mean']
                logstd = actor_info['logstd']
                action = mean
                # action = np.clip(action, self.config.conf['actor-output-bounds'][0],
                #                  self.config.conf['actor-output-bounds'][1])
                action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)

                f = self.env.rejectableForce_xy(1.0 / self.network_freq)
                rgb=self.env._render(roll=0,pitch=0,yaw=0)
                print(rgb.shape)
                self.image_list.append(rgb)

                # action = self.control.rescale(ref_action, self.config.conf['action-bounds'],
                #                               self.config.conf['actor-output-bounds'])
                self.control.update_ref(ref_angle, ref_vel, [])
                next_state, reward, terminal, info = self.control.control_step(action, self.force, gait_phase)
                self.ref_motion.index_count()

                # if step>= 1*self.network_freq and step< 1.3*self.network_freq:
                #     self.force = [800,0,0]
                # else:
                #     self.force = [0,0,0]

                total_reward += reward

                ob = self.env.getObservation()
                ob_filtered = self.env.getFilteredObservation()
                # for l in range(len(ob)):
                #     self.logging.add_run('observation' + str(l), ob[l])
                #     self.logging.add_run('filtered_observation' + str(l), ob_filtered[l])
                self.logging.add_run('action', action)
                self.logging.add_run('ref_action', ref_angle)
                joint_angle = self.control.get_joint_angle()
                self.logging.add_run('joint_angle', joint_angle)
                readings = self.env.getExtendedReading()
                for key, value in readings.items():
                    self.logging.add_run(key, value)
                self.logging.add_run('task_reward', info['task_reward'])
                self.logging.add_run('imitation_reward', info['imitation_reward'])
                self.logging.add_run('total_reward', info['total_reward'])
                self.logging.add_run('reward_joint_angle', info['reward_joint_angle'])
                self.logging.add_run('reward_joint_vel', info['reward_joint_vel'])
                #
                # while 1:
                #     if(time.time()-t)>1.0/self.network_freq:
                #         break

                if terminal:
                    break
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
    # dir_path = 'PPO/record/FCNN/3D_walk_imitation/without_external_force_disturbance/2018_10_17_15.35.32'  # '2017_05_29_18.23.49/with_force'
    dir_path = 'PPO/record/3D_walk_imitation/without_external_force_disturbance/2019_01_11_14.38.17'  # '2017_05_29_18.23.49/with_force'
    test = Run(config, dir_path)
    test.test()

if __name__ == '__main__':
    main()
