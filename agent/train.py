import copy
import gc
import os
import time
from datetime import datetime

import gym
import pybullet_envs
import pybullet

from agent.configuration import *
from agent.agent_supervise import *
from common.logger import logger

gc.enable()

class Train():
    def __init__(self, config):
        self.config = config

        env_name = 'Walker2DBulletEnv-v0'#'AntBulletEnv-v0'#'Walker2DBulletEnv-v0'#'HumanoidBulletEnv-v0'
        self.env = gym.make(env_name)

        self.config.conf['state-dim'] = 10
        self.config.conf['action-dim'] = 3

        self.agent = Agent(self.env, self.config)

        self.episode_count = 0
        self.step_count = 0
        self.train_iter_count = 0

        self.best_reward = 0
        self.best_episode = 0
        self.best_train_iter = 0

        # load weight from previous network
        # dir_path = 'record/2017_12_04_15.20.44/no_force'  # '2017_05_29_18.23.49/with_force'

        # create new network
        dir_path = 'TRPO/record/' + '3D/' +'/' + datetime.now().strftime('%Y_%m_%d_%H.%M.%S')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not os.path.exists(dir_path + '/saved_actor_networks'):
            os.makedirs(dir_path + '/saved_actor_networks')
        if not os.path.exists(dir_path + '/saved_critic_networks'):
            os.makedirs(dir_path + '/saved_critic_networks')
        self.logging = logger(dir_path)
        config.save_configuration(dir_path)
        config.record_configuration(dir_path)
        config.print_configuration()
        self.agent.load_weight(dir_path)
        self.dir_path = dir_path

        #load test data and training data
        self.train_data = [np.ones((10000,10)), np.zeros((10000,3))]
        self.test_data = [np.ones((10000,10)), np.zeros((10000,3))]

        self.min_test_MSE = 10000000000

    def train(self):
        self.episode_count += 1
        state = self.train_data[0]
        action = self.train_data[1]

        loss = self.agent.train_actor_supervised(state, action)

        self.logging.add_train('episode', self.episode_count)
        self.logging.add_train('train_loss', loss)

        pred_action = self.agent.agent.action_mean(state)
        MSE = np.mean(np.square(action-pred_action))

        self.logging.add_train('train_MSE', MSE)
        print('training data MSE', MSE)

    def test(self):
        state = self.test_data[0]
        action = self.test_data[1]
        pred_action = self.agent.agent.action_mean(state)
        MSE = np.mean(np.square(action-pred_action))
        self.logging.add_train('test_MSE', MSE)
        print('testing data MSE', MSE)
        self.logging.save_train()

        #save network with lowest test MSE
        if self.min_test_MSE>MSE:
            self.min_test_MSE = MSE
            self.agent.agent.save_network(self.episode_count, self.dir_path+'/best_network')
            self.agent.agent.save_actor_variable(self.episode_count, self.dir_path+'/best_network')
            self.agent.agent.save_critic_variable(self.episode_count, self.dir_path+'/best_network')
        # save latest network
        self.agent.agent.save_network(self.episode_count, self.dir_path+'/latest_network')
        self.agent.agent.save_actor_variable(self.episode_count, self.dir_path+'/latest_network')
        self.agent.agent.save_critic_variable(self.episode_count, self.dir_path+'/latest_network')

    def load(self):
        name = 's7ik3'
        file = open(name + '.txt', 'r')
        # print(file.read())
        # for line in file:
        #     for word in line.split():
        #         print(word)
        lines = []
        for line in file:
            lines.append(line.rstrip('\n'))

        state = []
        action = []
        for line in lines:
            datas = line.split() # list of data
            state.append(datas[0:11])
            action.append(datas[11:14])
        state = np.vstack(state)
        action = np.vstack(action)


        return

def main():
    config = Configuration()
    train = Train(config)
    while 1:
        train.train()
        #print(train.episode_count)
        # if train.episode_count == 10 or (train.episode_count%10 == 0 and train.step_count>train.config.conf['record-start-size']):
        #     train.test()
        # if train.episode_count <= 11 or (train.episode_count%10 == 0 and train.episode_count>train.config.conf['record-start-size']):
        #     train.test()
        # if train.episode_count <= 11 or (train.episode_count%10 == 0):
        #     train.test()
        train.test()
        if train.episode_count>config.conf['max-episode-num']:
            break

    return

if __name__ == '__main__':
    main()
