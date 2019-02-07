import copy
import gc
import os
import time
from datetime import datetime

from TRPO.configuration import *
from agent.agent_TRPO import *
from common.control import *
from common.logger import logger
from common.replay_buffer import ReplayBuffer
from common.value_trace import *
from valkyrie_gym_env import Valkyrie
from common.motion_new import Motion

gc.enable()

class Train():
    def __init__(self, config):
        self.config = config

        self.PD_freq = self.config.conf['LLC-frequency']
        self.Physics_freq = self.config.conf['Physics-frequency']
        self.network_freq = self.config.conf['HLC-frequency']
        self.sampling_skip = int(self.PD_freq/self.network_freq)

        self.reward_decay = 1.0
        self.reward_scale = config.conf['reward-scale']
        self.reward_scale = self.reward_scale / float(self.sampling_skip)  # /10.0#normalizing reward to 1

        self.max_time_per_train_episode = self.config.conf['max-train-time']
        self.max_step_per_train_episode = int(self.max_time_per_train_episode*self.network_freq)
        self.max_time_per_test_episode = self.config.conf['max-test-time']#16
        self.max_step_per_test_episode = int(self.max_time_per_test_episode*self.network_freq)
        self.train_external_force_disturbance = False
        if self.train_external_force_disturbance == True:
            path_str = 'with_external_force_disturbance/'
        else:
            path_str = 'without_external_force_disturbance/'
        self.test_external_force_disturbance = False

        self.env = Valkyrie(
            max_time=self.max_time_per_train_episode, renders=False, initial_gap_time=0.1, PD_freq=self.PD_freq,
            Physics_freq=self.Physics_freq, Kp=config.conf['Kp'], Kd=config.conf['Kd'],
            bullet_default_PD=config.conf['bullet-default-PD'], controlled_joints_list=config.conf['controlled-joints'])

        config.conf['state-dim'] = self.env.stateNumber + 1# 2 more variables for cyclic motion
        self.agent = Agent(self.env, self.config)

        self.episode_count = 0
        self.step_count = 0
        self.train_iter_count = 0

        self.best_reward = 0
        self.best_episode = 0
        self.best_train_iter = 0

        self.control = Control(self.config, self.env)

        # load weight from previous network
        # dir_path = 'record/2017_12_04_15.20.44/no_force'  # '2017_05_29_18.23.49/with_force'

        # create new network
        dir_path = 'TRPO/record/' + '3D_walk_imitation/' + path_str + datetime.now().strftime('%Y_%m_%d_%H.%M.%S')
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

        self.on_policy_paths = []
        self.off_policy_paths = []
        self.buffer = ReplayBuffer(self.config.conf['replay-buffer-size'])

        self.force = [0,0,0]
        self.force_chest = [0, 0, 0]  # max(0,force_chest[1]-300*1.0 / EXPLORE)]
        self.force_pelvis = [0, 0, 0]

        self.ref_motion = Motion(config=self.config, dsr_gait_freq=0.6)

    def get_single_path(self):
        observations = []
        next_observations = []
        actions = []
        rewards = []
        actor_infos = []
        means = []
        logstds = []
        dones = []
        task_rewards = []
        imitation_rewards = []

        self.control.reset(w_imitation=self.config.conf['imitation-weight'],w_task=self.config.conf['task-weight'])
        self.ref_motion.reset(index=0)
        # self.ref_motion.reset()
        self.episode_count+=1
        self.ref_motion.random_count()
        q_nom = self.ref_motion.ref_motion_dict()
        base_orn_nom = self.ref_motion.get_base_orn()

        state = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'], q_nom=q_nom, base_orn_nom=base_orn_nom, base_pos_nom=[0,0,1.175],fixed_base=False)
        state  = np.squeeze(state)
        gait_phase = self.ref_motion.count / self.ref_motion.dsr_length
        state = np.append(state,[np.sin(np.pi*2*gait_phase), np.cos(np.pi*2*gait_phase)])
        # state = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'])
        for step in range(self.max_step_per_train_episode):
            self.step_count+=1
            if self.train_external_force_disturbance:
                # if step == self.network_freq or step == 6 * self.network_freq or step == 11 * self.network_freq:  # apply force for every 5 second
                #     f = np.random.normal(0, 0.2) * 600 * self.network_freq / 10
                #     theta = np.random.uniform(-math.pi, math.pi)
                #     fx = f * math.cos(theta)
                #     fy = f * math.sin(theta)
                #     self.force = [fx, fy, 0]
                if step == 0 or step == 5 * self.network_freq or step == 10 * self.network_freq:  # apply force for every 5 second
                    f = np.random.uniform(1500, 3000) #forward force to encourage forward movement
                    self.force = [f, 0, 0]
                else:
                    self.force = [0, 0, 0]
            else:
                self.force = [0, 0, 0]

            gait_phase = self.ref_motion.count / self.ref_motion.dsr_length
            ref_angle = self.ref_motion.ref_joint_angle()
            ref_vel = self.ref_motion.ref_joint_vel()

            state = self.env.getExtendedObservation()
            state = np.squeeze(state)
            # state = np.append(state, [np.sin(np.pi*2*gait_phase), np.cos(np.pi*2*gait_phase)])
            state = np.append(state, [np.pi * 2 * gait_phase])
            action, actor_info = self.agent.agent.actor.get_action(state)
            mean = actor_info['mean']
            logstd = actor_info['logstd']
            #action = np.clip(action, self.config.conf['actor-output-bounds'][0], self.config.conf['actor-output-bounds'][1])
            action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)

            self.control.update_ref(ref_angle, ref_vel, [])
            next_state, reward, terminal, info = self.control.control_step(action, self.force, gait_phase)
            #next_state, reward, terminal, info = self.control.control_step(action, self.force, ref_action)
            self.ref_motion.index_count()

            gait_phase = self.ref_motion.count / self.ref_motion.dsr_length
            next_state = np.squeeze(next_state)
            next_state = np.append(next_state, [np.pi * 2 * gait_phase])

            observations.append(state)
            actions.append(action)
            rewards.append(reward)
            actor_infos.append(actor_info)
            means.append(mean)
            logstds.append(logstd)
            dones.append(terminal)
            next_observations.append(next_state)
            task_rewards.append(np.array(info['task_reward']))
            imitation_rewards.append(np.array(info['imitation_reward']))
            if terminal:
                break
            state = next_state
        path = dict(observations=np.array(observations), actions=np.array(actions), rewards=np.array(rewards),
                    actor_infos=actor_infos, means=means, logstds=logstds, dones=dones,
                    next_observations=next_observations, task_rewards=task_rewards, imitation_rewards=imitation_rewards)
        return path

    def get_paths(self, num_of_paths=None, prefix='', verbose=True):
        if num_of_paths is None:
            num_of_paths = self.config.conf['max-path-num']
        paths = []
        t = time.time()
        if verbose:
            print(prefix + 'Gathering Samples')
        step_count = 0

        path_count = 0
        while(1):
            path = self.get_single_path()
            paths.append(path)
            step_count += len(path['dones'])
            path_count +=1
            num_of_paths = path_count
            if step_count>=self.config.conf['max-path-step']:
                break

        if verbose:
            print('%i paths sampled. %i steps sampled. %i total paths sampled. Total time used: %f.' % (num_of_paths, step_count, self.episode_count, time.time() - t))
        return paths

    def train_paths(self):
        self.train_iter_count+=1
        self.on_policy_paths = []  # clear
        self.off_policy_paths = [] #clear
        self.on_policy_paths = self.get_paths()
        # self.buffer.add_paths(self.on_policy_paths)
        self.off_policy_paths = copy.deepcopy(self.on_policy_paths)

        self.train_actor(True)#TODO
        self.train_critic(True)


    def train_critic(self, on_policy=True, prefix='', verbose = True):
        t = time.time()
        if on_policy == True:
            paths = copy.deepcopy(self.on_policy_paths)

            for path in paths:
                path['V_target'] = []
                path['Q_target'] = []

                rewards = path['rewards']
                next_observations= path['next_observations']
                dones = path['dones']


                r = np.array(rewards)

                if dones[-1] == False:  # not terminal state
                    r[-1] = r[-1] + self.config.conf['gamma']*self.agent.agent.V(next_observations[-1])  # bootstrap

                #path['returns'] = discount(path['rewards'], self.config.conf['gamma'])
                path['returns'] = discount(r, self.config.conf['gamma'])
                # print(discount(path['rewards'], self.config.conf['gamma']))
                # print(discount(r, self.config.conf['gamma']))
                # print(discount(path['rewards'], self.config.conf['gamma'])-discount(r, self.config.conf['gamma']))

            observations = np.concatenate([path['observations'] for path in paths])
            returns = np.concatenate([path['returns'] for path in paths])

            observations = np.vstack(observations)
            returns = np.vstack(returns)

            qloss = 0
            vloss = 0
            vloss += self.agent.train_V(observations, returns)
            print('qloss', qloss, 'vloss', vloss)

        else:
            paths = copy.deepcopy(self.off_policy_paths)
            #paths = self.off_policy_paths
            for path in paths:
                length = len(path['rewards'])

                path['V_target'] = []
                path['Q_target'] = []
                rewards = path['rewards']
                states = path['states']
                actions = path['actions']
                next_states = path['next_states']
                dones = path['dones']
                means = path['means']
                logstds = path['logstds']

                r = np.array(rewards)

                if dones[-1][0] == False:  # not terminal state
                    r[-1][0] = r[-1][0] + self.config.conf['gamma']*self.agent.agent.V(next_states[-1])  # bootstrap

                V_trace = self.agent.V_trace(states, actions, next_states, rewards, dones, means, logstds)

                path['V_target'] = V_trace
            V_target = np.concatenate([path['V_target'] for path in paths])
            states = np.concatenate([path['states'] for path in paths])
            # print(states)
            actions = np.concatenate([path['actions'] for path in paths])
            # print(actions)

            qloss = 0
            vloss = 0
            vloss += self.agent.train_V(states, V_target)
            print('qloss', qloss, 'vloss', vloss)

        if verbose:
            print(prefix + 'Training critic network. Total time used: %f.' % (time.time() - t))

        return

    def train_actor(self, on_policy=True, prefix='', verbose = True): # whether or not on policy
        t = time.time()
        stats = dict()
        if on_policy == True:
            paths = copy.deepcopy(self.on_policy_paths)
            length = len(paths)

            for path in paths:
                rewards = path['rewards']
                observations = path['observations']
                next_observations = path['next_observations']
                dones = path['dones']

                path['baselines'] = self.agent.agent.V(path['observations'])
                path['returns'] = discount(path['rewards'], self.config.conf['gamma'])
                if not self.config.conf['GAE']:
                    path['advantages'] = path['returns'] - path['baselines']
                else:
                    b = np.append(path['baselines'], path['baselines'][-1])
                    deltas = path['rewards'] + self.config.conf['gamma'] * b[1:] - b[:-1]
                    deltas[-1] = path['rewards'][-1] + (1-dones[-1])*self.config.conf['gamma']*b[-1]-b[-1]
                    #path['advantages'] = discount(deltas, self.config.conf['gamma'] * self.config.conf['lambda'])
                    path['advantages'] = np.squeeze(self.agent.GAE(observations, next_observations, rewards, dones))
                    # print(discount(deltas, self.config.conf['gamma'] * self.config.conf['lambda']))
                    # print(self.agent.GAE(observations, next_observations, rewards, dones))
                    # print(discount(deltas, self.config.conf['gamma'] * self.config.conf['lambda'])-np.squeeze(self.agent.GAE(observations, next_observations, rewards, dones)))

                if not self.config.conf['use-critic']:
                    r = np.array(rewards)
                    path['advantages'] = discount(r, self.config.conf['gamma'])

            observations = np.concatenate([path['observations'] for path in paths])
            actions = np.concatenate([path['actions'] for path in paths])
            rewards = np.concatenate([path['rewards'] for path in paths])
            advantages = np.concatenate([path['advantages'] for path in paths])
            actor_infos = np.concatenate([path['actor_infos'] for path in paths])
            means = np.concatenate([path['means'] for path in paths])
            logstds = np.concatenate([path['logstds'] for path in paths])
            returns = np.concatenate([path['returns'] for path in paths])

            if self.config.conf['center-advantage']:
                advantages -= np.mean(advantages)
                advantages /= (np.std(advantages) + 1e-8)

            # advantages = np.vstack(advantages)
            #advantages = advantages.reshape(length,1)

            self.agent.train_actor_TRPO(observations, actions, advantages, means, logstds)

        else:
            paths = self.off_policy_paths
            off_policy_states = np.concatenate([path['states'] for path in paths])
            off_policy_actions = np.concatenate([path['actions'] for path in paths])
            # (states)

            self.agent.train_actor_DPG(off_policy_states, off_policy_actions)

        if verbose:
            print(prefix + 'Training actor network. Total time used: %f.' % (time.time() - t))

        return stats

    def test(self):
        total_reward = 0
        total_task_reward = 0
        total_imitation_reward = 0
        for i in range(self.config.conf['test-num']):#
            _ = self.env._reset(Kp=self.config.conf['Kp'], Kd=self.config.conf['Kd'], base_pos_nom=[0,0,1.175],fixed_base=False)
            self.control.reset(w_imitation=self.config.conf['imitation-weight'], w_task=self.config.conf['task-weight'])
            self.ref_motion.reset(index=0)

            for step in range(self.max_step_per_test_episode):

                gait_phase = self.ref_motion.count / self.ref_motion.dsr_length
                ref_angle = self.ref_motion.ref_joint_angle()
                ref_vel = self.ref_motion.ref_joint_vel()

                state = self.env.getExtendedObservation()
                state = np.squeeze(state)
                state = np.append(state, [np.pi * 2 * gait_phase])
                action, actor_info = self.agent.agent.actor.get_action(state)
                mean = actor_info['mean']
                logstd = actor_info['logstd']
                action = mean
                #action = np.clip(action, self.config.conf['actor-output-bounds'][0], self.config.conf['actor-output-bounds'][1])
                action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)

                if self.test_external_force_disturbance:
                    f = self.env.rejectableForce_xy(1.0 / self.network_freq)
                    if step == 5 * self.network_freq:
                        if f[1] == 0:
                            self.force = np.array([600.0 * self.network_freq / 10.0, 0.0, 0.0])
                        else:
                            self.force = np.array([1.0 * f[1], 0, 0])
                        print(self.force)
                    elif step == 11 * self.network_freq:
                        if f[0] == 0:
                            self.force = np.array([-600.0 * self.network_freq / 10.0, 0.0, 0.0])
                        else:
                            self.force = [1.0 * f[0], 0, 0]
                        print(self.force)
                    elif step == 17 * self.network_freq:
                        if f[2] == 0:
                            self.force = np.array([0.0, -800.0 * self.network_freq / 10.0, 0.0])
                        else:
                            self.force = [0, 1.0 * f[2], 0]
                        print(self.force)
                    elif step == 23 * self.network_freq:
                        if f[3] == 0:
                            self.force = np.array([0.0, 800.0 * self.network_freq / 10.0, 0.0])
                        else:
                            self.force = [0, 1.0 * f[3], 0]
                        print(self.force)
                    else:
                        self.force = [0, 0, 0]
                else:
                    self.force = [0, 0, 0]

                self.control.update_ref(ref_angle, ref_vel, [])
                next_state, reward, terminal, info = self.control.control_step(action, self.force, gait_phase)
                # next_state, reward, done, info = self.control.control_step(action, self.force, ref_action)
                self.ref_motion.index_count()

                total_reward += reward
                total_task_reward += info['task_reward']
                total_imitation_reward += info['imitation_reward']
                if terminal:
                    break
            #self.env.stopRendering()
        ave_reward = total_reward/self.config.conf['test-num']
        ave_task_reward = total_task_reward/self.config.conf['test-num']
        ave_imitation_reward = total_imitation_reward/self.config.conf['test-num']
        self.agent.save_weight(self.step_count, self.dir_path+'/latest_network')
        self.agent.agent.save_actor_variable(self.step_count, self.dir_path + '/latest_network')
        self.agent.agent.save_critic_variable(self.step_count, self.dir_path + '/latest_network')
        if self.best_reward<ave_reward and self.episode_count>self.config.conf['record-start-size']:
            self.best_episode = self.episode_count
            self.best_train_iter = self.train_iter_count
            self.best_reward=ave_reward
            self.agent.save_weight(self.step_count, self.dir_path+'/best_network')
            self.agent.agent.save_actor_variable(self.step_count, self.dir_path+'/best_network')
            self.agent.agent.save_critic_variable(self.step_count, self.dir_path+'/best_network')

        episode_rewards = np.array([np.sum(path['rewards']) for path in self.on_policy_paths])
        episode_task_rewards = np.array([np.sum(path['task_rewards']) for path in self.on_policy_paths])
        episode_imitation_rewards = np.array([np.sum(path['imitation_rewards']) for path in self.on_policy_paths])

        print('iter:' + str(self.train_iter_count) + ' episode:' + str(self.episode_count) + ' step:' + str(self.step_count)
              + ' Deterministic policy return:' + str(ave_reward) + ' Average return:' + str(np.mean(episode_rewards)))
        print('best train_iter', self.best_train_iter, 'best reward', self.best_reward)
        self.logging.add_train('episode', self.episode_count)
        self.logging.add_train('step', self.step_count)
        self.logging.add_train('ave_reward', ave_reward)
        self.logging.add_train('ave_task_reward', ave_task_reward)
        self.logging.add_train('ave_imitation_reward', ave_imitation_reward)
        self.logging.add_train('logstd',np.squeeze(self.agent.agent.logstd()))
        self.logging.add_train('average_return', np.mean(episode_rewards))
        self.logging.add_train('task_rewards', np.mean(episode_task_rewards))
        self.logging.add_train('imitation_rewards', np.mean(episode_imitation_rewards))
        #self.logging.save_train()
        #self.agent.ob_normalize1.save_normalization(self.dir_path)  # TODO test observation normalization
        #self.agent.ob_normalize2.save_normalization(self.dir_path)  # TODO test observation normalization
        self.logging.save_train()


def main():
    config = Configuration()
    train = Train(config)
    while 1:
        train.train_paths()
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
        if train.step_count>config.conf['max-step-num']:
            break
    return

if __name__ == '__main__':
    main()
