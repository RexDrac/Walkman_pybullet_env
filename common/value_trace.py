import numpy as np
from common.util import *

def Q_monte(state, action, next_state, reward, done, agent, config):
    r = reward + (1 - done) * agent.Q_mu(next_state)
    Q = discount(r, config.conf['gamma'])
    return Q


def Q_retrace(state, action, next_state, reward, done, mean, logstd, agent, config):
    length = len(done)
    mean_beh = mean
    logstd_beh = logstd

    mean_pi = agent.action(state, False)
    logstd_pi = np.tile(agent.logstd(), (length, 1))

    log_prob_beh = log_likelihood(action, mean_beh, logstd_beh)
    log_prob_pi = log_likelihood(action, mean_pi, logstd_pi)
    rho = np.exp(log_prob_pi - log_prob_beh)
    rho = np.nan_to_num(rho)
    rho = np.clip(rho, 0.0, 1.0)
    rho = np.vstack(rho)

    # rho = rho[np.isnan(rho)] = 0.0
    # rho = rho[np.isinf(rho)] = 0.0
    c = np.minimum(np.ones(np.shape(rho)), rho) * config.conf['lambda']

    v = agent.Q_pi_Est(state)  # self.agent.V(state)
    q_value = agent.Q(state, action)
    Q_target = np.zeros(np.shape(reward))

    if done[length - 1][0]:  # terminal state
        Q_ret = 0
    else:
        s = next_state[-1][:]
        Q_ret = agent.Q_pi_Est(s)  # self.agent.V(s)

    # print(c)
    for i in range(length - 1, -1, -1):
        Q_ret = reward[i][0] + config.conf['gamma'] * Q_ret
        Q_target[i][0] = Q_ret
        Q_ret = c[i][0] * (Q_ret - q_value[i][0]) + v[i][0]
    Q_target = np.vstack(Q_target)
    return Q_target


def Q_opc(state, action, next_state, reward, done, agent, config):
    length = len(done)
    v = agent.Q_pi_Est(state)  # self.agent.V(state)
    q_value = agent.Q(state, action)
    Q_target = np.zeros(np.shape(reward))
    if done[length - 1][0]:
        Q_opc = 0
    else:
        s = next_state[-1][:]
        Q_opc = agent.Q_pi_Est(s)  # self.agent.V(s)

    for i in range(length - 1, -1, -1):
        Q_opc = reward[i][0] + config.conf['gamma'] * Q_opc
        Q_target[i][0] = Q_opc
        Q_opc = (Q_opc - q_value[i][0]) + v[i][0]

    Q_target = np.vstack(Q_target)
    return Q_target


def Q_TD(state, action, next_state, next_action, reward, done, agent, config):
    length = len(done)

    q_value = agent.Q(next_state, next_action)
    Q_target = []
    for i in range(length):
        if done[i][0]:
            Q_target.append(reward[i][0])
        else:
            Q_target.append(reward[i][0] + config.conf['gamma'] * q_value[i][0])
    Q_target = np.vstack(Q_target)  # np.resize(y_batch, [self.batch_size, 1])
    return Q_target


def Q_nstep(state, action, next_state, reward, done, agent, config):
    n = 10
    length = len(reward)
    Q_target = []
    for i in range(length):
        start = i
        end = min(start + n, length)
        state_batch = state[start:end]
        action_batch = action[start:end]
        next_state_batch = next_state[start:end]
        reward_batch = reward[start:end]
        done_batch = done[start:end]

        if done[end - 1][0] == False:  # not terminal state
            Q = agent.Q_mu(next_state[end - 1][:])
        else:
            Q = 0

        for j in range(end - start - 1, -1, -1):  # [n-1,0]
            Q = reward_batch[j][0] + config.conf['gamma'] * Q
        # temp = discount(self.gamma, reward_batch)
        # if done[end-1][0] == False: # not terminal state
        #     Q = temp[0][0] + self.agent.Q_mu(next_state[end-1][:])
        # else:
        #     Q = temp[0][0]

        Q_target.append(Q)
    Q_target = np.vstack(Q_target)
    return Q_target


def Q_lambda(state, action, next_state, reward, done, agent, config):
    length = len(reward)
    lam = 0.5
    delta = reward + config.conf['gamma'] * agent.Q_mu(next_state) - agent.Q(state, action)
    if done[length - 1][0] == True:
        delta[length - 1][0] = reward - agent.Q(state, action)

    temp = discount(delta, config.conf['gamma'] * config.conf['lambda'])
    Q_target = temp + agent.Q(state, action)
    return Q_target


def V_monte(next_state, reward, done, agent, config):
    reward = np.array(reward)
    if done[-1][0] == False:  # not terminal state
        reward[-1][0] += agent.V(next_state[-1])  # bootstrap if not terminal state
    V = discount(reward, config.conf['gamma'])
    return V


def V_correction(state, action, Q_retrace, mean, logstd, agent, config):
    length = len(Q_retrace)
    mean_beh = mean
    logstd_beh = logstd

    mean_pi = agent.action(state, False)
    logstd_pi = np.tile(agent.logstd(), (length, 1))

    log_prob_beh = log_likelihood(action, mean_beh, logstd_beh)
    log_prob_pi = log_likelihood(action, mean_pi, logstd_pi)
    rho = np.exp(log_prob_pi - log_prob_beh)
    rho = np.nan_to_num(rho)
    rho = np.clip(rho, 0.0, 1.0)
    rho = np.vstack(rho)

    # rho = rho[np.isnan(rho)] = 0.0
    # rho = rho[np.isinf(rho)] = 0.0
    c = np.minimum(np.ones(np.shape(rho)), rho)
    # print(c)
    V_target = c * (Q_retrace - agent.Q(state, action)) + agent.V(state)
    return V_target


def V_trace(state, action, next_state, reward, done, mean, logstd, agent, config):
    # IMPALA: Scalable Deep-RL with Importance Weighted Actor-Learner Rchitectures
    rho_thresh = 1.0  # importance sampling
    c_thresh = 1.0  # trace cutting
    length = len(done)
    mean_beh = mean
    logstd_beh = logstd

    mean_pi = agent.action(state, False)
    logstd_pi = np.tile(agent.logstd(), (length, 1))

    log_prob_beh = log_likelihood(action, mean_beh, logstd_beh)
    log_prob_pi = log_likelihood(action, mean_pi, logstd_pi)
    rho = np.exp(log_prob_pi - log_prob_beh)
    rho = np.nan_to_num(rho)
    rho = np.clip(rho, 0.0, 10.0)
    rho = np.vstack(rho)

    # rho = rho[np.isnan(rho)] = 0.0
    # rho = rho[np.isinf(rho)] = 0.0
    rho = np.minimum(np.ones(np.shape(rho)) * rho_thresh, rho)
    c = np.minimum(np.ones(np.shape(rho)) * c_thresh, rho) * config.conf['lambda']

    v = agent.V(state)  # self.agent.Q_pi_Est(state) #self.agent.V(state)
    delta = reward + config.conf['gamma'] * agent.V(next_state) - agent.V(state)
    if done[length - 1][0] == True:  # terminal state
        s = next_state[length - 1][:]
        delta[length - 1][0] = reward[length - 1][0] - agent.V(s)
    delta = rho * delta

    V_target = np.zeros(np.shape(reward))

    if done[length - 1][0]:  # terminal state
        V_trace = 0
    else:
        s = next_state[-1][:]
        V_trace = agent.V(s)  # self.agent.Q_pi_Est(s)#self.agent.V(s)

    # print(c)
    for i in range(length - 1, -1, -1):
        s = next_state[i][:]
        temp = V_trace - agent.V(s)

        V_trace = v[i][0] + delta[i][0] + config.conf['gamma'] * c[i][0] * temp
        V_target[i][0] = V_trace

    V_target = np.vstack(V_target)
    return V_target