import pickle
import numpy as np

class Configuration:
    def __init__(self):
        self.conf={}
        self.conf['env-id'] = 'HumanoidBalanceFilter-v0'
        self.conf['render-eval'] = False

        self.conf['joint-interpolation'] = True

        control_joint_num = 11
        if control_joint_num == 7:
            self.conf['controlled-joints'] = list([
                "torsoPitch",
                "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch",
                "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", ])
        elif control_joint_num == 11:
            self.conf['controlled-joints'] = list([
                "torsoPitch",
                "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])
        elif control_joint_num == 12:
            self.conf['controlled-joints'] = list([
                "torsoPitch", "torsoRoll",
                "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])
        elif control_joint_num == 13:
            self.conf['controlled-joints'] = list([
                "torsoPitch",
                "rightHipYaw", "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipYaw", "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])

        actor_joint_num = 11
        if actor_joint_num == 4:
            self.conf['actor-action-joints'] = list([
                "torsoPitch",
                "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch"])
        elif actor_joint_num == 11:
            self.conf['actor-action-joints'] = list([
                "torsoPitch",
                "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])
        elif actor_joint_num == 12:
            self.conf['actor-action-joints'] = list([
                "torsoPitch", "torsoRoll",
                "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])
        elif actor_joint_num == 13:
            self.conf['actor-action-joints'] = list([
                "torsoPitch",
                "rightHipYaw", "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipYaw", "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])

        self.action_bounds = dict([
            ("torsoPitch",[-0.13,0.67]),
            ("torsoRoll", [-0.23, 0.255]),
            ("rightHipYaw", [-1.1, 0.4141]),
            ("rightHipRoll", [-0.5515, 0.467]),
            ("rightHipPitch", [-2.42, 1.619]),
            ("rightKneePitch", [-0.083, 2.057]),
            ("rightAnklePitch", [-0.93, 0.65]),
            ("rightAnkleRoll", [-0.4, 0.4]),
            ("leftHipYaw", [-0.4141, 1.1]),
            ("leftHipRoll", [-0.467, 0.5515]),
            ("leftHipPitch", [-2.42, 1.619]),
            ("leftKneePitch", [-0.083, 2.057]),
            ("leftAnklePitch", [-0.93, 0.65]),
            ("leftAnkleRoll", [-0.4, 0.4]),
            ])
        self.normalized_action_bounds = dict([
            ("torsoPitch",[-0.325,1.675]),
            ("torsoRoll", [-0.4742, 0.5258]),
            ("rightHipYaw", [-0.7265, 0.2735]),
            ("rightHipRoll", [-1.083, 0.917]),
            ("rightHipPitch", [-1.1983, 0.8017]),
            ("rightKneePitch", [-0.0776, 1.9224]),
            ("rightAnklePitch", [-1.1772, 0.8228]),
            ("rightAnkleRoll", [-1.0, 1.0]),
            ("leftHipYaw", [-0.2735, 0.7265]),
            ("leftHipRoll", [-0.917, 1.083]),
            ("leftHipPitch", [-1.1983, 0.8017]),
            ("leftKneePitch", [-0.0776, 1.9224]),
            ("leftAnklePitch", [-1.1772, 0.8228]),
            ("leftAnkleRoll", [-1.0, 1.0]),
            ])#waist pitch,right hip pitch, right hip roll, right knee pitch,right ankle pitch, right ankle roll

        self.conf['state-dim'] = 60
        self.conf['action-dim'] = 3#len(self.conf['actor-action-joints'])
        self.conf['action-bounds'] = np.zeros((2,self.conf['action-dim']))
        self.conf['normalized-action-bounds'] = np.zeros((2, self.conf['action-dim']))
        self.conf['actor-logstd-initial'] = np.zeros((1, self.conf['action-dim']))
        self.conf['actor-logstd-bounds'] = np.ones((2,self.conf['action-dim']))
        self.conf['actor-output-bounds'] = np.ones((2,self.conf['action-dim']))
        self.conf['actor-output-bounds'][0][:] = -1 * np.ones((self.conf['action-dim'],))
        self.conf['actor-output-bounds'][1][:] = 1* np.ones((self.conf['action-dim'],))
        #self.conf['actor-output-bounds'] = self.conf['action-bounds']
        for i in range(self.conf['action-dim']):
            # joint_name = self.conf['actor-action-joints'][i]
            # self.conf['action-bounds'][0][i] = self.action_bounds[joint_name][0] # lower bound
            # self.conf['action-bounds'][1][i] = self.action_bounds[joint_name][1] # upper bound
            #
            # self.conf['normalized-action-bounds'][0][i] = self.normalized_action_bounds[joint_name][0] # lower bound
            # self.conf['normalized-action-bounds'][1][i] = self.normalized_action_bounds[joint_name][1] # upper bound
            # # std = (self.action_bounds[joint_name][1]-self.action_bounds[joint_name][0])
            # std = (self.conf['actor-output-bounds'][1][i]-self.conf['actor-output-bounds'][0][i])
            # # self.conf['actor-logstd-initial'][0][i] = np.log(std*0.5/2)#0.5
            # # self.conf['actor-logstd-bounds'][0][i] = np.log(std*0.2/2)
            # # self.conf['actor-logstd-bounds'][1][i] = np.log(std*0.55/1.5)#0.6
            std = 1

            self.conf['actor-logstd-initial'][0][i] = np.log(std*0.2)#np.log(min(std*0.25, 1.0))#0.5
            self.conf['actor-logstd-bounds'][0][i] = np.log(std*0.01)
            self.conf['actor-logstd-bounds'][1][i] = np.log(std*1.0)#0.6

        self.conf['Physics-frequency'] =1000
        self.conf['LLC-frequency'] =500
        self.conf['HLC-frequency'] = 25
        self.conf['bullet-default-PD'] = False

        self.conf['Kp'] = dict([
            ("torsoYaw", 4500),
            ("torsoPitch", 3000),
            ("torsoRoll", 4500),
            ("rightHipYaw", 500),
            ("rightHipRoll", 1500),  # 1500
            ("rightHipPitch", 2000),  # -0.49
            ("rightKneePitch", 2000),  # 1.205
            ("rightAnklePitch", 2000),  # -0.71
            ("rightAnkleRoll", 1500),  # 1000
            ("leftHipYaw", 500),
            ("leftHipRoll", 1500),  # 1500
            ("leftHipPitch", 2000),  # -0.491
            ("leftKneePitch", 2000),  # 1.205
            ("leftAnklePitch", 2000),  # -0.71
            ("leftAnkleRoll", 1500),  # 1000
            ("rightShoulderPitch", 700),
            ("rightShoulderRoll", 1500),
            ("rightShoulderYaw", 200),
            ("rightElbowPitch", 200),
            ("leftShoulderPitch", 700),
            ("leftShoulderRoll", 1500),
            ("leftShoulderYaw", 200),
            ("leftElbowPitch", 200),
                           ])

        self.conf['Kd'] = dict([
            ("torsoYaw", 30),
            ("torsoPitch", 300),
            ("torsoRoll", 30),
            ("rightHipYaw", 50),
            ("rightHipRoll", 150),  # 150
            ("rightHipPitch", 180),  # 180
            ("rightKneePitch", 180),  # 120
            ("rightAnklePitch", 120),  # -0.71
            ("rightAnkleRoll", 120),  # 100
            ("leftHipYaw", 50),
            ("leftHipRoll", 150),  # 150
            ("leftHipPitch", 180),  # 180
            ("leftKneePitch", 180),  # 120
            ("leftAnklePitch", 120),  # -0.71
            ("leftAnkleRoll", 120),  # 100
            ("rightShoulderPitch", 10),
            ("rightShoulderRoll", 30),
            ("rightShoulderYaw", 2),
            ("rightElbowPitch", 5),
            ("leftShoulderPitch", 10),
            ("leftShoulderRoll", 30),
            ("leftShoulderYaw", 2),
            ("leftElbowPitch", 5),
                           ])

        self.conf['batch-size'] = 256#32

        self.conf['critic-layer-norm'] = False
        self.conf['critic-observation-norm'] = False #use batch norm to normalize observations
        self.conf['critic-l2-reg'] = 1e-2
        self.conf['critic-lr'] = 1e-3
        self.conf['critic-layer-size'] = [100,50,25]#[400,400]
        self.conf['critic-activation-fn'] = ['relu', 'relu', 'relu', 'None']
        self.conf['critic-dropout'] = False
        self.conf['critic-dropout-rate'] = 0.5
        self.conf['critic-opt-method'] = dict([
                                        ('name', 'none'),
                                        ('train-num', 1),#10
                                        ('critic-lr', 3e-4),#3e-3#3e-4
                                        ('critic-l2-reg', 1e-3),#1e-5#1e-2
                                        ('critic-l2', False),#False
                                    ])
        self.conf['use-critic'] = True
        self.conf['critic-batch-size'] = 256
        self.conf['critic-iteration'] = 40

        self.conf['actor-layer-norm'] = False
        self.conf['actor-observation-norm'] = False #use batch norm to normalize observations
        self.conf['actor-l2-reg'] = 1e-2
        self.conf['actor-lr'] = 1e-4
        self.conf['actor-layer-size'] = [100,50,25]#[400,400]
        self.conf['actor-activation-fn'] = ['tanh', 'tanh', 'tanh', 'None']
        self.conf['actor-dropout'] = False
        self.conf['actor-dropout-rate'] = 0.5
        self.conf['actor-output-bound-method'] = 'grad-invert'#'tanh'
        self.conf['actor-opt-method'] = dict([
                                                ('epsilon', 0.01),#0.1
                                                ('line-search', True), #True #TODO check seems to have huge effect
                                                ('conjugate-gradient-iteration', 30),
                                                ('line-search-backtrack', 10),
                                                ('cg-damping', 0.1),
                                                   ])

        self.conf['actor-iteration'] = 40
        self.conf['actor-batch-size'] = 256

        self.conf['center-advantage'] = False

        self.conf['IPG-method'] = dict(useCV = True, vu=0.2, DPG_flag=True)
        self.conf['QProp-method'] = ['adaptive','aggressive','conservative', 'none'][2]
        self.conf['PPO-method'] = dict([
                                    ('epsilon',0.2),
                                    ('actor-lr',3e-4),#1e-5 can not be too large
                                    ('actor-batch-size',256),#32#64
                                    ('epoch', 40)#too large will prevent PPO from learning, somewhat causes the ratio to exceed the clipping range
                                    ])
        self.conf['max-path-num'] = 20
        self.conf['max-path-step'] = 4096#512#1024

        self.conf['off-policy-update-num'] = 4#8
        self.conf['off-policy-critic-update-num'] = 4  # 8
        self.conf['off-policy-actor-update-num'] = 0  # 8

        self.conf['loss-entropy-coeff'] = 0.01
        self.conf['loss-symmetry-coeff'] = 0.1
        self.conf['loss-output-bound-coeff'] = 0.01

        self.conf['GAE'] = True
        self.conf['lambda'] = 0.95#0.95#0.95#0.9
        self.conf['tau'] = 0.001
        self.conf['gamma'] = 0.95#0.95#0.9
        self.conf['render'] = False

        self.conf['normalize-returns'] = False
        self.conf['normalize-observations'] = False

        self.conf['prioritized-exp-replay'] = True
        self.conf['replay-buffer-size'] = 5000 #episode
        self.conf['replay-start-size'] = 500
        self.conf['record-start-size'] = self.conf['replay-start-size']*1.0#1.5

        self.conf['reward-scale'] = 0.1#1.0
        self.conf['epoch-num'] = 5000
        self.conf['epoch-step-num'] = 5000000
        self.conf['total-step-num'] = 2500000000
        self.conf['max-train-time'] = 16 #second
        self.conf['max-test-time'] = 30 #second
        self.conf['test-num'] = 1
        self.conf['rollout-step-num'] = 1
        self.conf['train-step-num'] = 1
        self.conf['max-episode-num'] = 5000000#1000
        self.conf['max-step-num'] = 2500000000#2500000


    def save_configuration(self,dir):
        # write python dict to a file
        output = open(dir + '/configuration.obj', 'wb')
        pickle.dump(self.conf, output)
        output.close()

    def load_configuration(self,dir):
        # write python dict to a file
        pkl_file = open(dir + '/configuration.obj', 'rb')
        conf_temp = pickle.load(pkl_file)
        self.conf.update(conf_temp)
        pkl_file.close()

    def record_configuration(self,dir):
        # write python dict to a file
        output = open(dir + '/readme.txt', 'w')
        for key in self.conf:
            output.write("{}: {}\n".format(key,self.conf[key]))

    def print_configuration(self):
        for key in self.conf:
            print(key + ': ' + str(self.conf[key]))
