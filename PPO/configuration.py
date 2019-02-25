import pickle
import numpy as np

class Configuration:
    def __init__(self):
        self.conf={}
        self.conf['env-id'] = 'HumanoidBalanceFilter-v0'
        self.conf['render-eval'] = False

        self.conf['joint-interpolation'] = True

        control_joint_num = 14
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
                "torsoYaw", "torsoPitch", "torsoRoll",
                "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])
        elif control_joint_num == 14:
            self.conf['controlled-joints'] = list([
                "torsoYaw", "torsoRoll",
                "rightHipYaw", "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipYaw", "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])
        elif control_joint_num == 15:
            self.conf['controlled-joints'] = list([
                "torsoYaw", "torsoPitch", "torsoRoll",
                "rightHipYaw", "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipYaw", "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])

        actor_joint_num = 14
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
                "torsoYaw", "torsoPitch", "torsoRoll",
                "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])
        elif actor_joint_num == 14:
            self.conf['actor-action-joints'] = list([
                "torsoYaw", "torsoRoll",
                "rightHipYaw", "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipYaw", "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])
        elif actor_joint_num == 15:
            self.conf['actor-action-joints'] = list([
                "torsoYaw", "torsoPitch", "torsoRoll",
                "rightHipYaw", "rightHipRoll", "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch", "rightAnkleRoll",
                "leftHipYaw", "leftHipRoll", "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", "leftAnkleRoll",])

        self.action_bounds = dict([
            ("torsoYaw", [-1.74533, 1.74533]),#[-1.329, 1.181]
            ("torsoPitch",[-0.13,0.67]),
            ("torsoRoll", [-0.8727, 0.6981]),#[-0.23, 0.255]
            ("rightHipYaw", [-1.5708, 0.8727]),
            ("rightHipRoll", [-0.8727, 0.6981]),
            ("rightHipPitch", [-2.0944, 1.0472]),
            ("rightKneePitch", [0.0, 2.4435]),
            ("rightAnklePitch", [-1.3963, 0.6981]),
            ("rightAnkleRoll", [-0.7854, 0.7854]),
            ("leftHipYaw", [-0.8727, 1.5708]),
            ("leftHipRoll", [-0.6981, 0.8727]),
            ("leftHipPitch", [-2.0944, 1.0472]),
            ("leftKneePitch", [-0.0, 2.4425]),
            ("leftAnklePitch", [-1.3963, 0.6981]),
            ("leftAnkleRoll", [-0.7854, 0.7854]),
            ])
        self.normalized_action_bounds = dict([
            ("torsoYaw", [-1.0, 1.0]),
            ("torsoPitch",[-0.325,1.675]),
            ("torsoRoll", [-1.0, 1.0]),
            ("rightHipYaw", [-1.453, 0.547]),
            ("rightHipRoll", [-1.083, 0.917]),
            ("rightHipPitch", [-1.1983, 0.8017]),
            ("rightKneePitch", [-0.0776, 1.9224]),
            ("rightAnklePitch", [-1.1772, 0.8228]),
            ("rightAnkleRoll", [-1.0, 1.0]),
            ("leftHipYaw", [-0.547, 1.453]),
            ("leftHipRoll", [-0.917, 1.083]),
            ("leftHipPitch", [-1.1983, 0.8017]),
            ("leftKneePitch", [-0.0776, 1.9224]),
            ("leftAnklePitch", [-1.1772, 0.8228]),
            ("leftAnkleRoll", [-1.0, 1.0]),
            ])#waist pitch,right hip pitch, right hip roll, right knee pitch,right ankle pitch, right ankle roll

        self.conf['state-dim'] = 60
        self.conf['action-dim'] = len(self.conf['actor-action-joints'])
        self.conf['action-bounds'] = np.zeros((2,len(self.conf['actor-action-joints'])))
        self.conf['normalized-action-bounds'] = np.zeros((2, len(self.conf['actor-action-joints'])))
        self.conf['actor-logstd-initial'] = np.zeros((1, len(self.conf['actor-action-joints'])))
        self.conf['actor-logstd-bounds'] = np.ones((2,len(self.conf['actor-action-joints'])))
        self.conf['actor-output-bounds'] = np.ones((2,len(self.conf['actor-action-joints'])))
        self.conf['actor-output-bounds'][0][:] = -1 * np.ones((len(self.conf['actor-action-joints']),))
        self.conf['actor-output-bounds'][1][:] = 1* np.ones((len(self.conf['actor-action-joints']),))
        # self.conf['actor-output-bounds'] = self.conf['action-bounds'] #directly ouput joint angles
        for i in range(len(self.conf['actor-action-joints'])):
            joint_name = self.conf['actor-action-joints'][i]
            self.conf['action-bounds'][0][i] = self.action_bounds[joint_name][0] # lower bound
            self.conf['action-bounds'][1][i] = self.action_bounds[joint_name][1] # upper bound

            self.conf['normalized-action-bounds'][0][i] = self.normalized_action_bounds[joint_name][0] # lower bound
            self.conf['normalized-action-bounds'][1][i] = self.normalized_action_bounds[joint_name][1] # upper bound
            std = (self.conf['actor-output-bounds'][1][i]-self.conf['actor-output-bounds'][0][i])
            # std = 2
            # self.conf['actor-logstd-initial'][0][i] = np.log(std*0.5/2)#0.5
            # self.conf['actor-logstd-bounds'][0][i] = np.log(std*0.2/2)
            # self.conf['actor-logstd-bounds'][1][i] = np.log(std*0.55/1.5)#0.6
            self.conf['actor-logstd-initial'][0][i] = np.log(std*0.25)#np.log(min(std*0.25, 1.0))#0.5
            self.conf['actor-logstd-bounds'][0][i] = np.log(std*0.1)
            self.conf['actor-logstd-bounds'][1][i] = np.log(std*0.25)#0.6

        self.conf['Physics-frequency'] =1000
        self.conf['LLC-frequency'] =500
        self.conf['HLC-frequency'] = 25
        self.conf['bullet-default-PD'] = False

        #valkyrie 137
        #walkman 94

        self.conf['Kp'] = dict([
            ("torsoYaw", 700),
            ("torsoPitch", 2000),
            ("torsoRoll", 2000),
            ("rightHipYaw", 700),
            ("rightHipRoll", 1000),  # 1500
            ("rightHipPitch", 1300),  # -0.49
            ("rightKneePitch", 1300),  # 1.205
            ("rightAnklePitch", 1300),  # -0.71
            ("rightAnkleRoll", 1000),  # 1000
            ("leftHipYaw", 700),
            ("leftHipRoll", 1000),  # 1500
            ("leftHipPitch", 1300),  # -0.491
            ("leftKneePitch", 1300),  # 1.205
            ("leftAnklePitch", 1300),  # -0.71
            ("leftAnkleRoll", 1000),  # 1000
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
            ("torsoYaw", 70),
            ("torsoPitch", 200),
            ("torsoRoll", 200),
            ("rightHipYaw", 70),
            ("rightHipRoll", 100),  # 150
            ("rightHipPitch", 120),  # 180
            ("rightKneePitch", 120),  # 120
            ("rightAnklePitch", 80),  # -0.71
            ("rightAnkleRoll", 60),  # 100
            ("leftHipYaw", 70),
            ("leftHipRoll", 100),  # 150
            ("leftHipPitch", 120),  # 180
            ("leftKneePitch", 120),  # 120
            ("leftAnklePitch", 80),  # -0.71
            ("leftAnkleRoll", 60),  # 100
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

        self.conf['gating-layer-size'] = [32,32]
        self.conf['gating-activation-fn'] = ['relu','relu','None']
        self.conf['gating-index'] = [
            11,12,13,14,15,16,#torso yaw pitch roll
            17,18,19,20,21,22,23,24,25,26,27,28,
            32,33,34,35,36,37,38,39,40,41,42,43,
            #53,54, #foot contact
            55,56 #phase
        ]
        self.conf['expert-index'] = [
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10,11,12,13,14,15,16,17,18,19,
            20,21,22,23,24,25,26,27,28,29,
            30,31,32,33,34,35,36,37,38,39,
            40,41,42,43,44,45,46,47,48,49,
            50,51,52,53,54,
        ]
        self.conf['expert-num'] = 4

        self.conf['critic-layer-norm'] = False
        self.conf['critic-observation-norm'] = False #use batch norm to normalize observationsmlpOKN123

        self.conf['critic-l2-reg'] = 1e-2
        self.conf['critic-lr'] = 1e-3
        self.conf['critic-layer-size'] = [256, 256]#[100,50,25]#[64, 64]#[400,400]
        self.conf['critic-activation-fn'] = ['relu', 'relu', 'None']#['relu', 'relu', 'relu', 'None']#['relu', 'relu', 'None']#['leaky_relu', 'leaky_relu', 'leaky_relu', 'None']
        self.conf['critic-dropout'] = False
        self.conf['critic-dropout-rate'] = 0.5
        self.conf['critic-opt-method'] = dict([
                                        ('name', 'none'),
                                        ('train-num', 1),#10
                                        ('critic-lr', 3e-4),#3e-3#3e-4
                                        ('critic-l2-reg', 0),#1e-5#1e-2
                                        ('weight-decay', 1e-6),#1e-6
                                        ('critic-l2', False),#False
                                    ])
        self.conf['use-critic'] = True
        self.conf['critic-batch-size'] = 64
        self.conf['critic-iteration'] = 2

        self.conf['actor-layer-norm'] = False
        self.conf['actor-observation-norm'] = False #use batch norm to normalize observations
        self.conf['actor-l2-reg'] = 1e-3
        self.conf['actor-lr'] = 3e-4
        self.conf['actor-layer-size'] = [256, 256]#[100,50,25]#[64, 64]#[400,400]
        self.conf['actor-activation-fn'] = ['relu', 'relu', 'None']#['relu', 'relu', 'relu', 'None']#['relu', 'relu', 'None']#['leaky_relu', 'leaky_relu', 'leaky_relu', 'None']
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
        self.conf['center-advantage'] = False
        self.conf['norm-advantage-clip'] = 4

        self.conf['IPG-method'] = dict(useCV = True, vu=0.2, DPG_flag=True)
        self.conf['QProp-method'] = ['adaptive','aggressive','conservative', 'none'][2]
        self.conf['PPO-method'] = dict([
                                    ('epsilon',0.2),
                                    ('actor-lr',3e-4),#1e-5 can not be too large
                                    ('actor-batch-size',64),#32#64
                                    ('actor-l2-reg', 0), # 1e-5#1e-2
                                    ('weight-decay', 1e-6),#1e-60
                                    ('epoch', 2)#too large will prevent PPO from learning, somewhat causes the ratio to exceed the clipping range
                                    ])
        self.conf['max-path-num'] = 20
        self.conf['max-path-step'] = 4096#512#1024

        self.conf['off-policy-update-num'] = 4#8
        self.conf['off-policy-critic-update-num'] = 4  # 8
        self.conf['off-policy-actor-update-num'] = 0  # 8

        self.conf['loss-entropy-coeff'] = 0.0#0.01
        self.conf['loss-symmetry-coeff'] = 0.1
        self.conf['loss-output-bound-coeff'] = 1

        self.conf['GAE'] = True
        self.conf['lambda'] = 0.95#0.95#0.9
        self.conf['tau'] = 0.001
        self.conf['gamma'] = 0.95#0.9
        self.conf['render'] = False

        self.conf['normalize-returns'] = False
        self.conf['normalize-observations'] = False
        self.conf['actor-logstd-grad'] = True

        self.conf['prioritized-exp-replay'] = True
        self.conf['replay-buffer-size'] = 5000 #episode
        self.conf['replay-start-size'] = 0
        self.conf['record-start-size'] = self.conf['replay-start-size']*1.0#1.5

        self.conf['reward-scale'] = 0.1#1.0
        self.conf['epoch-num'] = 500
        self.conf['epoch-step-num'] = 5000000
        self.conf['total-step-num'] = 2500000000
        self.conf['max-train-time'] = 30 #second
        self.conf['max-test-time'] = 30 #second
        self.conf['test-num'] = 1
        self.conf['rollout-step-num'] = 1
        self.conf['train-step-num'] = 1
        self.conf['max-episode-num'] = 5000000
        #1000
        self.conf['max-step-num'] = 2500000000#2500000

        self.conf['imitation-weight'] = 0.5
        self.conf['task-weight'] = 0.5

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
