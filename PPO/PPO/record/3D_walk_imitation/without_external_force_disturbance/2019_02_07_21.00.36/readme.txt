off-policy-actor-update-num: 0
actor-layer-size: [256, 256]
epoch-step-num: 5000000
action-bounds: [[-1.181  -0.23   -1.1    -0.5515 -2.42   -0.083  -0.93   -0.4    -0.4141
  -0.467  -2.42   -0.083  -0.93   -0.4   ]
 [ 1.181   0.23    0.4141  0.467   1.619   2.057   0.65    0.4     1.1
   0.5515  1.619   2.057   0.65    0.4   ]]
critic-dropout-rate: 0.5
imitation-weight: 0.5
lambda: 0.95
expert-index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
record-start-size: 0.0
actor-observation-norm: False
normalize-observations: False
use-critic: True
batch-size: 256
epoch-num: 500
GAE: True
gamma: 0.95
critic-lr: 0.001
bullet-default-PD: False
replay-buffer-size: 5000
normalize-returns: False
actor-lr: 0.0003
critic-l2-reg: 0.01
max-path-num: 20
test-num: 1
Physics-frequency: 1000
max-step-num: 2500000000
actor-logstd-grad: True
actor-activation-fn: ['relu', 'relu', 'None']
actor-l2-reg: 0.001
train-step-num: 1
critic-iteration: 2
reward-scale: 0.1
Kd: {'torsoPitch': 300, 'leftHipYaw': 100, 'torsoYaw': 100, 'rightShoulderYaw': 2, 'leftAnklePitch': 120, 'rightShoulderRoll': 30, 'leftElbowPitch': 5, 'torsoRoll': 300, 'rightAnklePitch': 120, 'leftAnkleRoll': 90, 'leftShoulderYaw': 2, 'rightHipYaw': 100, 'rightAnkleRoll': 90, 'leftShoulderRoll': 30, 'rightElbowPitch': 5, 'leftKneePitch': 180, 'rightShoulderPitch': 10, 'rightKneePitch': 180, 'leftHipPitch': 180, 'leftHipRoll': 150, 'leftShoulderPitch': 10, 'rightHipPitch': 180, 'rightHipRoll': 150}
rollout-step-num: 1
state-dim: 40
actor-logstd-initial: [[-0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718]]
expert-num: 4
render-eval: False
actor-opt-method: {'epsilon': 0.01, 'line-search-backtrack': 10, 'cg-damping': 0.1, 'conjugate-gradient-iteration': 30, 'line-search': True}
actor-action-joints: ['torsoYaw', 'torsoRoll', 'rightHipYaw', 'rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll', 'leftHipYaw', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch', 'leftAnkleRoll']
joint-interpolation: True
action-dim: 14
actor-output-bounds: [[-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]
 [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
render: False
replay-start-size: 0
LLC-frequency: 500
critic-batch-size: 64
off-policy-update-num: 4
gating-activation-fn: ['relu', 'relu', 'None']
loss-symmetry-coeff: 0.1
HLC-frequency: 25
actor-output-bound-method: grad-invert
Kp: {'torsoPitch': 3000, 'leftHipYaw': 1000, 'torsoYaw': 1000, 'rightShoulderYaw': 200, 'leftAnklePitch': 2000, 'rightShoulderRoll': 1500, 'leftElbowPitch': 200, 'torsoRoll': 3000, 'rightAnklePitch': 2000, 'leftAnkleRoll': 1500, 'leftShoulderYaw': 200, 'rightHipYaw': 1000, 'rightAnkleRoll': 1500, 'leftShoulderRoll': 1500, 'rightElbowPitch': 200, 'leftKneePitch': 2000, 'rightShoulderPitch': 700, 'rightKneePitch': 2000, 'leftHipPitch': 2000, 'leftHipRoll': 1500, 'leftShoulderPitch': 700, 'rightHipPitch': 2000, 'rightHipRoll': 1500}
gating-layer-size: [32, 32]
normalized-action-bounds: [[-1.     -1.     -1.453  -1.083  -1.1983 -0.0776 -1.1772 -1.     -0.547
  -0.917  -1.1983 -0.0776 -1.1772 -1.    ]
 [ 1.      1.      0.547   0.917   0.8017  1.9224  0.8228  1.      1.453
   1.083   0.8017  1.9224  0.8228  1.    ]]
critic-observation-norm: False
center-advantage: False
QProp-method: conservative
critic-layer-norm: False
loss-output-bound-coeff: 1
actor-layer-norm: False
PPO-method: {'actor-l2-reg': 0, 'weight-decay': 1e-06, 'actor-lr': 0.0003, 'epsilon': 0.2, 'actor-batch-size': 64, 'epoch': 2}
critic-layer-size: [256, 256]
actor-dropout-rate: 0.5
controlled-joints: ['torsoYaw', 'torsoRoll', 'rightHipYaw', 'rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll', 'leftHipYaw', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch', 'leftAnkleRoll']
norm-advantage-clip: 4
actor-logstd-bounds: [[-1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791
  -1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791
  -1.60943791 -1.60943791]
 [-0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718]]
IPG-method: {'useCV': True, 'DPG_flag': True, 'vu': 0.2}
loss-entropy-coeff: 0.0
off-policy-critic-update-num: 4
max-episode-num: 5000000
env-id: HumanoidBalanceFilter-v0
prioritized-exp-replay: True
critic-dropout: False
max-path-step: 4096
max-train-time: 60
critic-opt-method: {'critic-lr': 0.0003, 'critic-l2': False, 'weight-decay': 1e-06, 'name': 'none', 'critic-l2-reg': 0, 'train-num': 1}
critic-activation-fn: ['relu', 'relu', 'None']
total-step-num: 2500000000
gating-index: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 55, 56]
actor-dropout: False
task-weight: 0.5
tau: 0.001
max-test-time: 60
