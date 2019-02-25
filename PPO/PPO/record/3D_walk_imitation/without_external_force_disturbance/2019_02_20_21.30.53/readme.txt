tau: 0.001
replay-start-size: 0
critic-l2-reg: 0.01
actor-opt-method: {'cg-damping': 0.1, 'line-search-backtrack': 10, 'conjugate-gradient-iteration': 30, 'line-search': True, 'epsilon': 0.01}
GAE: True
max-episode-num: 5000000
normalized-action-bounds: [[-1.     -1.     -1.453  -1.083  -1.1983 -0.0776 -1.1772 -1.     -0.547
  -0.917  -1.1983 -0.0776 -1.1772 -1.    ]
 [ 1.      1.      0.547   0.917   0.8017  1.9224  0.8228  1.      1.453
   1.083   0.8017  1.9224  0.8228  1.    ]]
loss-entropy-coeff: 0.0
bullet-default-PD: False
off-policy-update-num: 4
state-dim: 56
loss-symmetry-coeff: 0.1
controlled-joints: ['torsoYaw', 'torsoRoll', 'rightHipYaw', 'rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll', 'leftHipYaw', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch', 'leftAnkleRoll']
lambda: 0.95
max-path-step: 4096
LLC-frequency: 500
critic-observation-norm: False
expert-num: 4
actor-action-joints: ['torsoYaw', 'torsoRoll', 'rightHipYaw', 'rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll', 'leftHipYaw', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch', 'leftAnkleRoll']
normalize-observations: False
gating-layer-size: [32, 32]
critic-dropout-rate: 0.5
train-step-num: 1
Physics-frequency: 1000
actor-lr: 0.0003
critic-dropout: False
critic-layer-norm: False
off-policy-critic-update-num: 4
actor-logstd-bounds: [[-1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791
  -1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791
  -1.60943791 -1.60943791]
 [-0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718]]
max-train-time: 60
gating-index: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 55, 56]
norm-advantage-clip: 4
task-weight: 0.5
gating-activation-fn: ['relu', 'relu', 'None']
HLC-frequency: 25
max-test-time: 60
center-advantage: False
imitation-weight: 0.5
actor-l2-reg: 0.001
reward-scale: 0.1
prioritized-exp-replay: True
replay-buffer-size: 5000
Kd: {'leftHipPitch': 120, 'rightShoulderRoll': 30, 'rightHipRoll': 100, 'leftAnklePitch': 80, 'leftHipRoll': 100, 'rightElbowPitch': 5, 'leftKneePitch': 120, 'torsoYaw': 70, 'rightHipPitch': 120, 'leftAnkleRoll': 60, 'torsoPitch': 200, 'rightAnkleRoll': 60, 'leftShoulderPitch': 10, 'leftShoulderYaw': 2, 'leftShoulderRoll': 30, 'rightShoulderPitch': 10, 'torsoRoll': 200, 'rightShoulderYaw': 2, 'rightKneePitch': 120, 'leftHipYaw': 70, 'rightAnklePitch': 80, 'leftElbowPitch': 5, 'rightHipYaw': 70}
actor-output-bounds: [[-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]
 [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
critic-opt-method: {'critic-l2-reg': 0, 'name': 'none', 'critic-lr': 0.0003, 'weight-decay': 1e-06, 'critic-l2': False, 'train-num': 1}
Kp: {'leftHipPitch': 1300, 'rightShoulderRoll': 1500, 'rightHipRoll': 1000, 'leftAnklePitch': 1300, 'leftHipRoll': 1000, 'rightElbowPitch': 200, 'leftKneePitch': 1300, 'torsoYaw': 700, 'rightHipPitch': 1300, 'leftAnkleRoll': 1000, 'torsoPitch': 2000, 'rightAnkleRoll': 1000, 'leftShoulderPitch': 700, 'leftShoulderYaw': 200, 'leftShoulderRoll': 1500, 'rightShoulderPitch': 700, 'torsoRoll': 2000, 'rightShoulderYaw': 200, 'rightKneePitch': 1300, 'leftHipYaw': 700, 'rightAnklePitch': 1300, 'leftElbowPitch': 200, 'rightHipYaw': 700}
actor-observation-norm: False
joint-interpolation: True
actor-layer-size: [256, 256]
total-step-num: 2500000000
normalize-returns: False
max-step-num: 2500000000
epoch-step-num: 5000000
loss-output-bound-coeff: 1
action-bounds: [[-1.74533 -0.8727  -1.5708  -0.8727  -2.0944   0.      -1.3963  -0.7854
  -0.8727  -0.6981  -2.0944  -0.      -1.3963  -0.7854 ]
 [ 1.74533  0.6981   0.8727   0.6981   1.0472   2.4435   0.6981   0.7854
   1.5708   0.8727   1.0472   2.4425   0.6981   0.7854 ]]
QProp-method: conservative
off-policy-actor-update-num: 0
env-id: HumanoidBalanceFilter-v0
critic-activation-fn: ['relu', 'relu', 'None']
render: False
max-path-num: 20
IPG-method: {'vu': 0.2, 'DPG_flag': True, 'useCV': True}
epoch-num: 500
expert-index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
test-num: 1
use-critic: True
actor-layer-norm: False
record-start-size: 0.0
action-dim: 14
gamma: 0.95
PPO-method: {'actor-lr': 0.0003, 'weight-decay': 1e-06, 'epsilon': 0.2, 'actor-batch-size': 64, 'epoch': 2, 'actor-l2-reg': 0}
actor-output-bound-method: grad-invert
critic-batch-size: 64
rollout-step-num: 1
batch-size: 256
render-eval: False
critic-lr: 0.001
actor-activation-fn: ['relu', 'relu', 'None']
actor-dropout-rate: 0.5
actor-logstd-grad: True
critic-layer-size: [256, 256]
critic-iteration: 2
actor-logstd-initial: [[-0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718]]
actor-dropout: False
