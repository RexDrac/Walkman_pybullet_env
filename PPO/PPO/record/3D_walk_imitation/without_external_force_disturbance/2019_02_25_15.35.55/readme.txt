actor-action-joints: ['torsoYaw', 'torsoRoll', 'rightHipYaw', 'rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll', 'leftHipYaw', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch', 'leftAnkleRoll']
max-episode-num: 5000000
actor-activation-fn: ['relu', 'relu', 'None']
GAE: True
gating-index: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 55, 56]
lambda: 0.95
controlled-joints: ['torsoYaw', 'torsoRoll', 'rightHipYaw', 'rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll', 'leftHipYaw', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch', 'leftAnkleRoll']
Kd: {'leftShoulderRoll': 30, 'rightShoulderPitch': 10, 'rightElbowPitch': 5, 'rightShoulderYaw': 2, 'torsoPitch': 200, 'rightHipRoll': 100, 'leftShoulderPitch': 10, 'leftAnklePitch': 80, 'leftShoulderYaw': 2, 'leftAnkleRoll': 60, 'rightAnklePitch': 80, 'rightShoulderRoll': 30, 'leftHipRoll': 100, 'rightHipYaw': 70, 'rightHipPitch': 120, 'leftKneePitch': 120, 'rightAnkleRoll': 60, 'leftHipPitch': 120, 'torsoYaw': 70, 'torsoRoll': 200, 'rightKneePitch': 120, 'leftHipYaw': 70, 'leftElbowPitch': 5}
train-step-num: 1
tau: 0.001
loss-symmetry-coeff: 0.1
actor-logstd-bounds: [[-1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791
  -1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791 -1.60943791
  -1.60943791 -1.60943791]
 [-0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718]]
actor-output-bounds: [[-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]
 [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
critic-dropout-rate: 0.5
critic-opt-method: {'critic-l2': False, 'weight-decay': 1e-06, 'critic-lr': 0.0003, 'train-num': 1, 'critic-l2-reg': 0, 'name': 'none'}
epoch-step-num: 5000000
render-eval: False
render: False
reward-scale: 0.1
task-weight: 0.5
critic-l2-reg: 0.01
actor-dropout: False
critic-layer-size: [256, 256]
QProp-method: conservative
LLC-frequency: 500
action-dim: 14
off-policy-critic-update-num: 4
actor-logstd-grad: True
critic-lr: 0.001
HLC-frequency: 25
off-policy-actor-update-num: 0
critic-activation-fn: ['relu', 'relu', 'None']
off-policy-update-num: 4
actor-layer-norm: False
PPO-method: {'weight-decay': 1e-06, 'actor-l2-reg': 0, 'epoch': 2, 'actor-lr': 0.0003, 'epsilon': 0.2, 'actor-batch-size': 64}
max-path-step: 4096
normalize-returns: False
actor-dropout-rate: 0.5
state-dim: 56
max-step-num: 2500000000
actor-output-bound-method: grad-invert
critic-iteration: 2
norm-advantage-clip: 4
total-step-num: 2500000000
action-bounds: [[-1.74533 -0.8727  -1.5708  -0.8727  -2.0944   0.      -1.3963  -0.7854
  -0.8727  -0.6981  -2.0944  -0.      -1.3963  -0.7854 ]
 [ 1.74533  0.6981   0.8727   0.6981   1.0472   2.4435   0.6981   0.7854
   1.5708   0.8727   1.0472   2.4425   0.6981   0.7854 ]]
max-train-time: 30
critic-layer-norm: False
critic-observation-norm: False
Physics-frequency: 1000
gating-activation-fn: ['relu', 'relu', 'None']
expert-index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
gating-layer-size: [32, 32]
epoch-num: 500
critic-batch-size: 64
replay-buffer-size: 5000
gamma: 0.95
use-critic: True
imitation-weight: 0.5
actor-opt-method: {'epsilon': 0.01, 'cg-damping': 0.1, 'line-search': True, 'line-search-backtrack': 10, 'conjugate-gradient-iteration': 30}
max-test-time: 30
joint-interpolation: True
replay-start-size: 0
env-id: HumanoidBalanceFilter-v0
test-num: 1
max-path-num: 20
actor-observation-norm: False
rollout-step-num: 1
actor-logstd-initial: [[-0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718 -0.69314718
  -0.69314718 -0.69314718]]
expert-num: 4
actor-layer-size: [256, 256]
Kp: {'leftShoulderRoll': 1500, 'rightShoulderPitch': 700, 'rightElbowPitch': 200, 'rightShoulderYaw': 200, 'torsoPitch': 2000, 'rightHipRoll': 1000, 'leftShoulderPitch': 700, 'leftAnklePitch': 1300, 'leftShoulderYaw': 200, 'leftAnkleRoll': 1000, 'rightAnklePitch': 1300, 'rightShoulderRoll': 1500, 'leftHipRoll': 1000, 'rightHipYaw': 700, 'rightHipPitch': 1300, 'leftKneePitch': 1300, 'rightAnkleRoll': 1000, 'leftHipPitch': 1300, 'torsoYaw': 700, 'torsoRoll': 2000, 'rightKneePitch': 1300, 'leftHipYaw': 700, 'leftElbowPitch': 200}
center-advantage: False
loss-entropy-coeff: 0.0
loss-output-bound-coeff: 1
batch-size: 256
actor-l2-reg: 0.001
bullet-default-PD: False
actor-lr: 0.0003
critic-dropout: False
prioritized-exp-replay: True
IPG-method: {'vu': 0.2, 'DPG_flag': True, 'useCV': True}
normalize-observations: False
normalized-action-bounds: [[-1.     -1.     -1.453  -1.083  -1.1983 -0.0776 -1.1772 -1.     -0.547
  -0.917  -1.1983 -0.0776 -1.1772 -1.    ]
 [ 1.      1.      0.547   0.917   0.8017  1.9224  0.8228  1.      1.453
   1.083   0.8017  1.9224  0.8228  1.    ]]
record-start-size: 0.0
