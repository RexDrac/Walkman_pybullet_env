from common.motion import Motion
import numpy as np
import matplotlib.pyplot as plt
from TRPO.configuration import Configuration
config = Configuration()
motion = Motion(config)
#
# for key in motion.data_type[1:-1]:
#     data = motion.dsr_motion_data[key]
#     plt.plot(data, label=key)
#     plt.legend()
# plt.show()

for key in motion.joint_list:
    # if key == 'leftKneePitch':
    #     data = motion.joint_angles_list[6][key]
    #     plt.plot(data, label=key)
    #     plt.legend()
    # if key == 'leftHipPitch':
    #     data = motion.joint_angles_list[6][key]
    #     plt.plot(data, label=key)
    #     plt.legend()
    # if key == 'torsoPitch':
    #     data = motion.joint_angles_list[6][key]
    #     plt.plot(data, label=key)
    #     plt.legend()
    data = motion.joint_angles_list[5][key]
    plt.plot(data, label=key)
    plt.legend()
plt.show()

left = []
right = []

temp = np.array(motion.joint_angles_list[6]['leftHipPitch']) + \
        np.array(motion.joint_angles_list[6]['leftKneePitch']) + \
        np.array(motion.joint_angles_list[6]['leftAnklePitch'])

left=temp
temp = np.array(motion.joint_angles_list[6]['rightHipPitch']) + \
        np.array(motion.joint_angles_list[6]['rightKneePitch']) + \
        np.array(motion.joint_angles_list[6]['rightAnklePitch'])
right=temp

plt.plot(left, label='left foot')
plt.plot(right, label='right foot')
plt.legend()
plt.show()

print(max(motion.joint_angles_list[0]['leftHipPitch']))
print(min(motion.joint_angles_list[0]['rightHipPitch']))
print(max(motion.joint_angles_list[0]['rightHipPitch']))
print(min(motion.joint_angles_list[0]['leftHipPitch']))

print(motion.joint_angles_list[0]['leftHipPitch'][0])
print(motion.joint_angles_list[0]['leftKneePitch'][0])
print(motion.joint_angles_list[0]['leftAnklePitch'][0])
a = motion.joint_angles_list[0]['leftHipPitch'][0]+motion.joint_angles_list[0]['leftKneePitch'][0]+motion.joint_angles_list[0]['leftAnklePitch'][0]
print(a)
# while 1:
#     joint = motion.ref_motion()
#     print(joint)