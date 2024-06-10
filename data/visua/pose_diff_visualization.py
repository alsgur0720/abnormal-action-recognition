import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import math

left_arm = [5, 6, 7, 21, 22]
right_arm = [9,10,11,23,24]
leg = [13, 14, 15, 17, 18, 19]
body = [1, 4, 8, 20] #[0,1,2,3,4,8,20]

input = np.load('/home/vimlab/workspace/source/PyTorch-Deep-SVDD_real/ae_results/skel2_pretrained_parameters2(full)/full_range/left_arm/decoder_synthetic_left_arm_input.npy')
output = np.load('/home/vimlab/workspace/source/PyTorch-Deep-SVDD_real/ae_results/skel2_pretrained_parameters2(full)/full_range/left_arm/decoder_synthetic_left_arm_output.npy')
input = input.reshape(-1, 25, 3)
output = output.reshape(-1, 25, 3)

difference_map = np.abs(output[0] - input[0])
print(difference_map.shape)
la_value, ra_value, leg_value, body_value = 0, 0, 0, 0

for i in left_arm:
    la_power = np.power(difference_map[i], 2)
    la_sum = np.sum(la_power)
    la_distance = math.sqrt(la_sum)
    la_value += la_distance

la_value /= 5

for i in right_arm:
    ra_power = np.power(difference_map[i], 2)
    ra_sum = np.sum(ra_power)
    ra_distance = math.sqrt(ra_sum)
    ra_value += ra_distance

ra_value /= 5

for i in leg:
    leg_power = np.power(difference_map[i], 2)
    leg_sum = np.sum(leg_power)
    leg_distance = math.sqrt(leg_sum)
    leg_value += leg_distance

leg_value /= 5

for i in body:
    body_power = np.power(difference_map[i], 2)
    body_sum = np.sum(body_power)
    body_distance = math.sqrt(body_sum)
    body_value += body_distance

body_value /= 5

body_parts = ['Left_arm', 'Right_arm', 'Legs', 'Body']
values = [la_value, ra_value, leg_value, body_value]



# (25, 3) difference_map 

fig = plt.figure(dpi=300)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow(difference_map, cmap='gray')
ax2.bar(body_parts, values)
#ax2.xticks(np.arange(4), body_parts)


plt.title('25 x 3 Matrix')
plt.show()

# (5, ) difference_map by body part

