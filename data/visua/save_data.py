import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from .visualizations import Ax3DPose
import sys
from einops import rearrange
import pickle
import torch

# Visualize 3D Skeleton classes
#############################################################
# one person



## pgd alpha 0.01 threshold 0.1

# my_data = np.load('C:/Users/dooly/workspace/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/xsub/results_ad/L_inf_pgd/alpha00100/iter_50_thres0.10.npy',allow_pickle=True)

## pgd alpha 0.01 threshold 0.1


## pgd alpha 0.3 threshold 0.3

# my_data = np.load('C:/Users/dooly/datasets/adver_bon_length-data/iter_100_thres0.30.npy',allow_pickle=True)

## pgd alpha 0.3 threshold 0.3



## pgd alpha 0.5 threshold 1.0

# my_data = np.load('C:/Users/dooly/datasets/adver_bon_length-data/iter_100_thres1.00.npy',allow_pickle=True)

## pgd alpha 0.5 threshold 1.0

## clean

my_data = np.load('C:/Users/dooly/workspace/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/st-gcn-processed-data/data/NTU-RGB-D/xview/val_data.npy',allow_pickle=True)

# np.save('../first_frame.npy', my_data_clean[33][:,:,:,:])
## clean

# test = np.load('./condition_frame_data/stand_sit_frame.npy',allow_pickle=True)

# print(my_data[0][:,0,:,0])

# print('-------------------------------')

# print(test[0][:,0,:,0])
# exit()

with open('C:/Users/dooly/workspace/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/st-gcn-processed-data/data/NTU-RGB-D/xview/val_label.pkl', 'rb') as f:
  sample_name, label = pickle.load(f)


#32, 63, 71

# for i in range(101, 200):
  # print(label[i], "{}".format(i))

# print(label[71])


# exit()


# my_data = np.load('D:/minuk_folder/skeleton/NTU_RGB_D/Skeleton/numpy_skeleton/S001C001P001R001A058.skeleton.npy',allow_pickle=True).item()
# my_data = np.load('./S001C001P001R001A043.skeleton.npy',allow_pickle=True)


# data0 = my_data['skel_body1']



np.save('./first_frame2.npy',my_data[151,:,:,:,:])

exit()
N, C, T, V, M = my_data.shape


data1 = my_data[:,:,:,:,:]
data_motion = data1[:,:,1:,:,:] - data1[:,:,:-1,:,:]

data = np.zeros((N, C, T, V, M))

for i in range(0, N):
    all_zeros = np.all(data1[i,:,:,:,1] == 0)
    if all_zeros == True:
        if my_data[i][1][0][16][0] - my_data[i][1][0][17][0] < 0.1:
          data[i,:,0,:,0] = my_data[115,:,0,:,0]
        else:
          data[i,:,0,:,0] = my_data[33,:,0,:,0]
          
    else:
      data[i,:,0,:,:] = my_data[i,:,0,:,:]
      
        # if my_data[i][1][0][16][0] - my_data[i][1][0][17][0] < 0.1:
        #   data[i,:,0,:,0] = my_data[115,:,0,:,0]
        #   data[i,:,0,:,1] = my_data[115,:,0,:,0]
        # else:
        #   data[i,:,0,:,0] = my_data[33,:,0,:,0]
        #   data[i,:,0,:,1] = my_data[33,:,0,:,0]
        
data[:,:,1:,:,:] = data_motion
for i in range(0, T-1):
    if (np.all(data_motion[:,:,i,:,:] == 0)) == True:
        data[:,:,i+1,:,:] = 0
    else:
        data[:,:,i+1,:,:] = data[:,:,i,:,:] + data[:,:,i+1,:,:]


np.save('./condition_frame_data/stand_sit_frame2.npy', data)


# data1 = rearrange(my_data[115][:,:,:,0], 'x y z -> y z x')
# data1 = rearrange(my_data[33][:,:,:,1], 'x y z -> y z x')





