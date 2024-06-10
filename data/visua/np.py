import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from .visualizations import Ax3DPose
import sys
from einops import rearrange
import pickle
import torch
import cv2
from tslearn.clustering import TimeSeriesKMeans


# Visualize 3D Skeleton classes
#############################################################
# one person

class Ax3DPose_one(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.
    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation ---bones of skeleton
    self.I = np.arange(25)
    self.J = np.array([2,1,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,8,8,12,12])-1
    # Head_neck(0) / Left_arm(1) / right_arm(2) / Left_leg(3) / Right_leg(4) / body(5) indicator(POV)
    self.LR  = np.array([5,5,0,0,5,1,1,1,5,2,2,2,3,3,3,3,4,4,4,4,5,1,1,2,2])
    self.ax = ax

    # skeleton vals(joints, xyz)
    vals = np.zeros((25, 3))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=0.5, c=lcolor, marker="o", markersize=1))

    # self.ax.set_xlabel("x", labelpad=5)
    # self.ax.set_ylabel("y", labelpad=5)
    # self.ax.set_zlabel("z", labelpad=5)
  
  # refresh frame by frame of the pose
  def update(self, channels, HNcolor="#e74c3c", LAcolor="#e67e22", RAcolor="#2ecc71", LLcolor="#3498db", RLcolor="#9b59b6", Bcolor="#f1c40f"):
    """
    Update the plotted 3d pose.
    Args
      channels: 23*3=69-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == 75, "channels should have 69 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (25, -1) )
    #print('channels: ',channels)
    print(vals.shape)
    for i in np.arange( len(self.I) ):
        x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
        y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
        z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )

        # if x[0] != 0 and x[1] != 0:
        #     self.plots[i][0].set_xdata(x)
        # if x[0] != 0 and x[1] != 0:
        #     self.plots[i][0].set_ydata(y)
        # if x[0] != 0 and x[1] != 0:
        #     self.plots[i][0].set_3d_properties(z)

        self.plots[i][0].set_xdata(x)
        self.plots[i][0].set_ydata(y)
        self.plots[i][0].set_3d_properties(z)

        if self.LR[i]==0:
            self.plots[i][0].set_color(HNcolor)
        elif self.LR[i]==1:
            self.plots[i][0].set_color(LAcolor)
        elif self.LR[i]==2:
            self.plots[i][0].set_color(RAcolor)
        elif self.LR[i]==3:
            self.plots[i][0].set_color(LLcolor)
        elif self.LR[i]==4:
            self.plots[i][0].set_color(RLcolor)
        else:
            self.plots[i][0].set_color(Bcolor)

    r = 1
    #self.ax.scatter(xs,ys,zs, marker='o', s=1)
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('auto')

#############################################################
# two person

class Ax3DPose_two(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.
    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation ---bones of skeleton
    self.I = np.arange(25)
    self.J = np.array([2,1,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,8,8,12,12])-1
    # Head_neck(0) / Left_arm(1) / right_arm(2) / Left_leg(3) / Right_leg(4) / body(5) indicator(POV)
    self.LR  = np.array([5,5,0,0,5,1,1,1,5,2,2,2,3,3,3,3,4,4,4,4,5,1,1,2,2])
    self.ax = ax

    # skeleton vals(joints, x1y1z1x2y2z2)
    vals = np.zeros((25, 6))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
        x1 = np.array( [vals[self.I[i], 0], vals[self.J[i], 0], vals[self.I[i], 3], vals[self.J[i], 3]])
        y1 = np.array( [vals[self.I[i], 1], vals[self.J[i], 1], vals[self.I[i], 4], vals[self.J[i], 4]])
        z1 = np.array( [vals[self.I[i], 2], vals[self.J[i], 2], vals[self.I[i], 5], vals[self.J[i], 5]])
        self.plots.append(self.ax.plot(x1, y1, z1, lw=0.5, c=lcolor, marker="o", markersize=1))

    self.ax.set_xlabel("x", labelpad=5)
    self.ax.set_ylabel("y", labelpad=5)
    self.ax.set_zlabel("z", labelpad=5)
  
  # refresh frame by frame of the pose
  def update(self, channel, HNcolor="#e74c3c", LAcolor="#e67e22", RAcolor="#2ecc71", LLcolor="#3498db", RLcolor="#9b59b6", Bcolor="#f1c40f"):
    """
    Update the plotted 3d pose.
    Args
      channels: 23*3=69-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channel.size == 150, "channels should have 69 entries, it has %d instead" % channel.size
    vals = np.reshape( channel, (25, -1) )
    print(vals.shape)

    for i in np.arange( len(self.I) ):
        x1 = np.array( [vals[self.I[i], 0], vals[self.J[i], 0], vals[self.I[i], 3], vals[self.J[i], 3]])
        y1 = np.array( [vals[self.I[i], 1], vals[self.J[i], 1], vals[self.I[i], 4], vals[self.J[i], 4]])
        z1 = np.array( [vals[self.I[i], 2], vals[self.J[i], 2], vals[self.I[i], 5], vals[self.J[i], 5]])


        # if x1[0] != 0 and x1[1] != 0:
        #     self.plots[i][0].set_xdata(x1)
        # if x1[0] != 0 and x1[1] != 0:
        #     self.plots[i][0].set_ydata(y1)
        # if x1[0] != 0 and x1[1] != 0:
        #     self.plots[i][0].set_3d_properties(z1)


        self.plots[i][0].set_xdata(x1)

        self.plots[i][0].set_ydata(y1)

        self.plots[i][0].set_3d_properties(z1)

        if self.LR[i]==0:
            self.plots[i][0].set_color(HNcolor)
        elif self.LR[i]==1:
            self.plots[i][0].set_color(LAcolor)
        elif self.LR[i]==2:
            self.plots[i][0].set_color(RAcolor)
        elif self.LR[i]==3:
            self.plots[i][0].set_color(LLcolor)
        elif self.LR[i]==4:
            self.plots[i][0].set_color(RLcolor)
        else:
            self.plots[i][0].set_color(Bcolor)

    r = 1
    #self.ax.scatter(xs,ys,zs, marker='o', s=1)
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('equal')

#############################################################


# Making Occlusion
#############################################################

def erase_joint(skel_coord, joint_list):
    occluded_skel_coord =skel_coord
    for joint_num in joint_list:
        occluded_skel_coord[:, joint_num,:] = 0
    
    return occluded_skel_coord

def make_occlusion(skel_coords, occlusion_location='right_arm'):

    # POV
    right_arm = np.array([10, 11, 12 ,24, 25])-1 # 5
    left_arm = np.array([6, 7, 8, 22, 23])-1 # 5
    head_body = np.array([2, 5, 9, 21])-1 # 7 [1, 2, 3, 4, 5, 9, 21] 
    lower_body = np.array([14, 15, 16, 18, 19, 20])-1 #9 1번 joint 중복 [1, 13, 14, 15, 16, 17, 18, 19, 20]

    if occlusion_location == 'right_arm':
        occluded_skel_coords = erase_joint(skel_coords, right_arm)
    elif occlusion_location == 'left_arm':
        occluded_skel_coords = erase_joint(skel_coords, left_arm)
    elif occlusion_location == 'head_body':
        occluded_skel_coords = erase_joint(skel_coords, head_body)
    elif occlusion_location == 'lower_body':
        occluded_skel_coords = erase_joint(skel_coords, lower_body)
    
    return occluded_skel_coords

#############################################################


# Forward
#############################################################



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




with open('C:/Users/dooly/workspace/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/st-gcn-processed-data/data/NTU-RGB-D/xview/val_label.pkl', 'rb') as f:
  sample_name, label = pickle.load(f)


#32, 63, 71

# for i in range(101, 200):
  # print(label[i], "{}".format(i))

# print(label[71])


# exit()


input = np.load('./results_ae/input.npy',allow_pickle=True)
input = rearrange(input[1][:,:,:,0], 'x y z -> y z x')
output = np.load('./results_ae/output.npy',allow_pickle=True)
output = rearrange(output[1][:,:,:,0], 'x y z -> y z x')


fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

ob = Ax3DPose_one(ax)
# ob = Ax3DPose_two(ax)

def init():
    ax.view_init(-72, 90)


ani = FuncAnimation(fig, ob.update, init_func=init, frames=input, interval=1200)
ani_motion = FuncAnimation(fig, ob.update, init_func=init, frames=output, interval=1200)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# plt.show()


ani_motion.save('./results_ae/input_0.gif', fps=15)
ani.save('./results_ae/output_0.gif', fps=15)

print(input.shape)
print(output.shape)
exit()
# my_data = np.load('./S001C001P001R001A043.skeleton.npy',allow_pickle=True)


# data0 = my_data['skel_body1']

N, C, T, V, M = my_data.shape


np.set_printoptions(suppress=True)


data0 = rearrange(my_data[33][:,:,:,0], 'x y z -> y z x')
data1 = rearrange(my_data[101][:,:,:,0], 'x y z -> y z x')

# start_skel_local = np.stack((data0[0,:,0],data0[0,:,2]), axis=-1)

# start_skel_server = np.stack((data1[0,:,0],data1[0,:,2]), axis=-1)




# x_src, y_src, z_src = start_skel_local[0,:]
# x_dst, y_dst, z_dst = start_skel_server[0,:]

# src_center = np.array((0-x_src, 0-y_src, 0-z_src))
# dst_center = np.array((0-x_dst, 0-y_dst, 0-z_dst))
# src_points_centered = start_skel_local 
# dst_points_centered = start_skel_server

start_skel_local = data0[0,:,:]
start_skel_server = data1[0,:,:]


src_center = np.mean(start_skel_local, axis=0)
dst_center = np.mean(start_skel_server, axis=0)

src_points_centered = start_skel_local - src_center
dst_points_centered = start_skel_server - dst_center


U, S, Vt = np.linalg.svd(np.dot(dst_points_centered.T, src_points_centered))
R_matrix = np.dot(Vt.T, U.T)

if np.linalg.det(R_matrix) < 0:
    Vt[-1,:] *= -1  # 마지막 행 부호 변경
    R_matrix = np.dot(Vt.T, U.T)  # 재계산
    
translation = dst_center - np.dot(R_matrix, src_center)

transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = R_matrix
transformation_matrix[:3, 3] = translation


src_points_homogeneous = np.hstack((start_skel_local, np.ones((start_skel_local.shape[0], 1))))

transformed_src_points = transformation_matrix @ src_points_homogeneous.T

transformed_src_points = transformed_src_points[:3].T


# data0[0,:,0] = transformed_src_points[:,0]
# data0[0,:,2] = transformed_src_points[:,1]

data0[0,:,:] = transformed_src_points
# data0 = rearrange(my_data[34][:,:,:,0], 'x y z -> y z x')
# data1 = rearrange(my_data[33][:,:,:,1], 'x y z -> y z x')


data_motion = data1[1:,:,:] - data1[:-1,:,:]

data = np.zeros((300, 25, 3))
data[0,:,:] = data0[0,:,:]
data[1:, :, :] = data_motion

for i in range(0, 299):
  if (np.all(data_motion[i,:,:] == 0)) == True:
    data[i,:,:] = 0
  else:
    data[i+1,:,:] = data[i,:,:] + data[i+1,:,:]



# print(data0.shape)
# exit()

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

ob = Ax3DPose_one(ax)
# ob = Ax3DPose_two(ax)

def init():
    ax.view_init(-72, 90)


ani = FuncAnimation(fig, ob.update, init_func=init, frames=data1, interval=1200)
ani_motion = FuncAnimation(fig, ob.update, init_func=init, frames=data, interval=1200)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# plt.show()


ani_motion.save('./experiment/101_test_xyz.gif', fps=15)
ani.save('./experiment/101_original_xyz.gif', fps=15)


# ani.save('./clean_4.gif', fps=15)
#ani.save('./my_3d_visualization/occluded/hb_jump_up.gif', fps=15)
#ani.save('./my_3d_visualization/occluded/ra_hand_waving.gif', fps=15)
#ani.save('/home/vimlab/ae/real_occluded/full_range_reduce1/legs/decoder_synthetic_legs_input.mp4', fps=1)
#ani.save('./missing_file.gif', fps=15)

#############################################################