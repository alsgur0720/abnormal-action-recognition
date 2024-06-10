import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

# Visualize 3D Skeleton classes
#############################################################
# one person

class Ax3DPose_one(object):
  def __init__(self, ax1, ax2, lcolor="#3498db", rcolor="#e74c3c"):
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
    self.ax1 = ax1
    self.ax2 = ax2

    # skeleton vals(joints, xyz)
    vals1 = np.zeros((25, 3))
    vals2 = np.zeros((25, 3))

    # Make connection matrix
    self.plots1 = []
    self.plots2 = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals1[self.I[i], 0], vals1[self.J[i], 0]] )
      y = np.array( [vals1[self.I[i], 1], vals1[self.J[i], 1]] )
      z = np.array( [vals1[self.I[i], 2], vals1[self.J[i], 2]] )
      self.plots1.append(self.ax1.plot(x, y, z, lw=0.5, c=lcolor, marker="o", markersize=1))
    
    for i in np.arange( len(self.I) ):
      x = np.array( [vals2[self.I[i], 0], vals2[self.J[i], 0]] )
      y = np.array( [vals2[self.I[i], 1], vals2[self.J[i], 1]] )
      z = np.array( [vals2[self.I[i], 2], vals2[self.J[i], 2]] )
      self.plots2.append(self.ax2.plot(x, y, z, lw=0.5, c=lcolor, marker="o", markersize=1))

    # self.ax1.set_xlabel("x", labelpad=1)
    # self.ax1.set_ylabel("y", labelpad=1)
    # self.ax1.set_zlabel("z", labelpad=1)
    # self.ax2.set_xlabel("x", labelpad=1)
    # self.ax2.set_ylabel("y", labelpad=1)
    # self.ax2.set_zlabel("z", labelpad=1)
    
    self.ax1.set_title("AE_input\n")
    self.ax2.set_title("AE_output\n")

    self.ax1.set_axis_off()
    self.ax2.set_axis_off()
  
  # refresh frame by frame of the pose
  def update(self, channels1, HNcolor="#e74c3c", LAcolor="#e67e22", RAcolor="#2ecc71", LLcolor="#3498db", RLcolor="#9b59b6", Bcolor="#f1c40f"):
    """
    Update the plotted 3d pose.
    Args
      channels: 23*3=69-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """
    # 75씩 분할하기
    #print(channels1.shape)
    assert channels1.size == 150, "channels should have 69 entries, it has %d instead" % channels1.size
    channels1 = channels1.reshape(2, 75)
    vals1 = np.reshape( channels1[0], (25, -1) )
    vals2 = np.reshape( channels1[1], (25, -1) )

    for i in np.arange( len(self.I) ):
        x = np.array( [vals1[self.I[i], 0], vals1[self.J[i], 0]] )
        y = np.array( [vals1[self.I[i], 1], vals1[self.J[i], 1]] )
        z = np.array( [vals1[self.I[i], 2], vals1[self.J[i], 2]] )

        # if x[0] != 0 and x[1] != 0:
        #     self.plots[i][0].set_xdata(x)
        # if x[0] != 0 and x[1] != 0:
        #     self.plots[i][0].set_ydata(y)
        # if x[0] != 0 and x[1] != 0:
        #     self.plots[i][0].set_3d_properties(z)

        self.plots1[i][0].set_xdata(x)
        self.plots1[i][0].set_ydata(y)
        self.plots1[i][0].set_3d_properties(z)

        if self.LR[i]==0:
            self.plots1[i][0].set_color(HNcolor)
        elif self.LR[i]==1:
            self.plots1[i][0].set_color(LAcolor)
        elif self.LR[i]==2:
            self.plots1[i][0].set_color(RAcolor)
        elif self.LR[i]==3:
            self.plots1[i][0].set_color(LLcolor)
        elif self.LR[i]==4:
            self.plots1[i][0].set_color(RLcolor)
        else:
            self.plots1[i][0].set_color(Bcolor)

    for i in np.arange( len(self.I) ):
        x = np.array( [vals2[self.I[i], 0], vals2[self.J[i], 0]] )
        y = np.array( [vals2[self.I[i], 1], vals2[self.J[i], 1]] )
        z = np.array( [vals2[self.I[i], 2], vals2[self.J[i], 2]] )

        # if x[0] != 0 and x[1] != 0:
        #     self.plots[i][0].set_xdata(x)
        # if x[0] != 0 and x[1] != 0:
        #     self.plots[i][0].set_ydata(y)
        # if x[0] != 0 and x[1] != 0:
        #     self.plots[i][0].set_3d_properties(z)

        self.plots2[i][0].set_xdata(x)
        self.plots2[i][0].set_ydata(y)
        self.plots2[i][0].set_3d_properties(z)

        if self.LR[i]==0:
            self.plots2[i][0].set_color(HNcolor)
        elif self.LR[i]==1:
            self.plots2[i][0].set_color(LAcolor)
        elif self.LR[i]==2:
            self.plots2[i][0].set_color(RAcolor)
        elif self.LR[i]==3:
            self.plots2[i][0].set_color(LLcolor)
        elif self.LR[i]==4:
            self.plots2[i][0].set_color(RLcolor)
        else:
            self.plots2[i][0].set_color(Bcolor)

    r = 1

    xroot, yroot, zroot = vals1[0,0], vals1[0,1], vals1[0,2]
    self.ax1.set_xlim3d([-r+xroot, r+xroot])
    self.ax1.set_zlim3d([-r+zroot, r+zroot])
    self.ax1.set_ylim3d([-r+yroot, r+yroot])
    
    self.ax1.tick_params(axis='x', labelsize=3)
    self.ax1.tick_params(axis='y', labelsize=3)
    self.ax1.tick_params(axis='z', labelsize=3)

    self.ax1.set_aspect('equal')

    xroot, yroot, zroot = vals2[0,0], vals2[0,1], vals2[0,2]
    self.ax2.set_xlim3d([-r+xroot, r+xroot])
    self.ax2.set_zlim3d([-r+zroot, r+zroot])
    self.ax2.set_ylim3d([-r+yroot, r+yroot])

    self.ax2.tick_params(axis='x', labelsize=3)
    self.ax2.tick_params(axis='y', labelsize=3)
    self.ax2.tick_params(axis='z', labelsize=3)

    self.ax1.set_aspect('equal')


my_data1 = np.load('./S018C001P041R002A108.skeleton.npy',allow_pickle=True).item()
my_data2 = np.load('./S018C001P041R002A108.skeleton.npy',allow_pickle=True).item()


my_data1 = my_data1['skel_body0'].reshape(-1,1,75)
my_data2 = my_data2['skel_body0'].reshape(-1,1,75)
my_data = np.concatenate([my_data1, my_data2], axis=2)

fig = plt.figure(dpi=300)
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ob = Ax3DPose_one(ax1, ax2)

def init():
    ax1.view_init(-72, 90)
    ax2.view_init(-72, 90)


ani = FuncAnimation(fig, ob.update, init_func=init, frames=my_data[:100], interval=1200)
plt.show()
#ani.save('/home/vimlab/ae/no_axis/decoder_synthetic_right_leg.mp4', fps=1)