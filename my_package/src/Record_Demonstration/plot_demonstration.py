#! /usr/bin/python3
import os, sys
from math import *
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from common_utils import Robot_KM
from pose_transform import quat2mat


class PlotData:
    def __init__(self, file_name='demo'):
        self.load(name=file_name)

        self.N = self.position.shape[0]   # no of sample points
        self.DOF = self.q.shape[1]

        # Modified-DH Parameters 
        alpha = np.array([0, -np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2])   
        a = np.array([0, 0, 0.409, 0, 0, 0])  # data from parameter data-sheet (in meters)
        d = np.array([0.1555, 0, 0, 0.367, 0, 0.127])  # data from parameter data-sheet (in meters)
        d_nn = np.array([[0.0], [0.0], [0.0]])  # TCP coord in end-effector frame
        DH_params="modified"

        self.KM = Robot_KM(self.DOF, alpha, a, d, d_nn, DH_params)
        self.store_FK_data()
        self.store_rotation_matrices()
        
        self.x_demo = self.position[:,0]  # in meter
        self.y_demo = self.position[:,1]  # in meter
        self.z_demo = self.position[:,2]  # in meter

        self.Vx_demo = self.velocity[:,0]  # in meter/s
        self.Vy_demo = self.velocity[:,1]  # in meter/s
        self.Vz_demo = self.velocity[:,2]  # in meter/s

        # Setup the scale factor for orientation vectors
        self.orientation_scale = 0.15

    def load(self, name='demo'):
        curr_dir=os.getcwd()
        data = np.load(curr_dir+ '/data/' + str(name) + '.npz')
        self.q = data['q']    # shape: (N, 6), in rad
        self.position = data['traj']   # shape: (N, 3)
        self.orientation = data['ori']   # shape: (N, 4)
        self.velocity = data['vel']   # shape: (N, 6)
        # self.gripper = data['grip']

    def store_FK_data(self):
        X_coord, Y_coord, Z_coord = [], [], []

        for i in range(self.N):
            X, Y, Z = self.KM.taskspace_coord(self.q[i,:])
            X_coord.append(X)
            Y_coord.append(Y)
            Z_coord.append(Z)

        self.X_coord = np.array(X_coord)
        self.Y_coord = np.array(Y_coord)
        self.Z_coord = np.array(Z_coord)

    def store_rotation_matrices(self):
        self.rotation_matrices = []
        for i in range(self.N):
            rot_mat = quat2mat(self.orientation[i])
            self.rotation_matrices.append(rot_mat)

    def setup_demo_plot(self):
        self.fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 2)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[0,0], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,1], projection='3d'))
        return self.create_animation(self.update_plot)
    
    def create_animation(self, update_func):
        # Adjust interval based on data size to maintain smooth animation
        if self.N > 400:  # Very high frequency data (e.g., 50Hz)
            frames = np.linspace(0, self.N-1, 250, dtype=int)    # use a subset of frames for smoother playback
        else:
            frames = self.N-1  # use all the frames

        anim = FuncAnimation(self.fig, update_func, frames=frames, interval=10, blit=False, repeat=False)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        return anim

    def update_plot(self, frame):
        # Robot visualization
        self._plot_1(self.axs[0], frame)
        self._plot_2(self.axs[1], frame)
        self._set_3d_plot_properties(self.axs[0], 9, -25)   # set graph properties 
        self._set_3d_plot_properties(self.axs[1], 9, -25)   # set graph properties 
        return self.axs
    
    def _plot_1(self, ax, k):
        ax.clear()
        ax.plot(self.x_demo[:k+1], self.y_demo[:k+1], self.z_demo[:k+1], linewidth=2.5, color='k')

        for j in range(self.DOF):
            ax.plot(self.X_coord[k,j:j+2], self.Y_coord[k,j:j+2], self.Z_coord[k,j:j+2], '-', linewidth=9-j)  # Links
            ax.plot(self.X_coord[k,j], self.Y_coord[k,j], self.Z_coord[k,j], 'ko', markersize=9-j)   # Joints

        # Also plot orientation vectors in this plot
        self._plot_orientation_vectors(ax, k)

    def _plot_orientation_vectors(self, ax, k):
        position = self.position[k]
        
        # Use precomputed rotation matrix for better performance
        rotation_matrix = self.rotation_matrices[k]
        
        # Scale the orientation vectors
        scale = self.orientation_scale
        
        # Create a single quiver plot for all axes to improve performance
        origins = np.tile(position, (3, 1))
        
        # Create directions for all three axes
        directions = np.zeros((3, 3))
        directions[0] = rotation_matrix[:, 0] * scale  # X axis
        directions[1] = rotation_matrix[:, 1] * scale  # Y axis
        directions[2] = rotation_matrix[:, 2] * scale  # Z axis
        
        # Colors for the three axes
        colors = ['r', 'g', 'b']
        
        # Plot all three axes
        for i in range(3):
            ax.quiver(origins[i, 0], origins[i, 1], origins[i, 2], directions[i, 0], directions[i, 1], directions[i, 2], color=colors[i], alpha=0.5, linewidth=2)
    
    def _plot_2(self, ax, k):
        ax.clear()
        
    def _set_3d_plot_properties(self, ax, elev, azim):
        # ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-0.1, 1.0)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

    def show(self):
        plt.show()

if __name__ == "__main__":
    plotter = PlotData(file_name='demo')
    anim = plotter.setup_demo_plot()
    plotter.show()
