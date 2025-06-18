import numpy as np
import os
import os.path as osp
import argparse
from pypcd4 import PointCloud
import time
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib
from mpl_toolkits.mplot3d.proj3d import proj_transform

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to the sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from calibration.extrinsic_calib import T_lidar_ti, T_box_lidar, \
                                         opti2sensor_gt, create_transformation_matrix, \
                                         src2dst_transformation

class PCLVisualizer():
    def __init__(self, n_sets=1, calib=False):
        self.paused = [False]
        self.picked_points = {"a": {}, "b": {}}
        self.calib = calib

        # Init plot entity
        self.fig = plt.figure(figsize=(12,10))
        self.fig.tight_layout()

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=10, azim=-165)
        self.ax.set_xlabel('X-metre')
        self.ax.set_ylabel('Y-metre')
        self.ax.set_zlabel('Z-metre')
        self.ax.set_xlim(0, 7.0)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-2.5, 2.5)

        # Initial empty plots
        self.scat = []
        for i in range(n_sets):
            self.scat.append(self.ax.scatter([], [], [], c='blue', label=f'Set {chr(i+65)}'))

    def load_pcl(self, path, type):
        cloud = None
        if (type == "pcd"):
            pcd = PointCloud.from_path(path)
            cloud = pcd.numpy(("x", "y", "z"))
            cloud = cloud[cloud[:, 0] <= 5.0]
            cloud = cloud[cloud[:, 0] >= 0.0]
            cloud = cloud[cloud[:, 1] <= 5.0]
            cloud = cloud[cloud[:, 1] >= 0.0]
            cloud = cloud[cloud[:, 2] <= 0.0]
            cloud = cloud[cloud[:, 2] >= -1.5]
        else:
            pcd = np.loadtxt(path)
            cloud = pcd
            cloud[:, [0, 1]] = cloud[:, [1, 0]]

        return cloud

    def load_data(self, frame):
        idx, file_01, sen1_fmt, file_02, sen2_fmt = frame

        # Load sensor pcl 1
        cloud_01 = self.load_pcl(file_01, sen1_fmt)

        # Load sensor pcl 2
        cloud_02 = None
        if (file_02 != ""  and sen2_fmt != ""):
            cloud_02 = self.load_pcl(file_02, sen2_fmt)
            if (self.calib):
                cloud_02 = src2dst_transformation(np.linalg.inv(T_lidar_ti), cloud_02)

        return cloud_01, cloud_02

    def make_update(self):
        def update(frame):
            if self.paused[0]:
                return []
            
            st = time.time()
            points_a, points_b = self.load_data(frame)

            # Clear previous scatter plots and re-plot
            self.ax.collections.clear()
            self.ax.scatter(points_a[:,0], points_a[:,1], points_a[:,2], c='blue', label='Set A', s=1.0)
            if (points_b is not None):
                self.ax.scatter(points_b[:,0], points_b[:,1], points_b[:,2], c='red', label='Set B', s=1.0)
            self.ax.set_title(f"Frame {frame[0]}")
            self.ax.legend()
            plt.pause(0.001)
            return []
        return update

    def on_key(self, event):
        if event.key == " ":
            self.paused[0] = not self.paused[0]
        
            if (self.paused[0]):
                self.ani.pause()
            else:
                self.ani.resume()

    def on_click(self, event, points_a, points_b, ax):
        if (event.button != 2):
            return

        # Project 3D points to 2D
        if (points_b is not None):
            points = np.append(points_a, points_b, axis=0)
        else:
            points = points_a
        x_proj, y_proj, _ = proj_transform(points[:, 0], points[:, 1], points[:, 2], ax.get_proj())
        screen_coords = ax.transData.transform(np.vstack([x_proj, y_proj]).T)

        # Find closest point to click
        mouse_xy = np.array([event.x, event.y])
        dists = np.linalg.norm(screen_coords - mouse_xy, axis=1)
        idx = np.argmin(dists)
        selected_point = points[idx]
        set_id = "a" if idx < points_a.shape[0] else "b"

        if (idx in self.picked_points[set_id]):
            self.picked_points[set_id].pop(idx)
            ax.scatter(*selected_point, c='k', s=1.0)
        else:
            self.picked_points[set_id][idx] = selected_point
            ax.scatter(*selected_point, c='k', marker='x', s=50)

        print(selected_point)
    
    def animation(self, frame_list):
        self.ani = FuncAnimation(self.fig, self.make_update(), frames=frame_list, 
                                 interval=1, blit=False)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

