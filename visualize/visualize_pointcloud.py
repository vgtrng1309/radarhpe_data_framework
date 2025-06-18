import numpy as np
import os
import os.path as osp
import argparse
import time
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import sys
import os
from pcl_visualizer import PCLVisualizer

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to the sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from calibration.extrinsic_calib import T_lidar_ti, T_box_lidar, \
                                         opti2sensor_gt, create_transformation_matrix, \
                                         src2dst_transformation

def main():
    parser = argparse.ArgumentParser(
        prog="visualize_pointcloud.py",
        description="For showing pointcloud with Matplotlib"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        default="",
        help="Path to sequence root directory"
    )

    parser.add_argument(
        "--dir_01",
        type=str,
        default="radarpcl",
        help="Path suffix to first pcl dir from root dir",
    )

    parser.add_argument(
        "--dir_02",
        type=str,
        default="",
        help="Path suffix to second pcl dir from root dir. (Optional)",
    )

    parser.add_argument(
        "--calib",
        action="store_true",
        help="Perform extrinsic calibration for data before showing"
    )

    parser.add_argument(
        "-s", "--start_frame",
        type=int,
        default=0,
        help="Starting frame"
    )

    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="rt",
        help="Mode: rt-realtime, fbf-frame by frame, s-single frame at start frame"
    )

    args = parser.parse_args()

    # Dir 1 pcl format
    sen1_fmt = "pcd" if os.listdir(osp.join(args.dir,args.dir_01))[0][-3:] == "pcd" else "xyzdi"

    # Get mapping data if visualize two pcl sets
    ts_map = None
    n_sets = 1
    if (args.dir_02 != ""):
        # Dir 2 pcl format
        sen2_fmt = "pcd" if os.listdir(osp.join(args.dir,args.dir_02))[0][-3:] == "pcd" else "xyzdi"
        
        # Mapping file
        sensor_01 = args.dir_01[args.dir_01.rfind("/")+1:]
        sensor_02 = args.dir_02[args.dir_02.rfind("/")+1:]
        
        sen2sen_map = "calib/" + sensor_01 + "2" + sensor_02 + "_mapping.txt"
        if (osp.exists(osp.join(args.dir, sen2sen_map))):
            sen2sen = np.loadtxt(osp.join(args.dir, sen2sen_map), dtype=str)
        else:
            sen2sen_map = "calib/" + sensor_02 + "2" + sensor_01 + "_mapping.txt"
            if (osp.exists(osp.join(args.dir, sen2sen_map))):
                sen2sen = np.loadtxt(osp.join(args.dir, sen2sen_map), dtype=str)
                sen2sen[:,[0,1]] = sen2sen[:,[1,0]]
            else:
                print("Cannot find mapping file for two pcl sets. Exit.")
                exit(0)
        ts_map = sen2sen
        n_sets = 2
    
    viz = PCLVisualizer(n_sets, args.calib)
   
    # forming frame list
    frame_list = []
    if (ts_map is not None):
        for i, src2dst in enumerate(ts_map[args.start_frame:]):
            if (src2dst[0] == "nan" or src2dst[1] == "nan"):
                continue

            file_01 = src2dst[0] + "." + sen1_fmt
            file_01 = os.path.join(args.dir, args.dir_01, file_01)
            file_02 = src2dst[1] + "." + sen2_fmt
            file_02 = os.path.join(args.dir, args.dir_02, file_02)

            frame_list.append([i, file_01, sen1_fmt, file_02, sen2_fmt])
    else:
        for i, file in enumerate(sorted(os.listdir(osp.join(args.dir, args.dir_01)))):
            frame_list.append([i, osp.join(args.dir, args.dir_01, file), sen1_fmt, "", ""])

    # Start the animation
    if (args.mode == "s"):
        points_a, points_b = viz.load_data(frame_list[args.start_frame])

        # Clear previous scatter plots and re-plot
        viz.ax.set_title(f"Frame {frame_list[args.start_frame][0]}")
        viz.ax.legend()
        viz.ax.scatter(points_a[:,0], points_a[:,1], points_a[:,2], 
                     c='blue', label='Set A', s=1.0, picker=True)
        if (points_b is not None):
            viz.ax.scatter(points_b[:,0], points_b[:,1], points_b[:,2], 
                         c='red', label='Set B', s=1.0, picker=True)
            viz.fig.canvas.mpl_connect('button_press_event', 
                lambda event: viz.on_click(event, points_a[:,:3], points_b[:,:3], viz.ax))
        else:
            viz.fig.canvas.mpl_connect('button_press_event', 
                lambda event: viz.on_click(event, points_a[:,:3], None, viz.ax))
    else: 
        viz.animation(frame_list)

    plt.show()
    for pa in list(viz.picked_points["a"].values()): 
        print(f"[({pa[0]:0.2f}, {pa[1]:0.2f}, {pa[2]:0.2f}), ", end="")
    for pb in list(viz.picked_points["b"].values()):
        print(f"({pb[0]:0.2f}, {pb[1]:0.2f}, {pb[2]:0.2f})],")

if __name__ == "__main__":
    main()
