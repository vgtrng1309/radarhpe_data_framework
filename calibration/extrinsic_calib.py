import numpy as np
from scipy.spatial.transform import Rotation

# Create homogenous transformation matrix based on R and t
def create_transformation_matrix(R, t):
    T_mat = np.eye(4, 4)
    R = np.asarray(R)
    t = np.asarray(t)

    R = Rotation.from_quat(R).as_matrix()
    T_mat[:3,:3] = R
    T_mat[:3,3] = t
    return T_mat

# Transform ground truth points from optitrack coordinate to sensor coordinate
# Require: gt in optitrack coord                - points
#          trasform matrix from box to sensor   - T_box_sensor
#          box pose in optitrack coordinate     - T_opti_box   
def opti2sensor_gt(T_opti_box, T_box_sensor, points):
    points = np.hstack((points, np.ones((points.shape[0], 1)))).T
    
    # (4,4) @ (4,4) @ (4, N) => (4, N)
    points_tf = np.linalg.inv(T_box_sensor) @ np.linalg.inv(T_opti_box) @ points
    
    # return (N, 3)
    return points_tf[:3, :].T

# Transform points in src coord to dst coord
def src2dst_transformation(T_dst_src, pnts_src):
    pnts_src = np.hstack((pnts_src, np.ones((pnts_src.shape[0], 1)))).T
    
    # (4,4) @ (4,4) @ (4, N) => (4, N)
    pnts_dst = (T_dst_src @ pnts_src)[:3, :].T
    return pnts_dst

left_cam_info = {
    "height": 576,
    "width": 1024,
    "distortion_model": "rational_polynomial",
    "d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "k": [756.9138793945312,               0.0, 534.088134765625,
                        0.0, 756.9138793945312, 288.1670227050781,
                        0.0,               0.0,               1.0],
    "r": [1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0],
    "p": [756.9138793945312,               0.0, 534.088134765625, 0.0,
                        0.0, 756.9138793945312, 288.1670227050781, 0.0, 
                        0.0,               0.0,               1.0, 0.0]
}

# T_box_bosch = create_transformation_matrix([0.509, -0.506, -0.487, -0.498], \
#                                             [0.063, -0.992, -0.096])

# T_box_ti = create_transformation_matrix([0.509, -0.506, -0.487, -0.498],
#                                         [0.117, -0.923, -0.074])

T_lidar_ti =  create_transformation_matrix([0.027754399557206537,0.008579097606943222,0.24272697517377928,0.9696595835201429],
                                            [-0.07235461471255444,0.0026614269071107977,-0.8108131133744987])

# T_lidar_ti =  create_transformation_matrix([0.00780112, 0.00873333, 0.21170263, 0.97726397],
#                                             [-0.12542222, 0.19707212, -0.78611505])

T_lidar_cam = create_transformation_matrix([0.970, 0.242, -0.002, 0.005],
                                           [0.153, -0.091, -0.683])

T_box_bosch = create_transformation_matrix([0.509, -0.506, -0.487, -0.498], \
                                            [-0.027, -1.122, -0.096])

# T_box_lidar = create_transformation_matrix([-0.603, 0.366, 0.363, 0.610], \
#                                            [0.140, -0.055, -0.059])
T_box_lidar = {
    # T_box_lidar
    "march": create_transformation_matrix([0.62997149, -0.35287469, -0.36045185, -0.59049966], \
                                            [-0.06407701, 0.00918868, -0.0389507]),
    # "march": create_transformation_matrix([0.64567582,-0.33829661,-0.36969521,-0.57618017], \
    #                                         [0.00831144,-0.23640119,-0.20225951]),

    # T_box_lidar
    "april": create_transformation_matrix([-0.5746335131808743,0.388144200705127,0.3908205828041115,0.6053095712484965], \
                                            [-0.004329832357710589,-0.24160038431311914,-0.004170585853073838]),
   
    # T_box_lidar
    "april2": create_transformation_matrix([0.62634815169296,-0.37301353185619474,-0.2819017352276216,-0.6237630235903988], \
                                            [-0.05238446432323307,-0.05216343588799699,-0.030366911712931266]),

    # T_box_lidar 
    "april3": create_transformation_matrix([0.6154763307073661,-0.3889630479235408,-0.3211511332696329,-0.6056059637162239], \
                                            [0.1631897159528054,-0.006109490130329132,-0.001793091790514012]),
    # Processing 
    "april4": create_transformation_matrix([0.6197387994831763,-0.3594576843408586,-0.3453146253520921,-0.6061945257930028], \
                                            [0.020993421422225544,-0.139498906016142,0.02655068280459716]),
}

#391
T_calib_ti = {
    "march": create_transformation_matrix([-0.0072173, -0.00550122, 0.00250797, 0.99995568],
                                        [0, 0, 0]),
    # "march": create_transformation_matrix([0.53115215, -0.49995926, -0.48340444, -0.48398169],
    #                                     [0.041, -1.042, -0.090]),
    "april": create_transformation_matrix([-0.02164471, 0.0315108, -0.00191481, 0.99926719],
                                        [0, 0, 0]),
    # "april": create_transformation_matrix([0.53115215, -0.49995926, -0.48340444, -0.48398169],
    #                                     [-0.391, -0.842, -0.090]),
    "april2": create_transformation_matrix([-5.28932919e-02, 8.68221921e-05, 5.54430677e-03, 9.98584775e-01],
                                        [0, 0, 0]),                                                                                
    # "april2": create_transformation_matrix([-0.50726049, 0.43941915, 0.52738946, 0.52101628],
    #                                     [0.041, -1.242, -0.090]),                                                                                

    "april3": create_transformation_matrix([-0.04494116, -0.00325664, -0.01580687, 0.99885926],
                                        [0, 0, 0]),
    # "april3": create_transformation_matrix([0.53115215, -0.49995926, -0.48340444, -0.48398169],
    #                                     [-0.391, -0.842, -0.090]),

    "april4": create_transformation_matrix([-0.01147786, 0.01509525, -0.01428812, 0.99971808],
                                        [0, 0, 0]),
}


T_box_lid = T_calib_ti["march"] @ np.linalg.inv(T_lidar_ti)
R_box_ti = Rotation.from_matrix(T_box_lid[:3,:3]).as_quat()
print(R_box_ti, T_box_lid[:3,3])

