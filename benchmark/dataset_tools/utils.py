from scipy.spatial.transform import Rotation as Rot
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from decimal import Decimal, getcontext

getcontext().prec = 30  # enough for seconds + 9 fractional digits


def single_depths2colors(depth, min_depth, max_depth):
    depth = np.asarray(depth, dtype=np.float32)

    # clip and normalize to [0, 255]
    depth_clipped = np.clip(depth, min_depth, max_depth)
    normalized = 255 * (depth_clipped - min_depth) / (max_depth - min_depth)
    normalized = (255 - normalized).astype(np.uint8)

    # matplotlib colormap expects values in [0, 1]
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    colors_rgb = (cmap(normalized / 255.0)[..., :3] * 255).astype(np.uint8)

    # RGB -> BGR
    colors_bgr = colors_rgb[..., ::-1]

    return colors_bgr

def single_depth2color(depth, min_depth, max_depth):

    depth_clipped = np.clip(depth, min_depth, max_depth)
    normalized = int(255 * (depth_clipped - min_depth) / (max_depth - min_depth))    
    normalized = np.uint8(255 - normalized)  # Flip so small depth is red

    # Apply colormap on a 1x1 image
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    color = np.array(cmap(normalized)[:3]) * 255
    color = color.astype(np.uint8)
    # Extract the color as (B, G, R)
    return (int(color[2]), int(color[1]), int(color[0]))


def depth2color(depth, min_value = 0.3, max_value = 255):

    valid_depth = depth > min_value
    inv_depth = np.zeros_like(depth)
    inv_depth[valid_depth] = 1.0 / (depth[valid_depth])  
    vis = cv2.normalize(inv_depth, None, min_value, max_value, cv2.NORM_MINMAX)
    vis = vis.astype(np.uint8).squeeze()

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    vis = (cmap(vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    return vis

def invert_transform(T):
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv

def isInImage(u, v, z, width, height):
    if z <= 0 or u < 0 or u > width - 1 or v < 0 or v > height - 1:
        return False
    else:
        return True

def pose_to_T(x, y, z, qx, qy, qz, qw):
    R = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = [x, y, z]
    return T

def T_to_pose(T: np.ndarray):
    """
    Convert a 4x4 homogeneous transform to translation + quaternion.

    Args:
        T: (4,4) array-like homogeneous transform.

    Returns:
        t: (3,) translation vector [x, y, z]
        q: (4,) quaternion [q_x, q_y, q_z, q_w]  (SciPy order: x,y,z,w)
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T must be shape (4,4), got {T.shape}")

    t = T[:3, 3].copy()
    q = Rot.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w
    return t, q

def getSynchronizedSensorIdx(ref_ts, sensor_names, ts_tol):

    # whether it can find out the synchronized signal or not, it will return the idx that is closest to the ref_ts in timestamp
    

    synchronized_idx = -1
    minimum_ts_diff = float("inf")
    for i, sensor_name in enumerate(sensor_names):
        sensor_ts = sensor_name.split(".")[0]
        sensor_ts = int(sensor_ts) / 1e9
        ts_diff = abs(ref_ts - sensor_ts)
        if ts_diff < minimum_ts_diff:
            minimum_ts_diff = ts_diff
            synchronized_idx = i
    
    if minimum_ts_diff <= ts_tol :
        return True, synchronized_idx
    else:
        return False, synchronized_idx

def getSensorIdxWithClosestTimeStamp(ref_ts, sensor_names):

    # whether it can find out the synchronized signal or not, it will return the idx that is closest to the ref_ts in timestamp
    

    idx_with_closest_timestamp = -1
    minimum_ts_diff = float("inf")
    for i, sensor_name in enumerate(sensor_names):
        sensor_ts = sensor_name.split(".")[0]
        sensor_ts = int(sensor_ts) / 1e9
        ts_diff = abs(ref_ts - sensor_ts)
        if ts_diff < minimum_ts_diff:
            minimum_ts_diff = ts_diff
            idx_with_closest_timestamp = i
    
    return idx_with_closest_timestamp


def getSynchronizedPoseIdx(ref_ts, samples, ts_tol):

    synchronized_idx = -1
    minimum_ts_diff = float("inf")
    for i, sample in enumerate(samples):
        sample_ts = sample['ts']
        ts_diff = abs(ref_ts - sample_ts)
        if ts_diff < minimum_ts_diff:
            minimum_ts_diff = ts_diff
            synchronized_idx = i
    
    if minimum_ts_diff <= ts_tol :
        return True, synchronized_idx
    else:
        return False, synchronized_idx


def timestamp_str_to_float(ts_str):

    return float(ts_str[:10]+"."+ts_str[10:])

def calculateUndistortedRemap(distortion_model, resolution, intrinsics, distortion):

    image_w, image_h = resolution
    fx, fy, cx, cy = intrinsics
    d1, d2, d3, d4 = distortion
    D = np.array([d1, d2, d3, d4 ], dtype=np.float32)  # (4,)
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]], dtype=np.float32
    )

    if distortion_model == 'radtan':

        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (image_w, image_h), alpha=0.0)
        K = newK
        remap1, remap2 = cv2.initUndistortRectifyMap(
            K, D, R=None, newCameraMatrix=newK, size=(image_w, image_h), m1type=cv2.CV_32FC1
        )

    elif distortion_model == 'equidistant':
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (image_w, image_h), np.eye(3), balance=0.0, new_size=(image_w, image_h), fov_scale=1.0
        )
        K = newK
        remap1, remap2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), newK, (image_w, image_h), m1type=cv2.CV_32FC1
            # or CV_32FC1 if you prefer float maps
        )

    return newK, remap1, remap2



def undistortedDepth2Pointcloud(depth_image, intrinsics):

    fx, fy, cx, cy = intrinsics

    pc_h_list = []
    for v in range(depth_image.shape[0]):
        for u in range(depth_image.shape[1]):
            z =  depth_image[v, u]
            if z <= 1e-3 or z > 1e3:
                # print(f"{v},{u} invalid z:{z}")
                continue                            
            x = (u - cx)*z/fx
            y = (v - cy)*z/fy
            pc_h_list.append(np.array([x,y,z,1], dtype='f4'))

    pc_hs = np.vstack(pc_h_list, dtype='f4')

    return pc_hs