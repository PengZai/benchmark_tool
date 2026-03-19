import pycolmap
import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
import argparse
import yaml
from benchmark.dataset_tools.datasets import CameraData

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


parser = argparse.ArgumentParser()
parser.add_argument("--configdir", default="../../../configs/dataset_tools/BotanicGarden.yaml", type=str, help="path to configure directory")
parser.add_argument("--colmap_sparse_model_path", default="/media/spiderman/zhipeng_8t1/datasets/colmap_datasets/BotanicGarden_1018_00/sparse/0", type=str, help="path to configure directory")
parser.add_argument("--output_dir", default="/media/spiderman/zhipeng_8t1/datasets/BotanicGarden/1018-00/1018_00_img10hz600p/colmap_sparse_points/right_rgb", type=str, help="path to configure directory")
parser.add_argument("--used_camera_idx", default=1, type=int, help="path to configure directory")
parser.add_argument("--isOutputUndistortedDepth", default=1, type=bool, help="path to configure directory")

args = parser.parse_args()


with open(args.configdir, "r", encoding="utf-8") as f:
    configs = yaml.safe_load(f)

camera_data = CameraData(camera_config = configs['cameras']['camera' + str(args.used_camera_idx)])

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'depth'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'depth_vis'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'pointcloud'), exist_ok=True)

# Load a reconstruction (sparse model folder with cameras.txt, images.txt, points3D.txt)

reconstruction = pycolmap.Reconstruction(args.colmap_sparse_model_path)

# reconstruction = pycolmap.Reconstruction("/media/spiderman/zhipeng_8t1/datasets/colmap_datasets/PolyTunnel_haygrove_Sep2025_easy_bag/sparse/1")

reconstruction_image_list = [reconstruction.images[i] for i in sorted(reconstruction.images)]

# output_dir = "/media/spiderman/zhipeng_8t1/datasets/PolyTunnel_haygrove_Sep2025/colmap_trajectory"

width, height = camera_data.config['resolution']
fx, fy, cx, cy = camera_data.K[0,0], camera_data.K[1,1], camera_data.K[0,2], camera_data.K[1,2]

for i, rec_image_item in enumerate(reconstruction_image_list):

    str_name = rec_image_item.name
    str_filename = str_name.split("/")[-1]
    str_timestamp = str_filename.split(".png")[0]
    timestamp = float(str_timestamp[:10]+"."+str_timestamp[10:])

    point_c_list = []
    point_color_list = []

    if rec_image_item.camera_id == camera_data.id+1:

        print(f"processing image:{str_name} with camera id :{rec_image_item.camera_id}")        

        image_path = os.path.join(camera_data.config['imagepath'], str_filename)
        img_bgr = cv2.imread(image_path)
        if args.isOutputUndistortedDepth == True and np.sum(camera_data.config['distortion_coeffs']) > 0:
            img_bgr = cv2.remap(img_bgr, camera_data.remap1, camera_data.remap2, cv2.INTER_LINEAR) # undistorted image
                    
        vis_img_bgr_drawed_colmap_sparse_points_with_valid_depth = img_bgr.copy()
        colmap_sparse_depth = np.zeros((height, width), dtype=np.float32)

        for points2D in rec_image_item.points2D:
            if points2D.point3D_id < 0 or points2D.point3D_id not in reconstruction.points3D:
                continue 
            # u, v = points2D.xy
            # u = round(u)
            # v = round(v)
            point3D = reconstruction.points3D[points2D.point3D_id]
            point3D_in_cam = rec_image_item.cam_from_world() * point3D.xyz
            x, y, z = point3D_in_cam


            u = round(x * fx / z + cx)
            v = round(y * fy / z + cy)

            if z <= 1e-3 or z > 1e3:
                continue

            if 0 <= u <= width-1 and 0<= v <= height - 1 and z > 0:

                point_c_list.append(point3D_in_cam)
                c = img_bgr[v,u].astype("f4")/255.
                color = np.array([c[2], c[1], c[0]], dtype='f4')
                point_color_list.append(color)
                if colmap_sparse_depth[v, u] == 0:
                    colmap_sparse_depth[v, u] = z
                    cv2.circle(vis_img_bgr_drawed_colmap_sparse_points_with_valid_depth, (u, v), 2, single_depth2color(z, 0.01, 50), -1)

                elif colmap_sparse_depth[v, u] > z:
                    colmap_sparse_depth[v, u] = z
                    cv2.circle(vis_img_bgr_drawed_colmap_sparse_points_with_valid_depth, (u, v), 2, single_depth2color(z, 0.01, 50), -1)


        cv2.imwrite(os.path.join(args.output_dir, 'depth_vis', str_timestamp+".png"), vis_img_bgr_drawed_colmap_sparse_points_with_valid_depth)
        cv2.imwrite(os.path.join(args.output_dir, 'depth', str_timestamp+".tiff"), colmap_sparse_depth)

        point_cs = np.vstack(point_c_list, dtype='f4')
        point_cs = np.ascontiguousarray(point_cs)
        point_colors = np.vstack(point_color_list, dtype='f4')
        point_colors = np.ascontiguousarray(point_colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cs)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        o3d.io.write_point_cloud(os.path.join(args.output_dir, 'pointcloud', str_timestamp+".pcd"), pcd, write_ascii=False)  # binary (smaller/faster)
