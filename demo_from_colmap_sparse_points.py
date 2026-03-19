import pycolmap
import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

import torch
from prior_depth_anything import PriorDepthAnything



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

colmap_sparse_model_path2 =  "/media/spiderman/zhipeng_8t1/datasets/colmap_datasets/BotanicGarden_1018_00"
colmap_sparse_model_path = "/media/spiderman/zhipeng_8t1/datasets/colmap_datasets/BotanicGarden_1018_00/sparse/0"
image_dir = "/media/spiderman/zhipeng_8t1/datasets/colmap_datasets/BotanicGarden_1018_00/images/rig1/left_rgb"
output_dir = "/media/spiderman/zhipeng_8t1/datasets/BotanicGarden/1018-00/1018_00_img10hz600p/Prior-Depth-Anything/left_rgb"
height = 600
width = 960
selected_camera_id = 1
isVis = True

mvs = pycolmap.MVSModel()
mvs.read_from_colmap(
    colmap_sparse_model_path2,
    sparse_path="sparse/0",
    images_path="images",
)


shared = mvs.compute_shared_points()


os.makedirs(os.path.join(output_dir, "corse_refine_depth"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "vis_corse_refine_depth"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "refine_depth"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "vis_refine_depth"), exist_ok=True)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# priorda = PriorDepthAnything(device=device, coarse_only=True)
priorda = PriorDepthAnything(device=device)
priorda_coarse_only = PriorDepthAnything(device=device, coarse_only=True)


# Load a reconstruction (sparse model folder with cameras.txt, images.txt, points3D.txt)
reconstruction = pycolmap.Reconstruction(colmap_sparse_model_path)

# reconstruction = pycolmap.Reconstruction("/media/spiderman/zhipeng_8t1/datasets/colmap_datasets/PolyTunnel_haygrove_Sep2025_easy_bag/sparse/1")

reconstruction_image_list = [reconstruction.images[i] for i in sorted(reconstruction.images)]

# output_dir = "/media/spiderman/zhipeng_8t1/datasets/PolyTunnel_haygrove_Sep2025/colmap_trajectory"



for i, rec_image_item in enumerate(reconstruction_image_list):

    str_name = rec_image_item.name
    str_filename = str_name.split("/")[-1]
    str_timestamp = str_filename.split(".png")[0]
    timestamp = float(str_timestamp[:10]+"."+str_timestamp[10:])

    print(f"processing image:{str_name} with camera id :{rec_image_item.camera_id}")        

    if rec_image_item.camera_id == selected_camera_id:

        colmap_sparse_matrix = np.zeros((height, width), dtype=np.float32)

        for points2D in rec_image_item.points2D:
            if points2D.point3D_id < 0 or points2D.point3D_id not in reconstruction.points3D:
                continue 
            u, v = points2D.xy
            u = round(u)
            v = round(v)
            point3D = reconstruction.points3D[points2D.point3D_id]

            point3D_in_cam = rec_image_item.cam_from_world() * point3D.xyz
            x, y, z = point3D_in_cam
            if 0 <= u <= width-1 and 0<= v <= height - 1 and z > 0:
                colmap_sparse_matrix[v, u] = point3D_in_cam[2]


 
        image_path = os.path.join(image_dir, str_filename)

        corse_refine_depth = priorda_coarse_only.infer_one_sample(image=image_path, prior=colmap_sparse_matrix, visualize=False)
        corse_refine_depth = corse_refine_depth.detach().cpu().numpy() 
        refine_depth = priorda.infer_one_sample(image=image_path, prior=colmap_sparse_matrix, visualize=False)
        refine_depth = refine_depth.detach().cpu().numpy() 


        vis_corse_refine_depth = depth2color(corse_refine_depth)
        vis_refine_depth = depth2color(refine_depth)

        cv2.imwrite(os.path.join(output_dir, "corse_refine_depth", str_timestamp+".tiff"), corse_refine_depth)
        cv2.imwrite(os.path.join(output_dir, "vis_corse_refine_depth", str_timestamp+".png"), vis_corse_refine_depth)
        cv2.imwrite(os.path.join(output_dir, "refine_depth",  str_timestamp+".tiff"), refine_depth)
        cv2.imwrite(os.path.join(output_dir, "vis_refine_depth",  str_timestamp+".png"), vis_refine_depth)






