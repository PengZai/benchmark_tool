import pycolmap
import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d
import argparse
import yaml
from dense_slam_benchmark.dataset_tools.datasets import build_dataset


def load_reconstruction(model_path):
    if not os.path.isdir(model_path):
        raise ValueError(f"COLMAP sparse model path does not exist or is not a directory: {model_path}")

    has_bin = any(
        os.path.exists(os.path.join(model_path, name))
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    )
    has_txt = any(
        os.path.exists(os.path.join(model_path, name))
        for name in ("cameras.txt", "images.txt", "points3D.txt")
    )

    if not has_bin and not has_txt:
        raise ValueError(
            f"No COLMAP model files found in {model_path}. Expected cameras/images/points3D as .bin or .txt."
        )

    return pycolmap.Reconstruction(model_path)


def resolve_reconstruction_camera_id(reconstruction, camera_data):
    available_camera_ids = set(reconstruction.cameras.keys())
    if camera_data.id in available_camera_ids:
        return camera_data.id
    if camera_data.id + 1 in available_camera_ids:
        return camera_data.id + 1
    raise ValueError(
        f"Could not match config camera id {camera_data.id} to reconstruction camera IDs "
        f"{sorted(available_camera_ids)}"
    )


def filename_to_timestamp(stem):
    if stem.isdigit() and len(stem) > 10:
        return float(stem[:10] + "." + stem[10:])
    return stem


def has_nonzero_distortion(distortion_coeffs):
    distortion_coeffs = np.asarray(distortion_coeffs, dtype=np.float32).reshape(-1)
    return np.any(np.abs(distortion_coeffs) > 1e-12)


def load_image_like_benchmark(camera_data, image_filename, undistort_image):
    image_path = os.path.join(camera_data.config["imagepath"], image_filename)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    if undistort_image and has_nonzero_distortion(camera_data.config["distortion_coeffs"]):
        return cv2.remap(img_bgr, camera_data.remap1, camera_data.remap2, cv2.INTER_LINEAR)
    return img_bgr


def get_valid_crop_from_remap(remap1, remap2):
    valid = (remap1 >= 0.0) & (remap2 >= 0.0)
    ys, xs = np.where(valid)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def crop_and_resize_image(image, crop_box, output_size, interpolation):
    x0, y0, x1, y1 = crop_box
    cropped = image[y0:y1, x0:x1]
    if cropped.size == 0:
        raise ValueError(f"Invalid crop box {crop_box} for image with shape {image.shape}")
    out_w, out_h = output_size
    if cropped.shape[1] == out_w and cropped.shape[0] == out_h:
        return cropped
    return cv2.resize(cropped, (out_w, out_h), interpolation=interpolation)


def adjusted_intrinsics_for_crop_resize(K, crop_box, output_size):
    x0, y0, x1, y1 = crop_box
    crop_w = max(x1 - x0, 1)
    crop_h = max(y1 - y0, 1)
    out_w, out_h = output_size
    sx = float(out_w) / float(crop_w)
    sy = float(out_h) / float(crop_h)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array(
        [
            [fx * sx, 0.0, (cx - x0) * sx],
            [0.0, fy * sy, (cy - y0) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


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
parser.add_argument("--configdir", default="../../../configs/dataset_tools/ETH3d/multi_view_training/delivery_area.yaml", type=str, help="path to configure directory")
parser.add_argument("--colmap_sparse_model_path", default="/mnt/lboro_nas/personal/Zhipeng/eth3d/multi_view_training_rig/delivery_area/rig_calibration", type=str, help="path to COLMAP sparse model directory containing .bin or .txt model files")
parser.add_argument("--output_dir", default="/mnt/lboro_nas/personal/Zhipeng/eth3d/multi_view_training_rig/delivery_area/colmap_sparse_points", type=str, help="path to configure directory")
parser.add_argument("--used_camera_idxes", default=[0,1], nargs="+", type=int, help="camera entry indexes such as 0 1")
parser.add_argument("--isOutputUndistortedDepth", default=1, type=int, help="1 to output in the undistorted image frame, 0 to keep the original image frame")

def main():
    args = parser.parse_args()

    with open(args.configdir, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    dataset = build_dataset(configs, config_path=args.configdir)

    reconstruction = load_reconstruction(args.colmap_sparse_model_path)
    reconstruction_image_list = [reconstruction.images[i] for i in sorted(reconstruction.images)]
    requested_camera_idxes = list(dict.fromkeys(args.used_camera_idxes))
    loaded_camera_ids = [camera_data.id for camera_data in dataset.camera_data_lists]

    for used_camera_idx in requested_camera_idxes:
        camera_entry_key = 'camera' + str(used_camera_idx)
        requested_camera_id = configs['cameras'][camera_entry_key]['id']
        camera_data = next(
            (camera_data for camera_data in dataset.camera_data_lists if camera_data.id == requested_camera_id),
            None,
        )
        if camera_data is None:
            raise ValueError(
                f"Could not find loaded camera data for {camera_entry_key} with id {requested_camera_id}. "
                f"Loaded camera ids: {loaded_camera_ids}"
            )

        os.makedirs(os.path.join(args.output_dir, camera_data.config['name']), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, camera_data.config['name'], 'depth'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, camera_data.config['name'], 'depth_vis'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, camera_data.config['name'], 'pointcloud'), exist_ok=True)

        reconstruction_camera_id = resolve_reconstruction_camera_id(reconstruction, camera_data)
        use_undistorted_output = bool(args.isOutputUndistortedDepth)
        if use_undistorted_output:
            projection_K = camera_data.K
            width, height = camera_data.remap1.shape[1], camera_data.remap1.shape[0]
        else:
            fx, fy, cx, cy = camera_data.config["original_intrinsics"]
            projection_K = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            width, height = camera_data.config.get("original_resolution", camera_data.config["resolution"])
        fx, fy, cx, cy = projection_K[0,0], projection_K[1,1], projection_K[0,2], projection_K[1,2]

        for rec_image_item in reconstruction_image_list:

            str_name = rec_image_item.name
            str_filename = str_name.split("/")[-1]
            str_timestamp = os.path.splitext(str_filename)[0]
            point_c_list = []
            point_color_list = []

            if rec_image_item.camera_id != reconstruction_camera_id:
                continue

            print(f"processing image:{str_name} with camera id :{rec_image_item.camera_id}")

            img_bgr = load_image_like_benchmark(
                camera_data,
                str_filename,
                undistort_image=use_undistorted_output,
            )

            vis_img_bgr_drawed_colmap_sparse_points_with_valid_depth = img_bgr.copy()
            colmap_sparse_depth = np.zeros((height, width), dtype=np.float32)

            for points2D in rec_image_item.points2D:
                if points2D.point3D_id < 0 or points2D.point3D_id not in reconstruction.points3D:
                    continue
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

            cv2.imwrite(os.path.join(args.output_dir, camera_data.config['name'], 'depth_vis', str_timestamp+".png"), vis_img_bgr_drawed_colmap_sparse_points_with_valid_depth)
            cv2.imwrite(os.path.join(args.output_dir, camera_data.config['name'], 'depth', str_timestamp+".tiff"), colmap_sparse_depth)

            if len(point_c_list) == 0:
                continue

            point_cs = np.vstack(point_c_list, dtype='f4')
            point_cs = np.ascontiguousarray(point_cs)
            point_colors = np.vstack(point_color_list, dtype='f4')
            point_colors = np.ascontiguousarray(point_colors)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cs)
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
            o3d.io.write_point_cloud(os.path.join(args.output_dir, camera_data.config['name'], 'pointcloud', str_timestamp+".pcd"), pcd, write_ascii=False)


if __name__ == "__main__":
    main()
