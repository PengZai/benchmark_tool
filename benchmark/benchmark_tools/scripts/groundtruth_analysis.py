
import json
from omegaconf import DictConfig, OmegaConf
import hydra
import os
from benchmark.utils import updateConfig
from benchmark.dataloader import CameraDataset
import open3d as o3d
import numpy as np
import cv2
from benchmark import metrics
from collections import defaultdict
from benchmark.postprocessing import make_pts3d
from benchmark.dataset_tools.utils import single_depths2colors



def groundtruth_analysis(config):
    with open(os.path.join(config['root_data_dir'], config['scene_name'], "scene.json"), "r", encoding="utf-8") as f:
        camera_configs = json.load(f)
    
    
    config = updateConfig(config, camera_configs)

    all_result = {
        'overall':defaultdict(list),
        'result_per_view':[],
    }

    camera_dataset_list = []
    for camera_config_name, camera_config in config.cameras.items():
        camera_dataset = CameraDataset(camera_config)
        camera_dataset_list.append(camera_dataset)

        K_matrix = np.array(
            [[camera_dataset.config['undistorted_intrinsics'][0],0,camera_dataset.config['undistorted_intrinsics'][2]],
             [0,camera_dataset.config['undistorted_intrinsics'][1],camera_dataset.config['undistorted_intrinsics'][3]],
             [0,0,1]])
        
        for i in range(len(camera_dataset.samples)):

            undistorted_image_path = os.path.join(camera_config['datapath']['undistorted_images'], camera_dataset.undistorted_image_names[i])
            GT_depth_path = os.path.join(camera_config['datapath']['GT_depth'], camera_dataset.GT_depth_names[i])
            GT_pointcloud_path = os.path.join(camera_config['datapath']['GT_pointcloud'], camera_dataset.GT_pointcloud_names[i])
            input_depth_path = os.path.join(camera_config['datapath']['input_depth'], camera_dataset.input_depth_names[i])
            input_pointcloud_path = os.path.join(camera_config['datapath']['input_pointcloud'], camera_dataset.input_pointcloud_names[i])


            input_depth = cv2.imread(input_depth_path, cv2.IMREAD_UNCHANGED)
            # input_depth = cv2.resize(input_depth, (512, 336), interpolation=cv2.INTER_NEAREST)
            input_depth_mask = (input_depth > 0) & ( input_depth <= 10)
            # input_depth_mask = input_depth > 0
            # input_depth_mask = cv2.resize(input_depth_mask.astype(np.uint8), (512, 336), interpolation=cv2.INTER_NEAREST).astype(bool)

            input_depth = np.expand_dims(input_depth, axis=-1)
            reproject_input_depth_pointcloud = make_pts3d(input_depth, K_matrix, input_depth_mask)
            reproject_input_depth_pointcloud = reproject_input_depth_pointcloud[input_depth_mask]

            input_pointcloud = np.asarray(o3d.io.read_point_cloud(input_pointcloud_path).points, dtype="f4") 

            GT_depth = cv2.imread(GT_depth_path, cv2.IMREAD_UNCHANGED)
            # GT_depth = cv2.resize(GT_depth, (512, 336), interpolation=cv2.INTER_NEAREST)
            # GT_depth_mask = GT_depth > 0
            GT_depth_mask = (GT_depth > 0) & (GT_depth <= 10)

            # GT_depth_mask = cv2.resize(GT_depth_mask.astype(np.uint8), (512, 336), interpolation=cv2.INTER_NEAREST).astype(bool)

            GT_depth = np.expand_dims(GT_depth, axis=-1)

            GT_pointcloud = np.asarray(o3d.io.read_point_cloud(GT_pointcloud_path).points, dtype="f4") 
            input_depth_and_GT_depth_mask = input_depth_mask & GT_depth_mask

            GT_pointcloud_vs_input_pointcloud_acc_mean, GT_pointcloud_vs_input_pointcloud_acc_median, GT_pointcloud_vs_input_pointcloud_acc_inlier_ratio = metrics.pointcloud_accuracy(GT_pointcloud, input_pointcloud)
            GT_pointcloud_vs_input_pointcloud_comp_mean, GT_pointcloud_vs_input_pointcloud_comp_median, GT_pointcloud_vs_input_pointcloud_comp_inlier_ratio = metrics.pointcloud_completion(GT_pointcloud, input_pointcloud)
            
            reproject_input_depth_pointcloud_acc_mean, reproject_input_depth_pointcloud_acc_median, reproject_input_depth_pointcloud_acc_inlier_ratio = metrics.pointcloud_accuracy(GT_pointcloud, reproject_input_depth_pointcloud)
            reproject_input_depth_pointcloud_comp_mean, reproject_input_depth_pointcloud_comp_median, reproject_input_depth_pointcloud_comp_inlier_ratio = metrics.pointcloud_completion(GT_pointcloud, reproject_input_depth_pointcloud)


            result_dict = {}
            result_dict['basics'] = {
             'sample_idx': i,
             'undistorted_image_name': camera_dataset.undistorted_image_names[i],
             'camera_id': camera_dataset.config.id,
             'input_geometry': config.GT_geometry_dir_name, 
             'GT_geoemtry': config.input_geometry_dir
            }

            result_dict['metrics'] = {
                'num_valid_input_depth': int(input_depth_mask.sum()),
                'num_valid_GT_depth': int(GT_depth_mask.sum()),
                'num_input_depth_vs_GT_depth': int(input_depth_and_GT_depth_mask.sum()),
                'num_input_pointcloud': input_pointcloud.shape[0],
                'num_GT_pointcloud': GT_pointcloud.shape[0],
                'GT_depth_vs_input_depth_rel_inlier_ratio':metrics.rel_thresh_inliers(GT_depth, input_depth, mask = input_depth_and_GT_depth_mask),
                'GT_depth_vs_input_depth_m_rel_ae':metrics.m_rel_ae(GT_depth, input_depth, mask = input_depth_and_GT_depth_mask),
                'GT_depth_vs_input_depth_abs_thresh_inliers': metrics.abs_thresh_inliers(GT_depth, input_depth, mask = input_depth_and_GT_depth_mask),
                'GT_depth_vs_input_depth_m_ae': metrics.m_ae(GT_depth, input_depth, mask = input_depth_and_GT_depth_mask),
                'GT_pointcloud_vs_input_pointcloud_acc_mean': GT_pointcloud_vs_input_pointcloud_acc_mean,
                'GT_pointcloud_vs_input_pointcloud_acc_inlier_ratio': GT_pointcloud_vs_input_pointcloud_acc_inlier_ratio,
                'GT_pointcloud_vs_input_pointcloud_acc_median': GT_pointcloud_vs_input_pointcloud_acc_median,
                'GT_pointcloud_vs_input_pointcloud_comp_mean': GT_pointcloud_vs_input_pointcloud_comp_mean,
                'GT_pointcloud_vs_input_pointcloud_comp_meidan': GT_pointcloud_vs_input_pointcloud_comp_median,
                'GT_pointcloud_vs_input_pointcloud_comp_inlier_ratio': GT_pointcloud_vs_input_pointcloud_comp_inlier_ratio,
                'GT_pointcloud_vs_reproject_input_depth_pointcloud_acc_mean': reproject_input_depth_pointcloud_acc_mean,
                'GT_pointcloud_vs_reproject_input_depth_pointcloud_acc_median': reproject_input_depth_pointcloud_acc_median,
                'GT_pointcloud_vs_reproject_input_depth_pointcloud_acc_inlier_ratio': reproject_input_depth_pointcloud_acc_inlier_ratio,
                'GT_pointcloud_vs_reproject_input_depth_pointcloud_comp_mean': reproject_input_depth_pointcloud_comp_mean,
                'GT_pointcloud_vs_reproject_input_depth_pointcloud_comp_meidan': reproject_input_depth_pointcloud_comp_median,
                'GT_pointcloud_vs_reproject_input_depth_pointcloud_comp_inlier_ratio': reproject_input_depth_pointcloud_comp_inlier_ratio,
            }

            if config.isVis == True:
                undistorted_image = cv2.imread(undistorted_image_path)
                vis_undistorted_image = undistorted_image.copy()
                vis_undistorted_image_for_very_large_z_diff = undistorted_image.copy()

                u, v = np.meshgrid(np.arange(undistorted_image.shape[1]), np.arange(undistorted_image.shape[0]))

                v = v[input_depth_and_GT_depth_mask]
                u = u[input_depth_and_GT_depth_mask]

                input_depth_z = input_depth[input_depth_and_GT_depth_mask]
                GT_depth_z = GT_depth[input_depth_and_GT_depth_mask]
                z_diff = np.abs(input_depth_z - GT_depth_z).squeeze()
                very_large_z_diff_mask = z_diff > 0.1
                v_very_large_z_diff = v[very_large_z_diff_mask]
                u_for_very_large_z_diff = u[very_large_z_diff_mask]
                z_diff_for_very_large_z_diff = z_diff[very_large_z_diff_mask]
                
                depth_colors = single_depths2colors(z_diff, 0.001, 2)
                vis_undistorted_image[v, u] = depth_colors
                depth_colors_for_very_large_z_diff = single_depths2colors(z_diff_for_very_large_z_diff, 0.001, 2)
                vis_undistorted_image_for_very_large_z_diff[v_very_large_z_diff, u_for_very_large_z_diff] = depth_colors_for_very_large_z_diff

                cv2.imshow(f"{camera_dataset.undistorted_image_names[i]}_num_{input_depth_and_GT_depth_mask.sum()}", vis_undistorted_image)
                cv2.imshow(f"{camera_dataset.undistorted_image_names[i]}_for_very_large_z_diff_num_{very_large_z_diff_mask.sum()}", vis_undistorted_image_for_very_large_z_diff)
                cv2.waitKey(0)

            all_result['result_per_view'].append(result_dict)

    for result_dict in all_result['result_per_view']:
        for k,v in result_dict['metrics'].items():
            all_result['overall'][k].append(v)

    
    for k,v in all_result['overall'].items():
        all_result['overall'][k] = float(np.mean(all_result['overall'][k]))


    output_path = os.path.join(config.machine.root_experiments_dir, f"groundtruth_analysis_{config.GT_geometry_dir_name}_vs_{config.input_geometry_dir}")
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "all_result.json"), "w") as f:
        json.dump(all_result, f, indent=4)        
    




@hydra.main(
    version_base=None, config_path="../../configs", config_name="grountruth_analysis"
)
def execute_groundtruth_analysis(cfg: DictConfig):
    # Allow the config to be editable
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))


    # Run the testing
    groundtruth_analysis(cfg)



if __name__ == '__main__':
    execute_groundtruth_analysis()