import os
import open3d as o3d
from collections import defaultdict
import numpy as np
import json
from benchmark.dataset_tools.utils import invert_transform
from omegaconf import DictConfig, OmegaConf
from benchmark.dataset_tools.utils import depth2color, save_depth_histogram, depth_range_by_ratio
import cv2


def saveMetricsLogAndResults(config, result_list):

    def make_unique_dir(base_path):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            return base_path

        idx = 1
        while True:
            new_path = f"{base_path}_{idx}"
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                return new_path
            idx += 1

    output_path = os.path.join(config.machine.root_experiments_dir, f"dense_{config.num_view}_view", config.model.model_str)
    output_path = make_unique_dir(output_path)
    point_cloud_path_per_sub_scene = os.path.join(output_path, "point_cloud_per_sub_scene")
    os.makedirs(point_cloud_path_per_sub_scene, exist_ok=True)



    all_result = {
        'overall':defaultdict(list),
        'result_per_scene':[],
    }

    for result_list_per_sub_scene in result_list:

        scene_result = {}
        scene_result['basic'] = defaultdict(list)
        scene_result['metrics'] = defaultdict(list)

        for result_dict in result_list_per_sub_scene:
            
            for k,v in result_dict['basic'].items():
                scene_result['basic'][k].append(result_dict['basic'][k])

            for k,v in result_dict['metrics'].items():
                scene_result['metrics'][k].append(float(result_dict['metrics'][k]))
                # all_result['overall'][k].append(float(result_dict['metrics'][k]))
        

        for k,v in scene_result['metrics'].items():
                
            if k in ['runtime', 'postprocess_time']:
                scene_result['metrics'][k] = float(np.sum(v))
            else:
                scene_result['metrics'][k] = float(np.mean(v))

        all_result['result_per_scene'].append(scene_result)


    for scene_result in all_result['result_per_scene']:
        for k,v in scene_result['metrics'].items():
            all_result['overall'][k].append(float(v))

    for k,v in all_result['overall'].items():
        all_result['overall'][k] = float(np.mean(v))

    with open(os.path.join(output_path, "all_result.json"), "w") as f:
        json.dump(all_result, f, indent=4)        
    

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(config, resolve=True), f, indent=4)      


    def saveResultsAsDepthImage(result_list, output_path):
        
        os.makedirs(output_path, exist_ok=True)
        for result_dict in result_list:
            pred_depth = result_dict['pred']['depth'].squeeze()
            # pred_depth_mask = result_dict['pred']['depth_mask']
            depth_range = depth_range_by_ratio(pred_depth, keep=0.98)
            vis_pred_depth = depth2color(pred_depth.squeeze(), min_depth=depth_range[0], max_depth=depth_range[1])
            cv2.imwrite(os.path.join(output_path, f"pred_depth_{result_dict['basic']['sample_idx']}.png"), vis_pred_depth)
            # save_depth_histogram(
            #     pred_depth,
            #     os.path.join(output_path, f"pred_depth_hist_{result_dict['basic']['sample_idx']}.png"), pred_depth.astype(np.float32)
            # )
            

    def saveResultsAsPointCloud(result_list, output_path):
        color_list = []
        points_w_h_list = []
        sub_scene_name = []
        T_w_first_image = result_list[0]['GT']['T_w_c']
        for result_dict in result_list:

            sub_scene_name.append(str(result_dict['basic']['sample_idx']))
            pcs = result_dict['pred']['pts3d']
            output_depth_mask = result_dict['pred']['depth_mask']
            undistorted_raw_image = result_dict['GT']['undistorted_raw_image']
            T_w_c = result_dict['GT']['T_w_c']
            T_w_c_Trans = T_w_c.T
            pw_h = pcs @ T_w_c_Trans[:-1, :] + T_w_c_Trans[-1:, :]
            pw_h = pw_h[output_depth_mask]
            points_w_h_list.append(pw_h)
            color_list.append(undistorted_raw_image[output_depth_mask].astype('f4')/255.)

        points_w_h = np.vstack(points_w_h_list, dtype='f4')
        colors = np.vstack(color_list, dtype='f4')
        colors = colors[:, [2, 1, 0]]
        points_first_image_h = (invert_transform(T_w_first_image) @ points_w_h.T).T


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_first_image_h[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(output_path, 'pc' + "_" + '_'.join(sub_scene_name)+".pcd"), pcd, write_ascii=False)


    for result_list_per_sub_scene in result_list:

        saveResultsAsPointCloud(result_list_per_sub_scene, point_cloud_path_per_sub_scene)
        saveResultsAsDepthImage(result_list_per_sub_scene, point_cloud_path_per_sub_scene)

    all_results = [result_dict for result_list_per_sub_scene in result_list for result_dict in result_list_per_sub_scene]

    saveResultsAsPointCloud(all_results, output_path)



def updateConfig(config, camera_configs):

    
    config['cameras'] = {}

    for camera_id in config['used_camera_idx_per_view']:
        
        config['cameras']["camera" + str(camera_id)] = camera_configs["camera" + str(camera_id)]
        camera_name = config['cameras']["camera" + str(camera_id)]['name']
        config['cameras']["camera" + str(camera_id)]['datapath'] = {
            "undistorted_images": os.path.join(config['root_data_dir'], config['scene_name'], camera_name, "undistorted_images"),
            "input_pose": os.path.join(config['root_data_dir'], config['scene_name'], camera_name, config['input_pose_name']),
            "input_depth": os.path.join(config['root_data_dir'], config['scene_name'], camera_name, config['input_geometry_dir'], "depth"),
            "input_pointcloud": os.path.join(config['root_data_dir'], config['scene_name'], camera_name, config['input_geometry_dir'], "pointcloud"),
            "GT_depth": os.path.join(config['root_data_dir'], config['scene_name'], camera_name, config['GT_geometry_dir_name'], "depth"),
            "GT_pointcloud": os.path.join(config['root_data_dir'], config['scene_name'], camera_name, config['GT_geometry_dir_name'], "pointcloud")
        }

    

    return config