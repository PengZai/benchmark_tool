import json
import argparse
import os
from benchmark.dataloader import Testdataset
from torch.utils.data import DataLoader
from benchmark.dataset_tools import utils
import torch
import cv2
import metrics
from benchmark.external import init_model
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from postprocessing import simple_postprocess
import open3d as o3d
from collections import defaultdict
from dataset_tools.utils import invert_transform



def saveMetricsLogAndPointCloud(config, result_list):

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
                all_result['overall'][k].append(float(result_dict['metrics'][k]))
        

        for k,v in scene_result['metrics'].items():
                scene_result['metrics'][k] = float(np.mean(v))

        all_result['result_per_scene'].append(scene_result)

    for k,v in all_result['overall'].items():
        all_result['overall'][k] = float(np.mean(v))

    with open(os.path.join(output_path, "all_result.json"), "w") as f:
        json.dump(all_result, f, indent=4)        
    

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(config, resolve=True), f, indent=4)      


    def saveResultsAsPointCloud(result_list, output_path):
        color_list = []
        points_w_h_list = []
        sub_scene_name = []
        T_w_first_image = result_list[0]['GT']['T_w_c']
        for result_dict in result_list:
            # pred_depth = result_dict['GT']['GT_depth']
            # pred_depth_mask = result_dict['GT']['GT_depth_mask']
            # output_depth = result_dict['pred'][output_depth_name]
            # output_depth_mask = result_dict['pred'][output_depth_mask_name]
            # undistorted_raw_image = result_dict['GT']['undistorted_raw_image']
            # K_matrix =  result_dict['GT']['intrinsics']
            # T_w_c = result_dict['GT']['T_w_c']
            # # T_w_c = result_dict['pred']['T_w_c']
            # sub_scene_name.append(str(result_dict['basic']['sample_idx']))
            # point_c_h_list = []
            # for v in range(output_depth.shape[0]):
            #     for u in range(output_depth.shape[1]):
            #         z =  output_depth[v, u][0]
            #         if output_depth_mask[v,u] == False:
            #             # print(f"{v},{u} invalid z:{z}")
            #             continue                            
            #         x = (u - K_matrix[0,2])*z/K_matrix[0,0]
            #         y = (v - K_matrix[1,2])*z/K_matrix[1,1]
            #         color = undistorted_raw_image[v,u].astype('f4')
            #         color_list.append(color/255.)
            #         point_c_h_list.append(np.array([x,y,z,1], dtype='f4'))

            # point_c_h = np.vstack(point_c_h_list, dtype='f4')
            # points_w_h = (T_w_c @ point_c_h.T).T
            # points_w_h_list.append(points_w_h)

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
            "GT_depth": os.path.join(config['root_data_dir'], config['scene_name'], camera_name, config['GT_geometry_dir_name'], "depth"),
            "GT_pointcloud": os.path.join(config['root_data_dir'], config['scene_name'], camera_name, config['GT_geometry_dir_name'], "pointcloud")
        }

    

    return config

def benchmark(config):


    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    with open(os.path.join(config['root_data_dir'], config['scene_name'], "scene.json"), "r", encoding="utf-8") as f:
        camera_configs = json.load(f)
    
    
    config = updateConfig(config, camera_configs)
    test_dataset = Testdataset(config)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=1)


    # priorda_model = model_factory("priorda", name = "priorda")
    # mapanything_model = model_factory("mapanything", name = "mapanything")

    model = init_model(
        config.model.model_str, config.model.model_config, torch_hub_force_reload=False
    )

    if isinstance(model, torch.nn.Module):
        model.to(device)
    # priorda = PriorDepthAnything(device=device)
    # priorda_coarse_only = PriorDepthAnything(device=device, coarse_only=True)


    with torch.no_grad():

        result_list = []

        for frames in test_dataloader:

            num_frame = len(frames)
            batch_size = frames[0][0]['undistorted_image'].shape[0]

            # Transfer batch to device
            ignore_keys = set(
                [
                    "idx",
                    "name",
                    "camera_id",
                    "camera_name",
                    "dataset",
                    "scene_name",
                    "data_norm_type",
                    "ts",
                    "undistorted_raw_image",
                ]
            )
            for frame in frames:
                for view in frame:
                    for name in view.keys():  # pseudo_focal
                        if name in ignore_keys:
                            continue
                        view[name] = view[name].to(device, non_blocking=True)

            results = model(frames)


            for batch_idx in range(batch_size):

                result_list_per_sub_scene = []
                for frame_idx in range(num_frame):
            
                    input_view = frames[frame_idx][0]
                    res = results[frame_idx]
                    
                    result_dict = {}

                    input_depth = np.expand_dims(input_view['input_depth'][batch_idx].cpu().numpy().squeeze(), axis=-1)
                    input_depth_mask = input_view['input_depth_mask'][batch_idx].cpu().numpy().squeeze()

                    pred_depth = np.expand_dims(res['pred_depth'][batch_idx], axis=-1)
                    pred_depth_mask = res['pred_depth_mask'][batch_idx]

                    GT_depth = np.expand_dims(input_view['GT_depth'][batch_idx].cpu().numpy().squeeze(), axis=-1)
                    GT_depth_mask = input_view['GT_depth_mask'][batch_idx].cpu().numpy().squeeze()

                    result_dict['basic'] = {
                        'sample_idx': input_view['idx'][batch_idx].item(),
                        'dataset_name': input_view['dataset'][batch_idx],
                        'scene_name': input_view['scene_name'][batch_idx],
                        'name': input_view['name'][batch_idx],
                        'camera_name':input_view['camera_name'][batch_idx],
                        'camera_id':input_view['camera_id'][batch_idx].item(),
                        'pred_model': model.name
                    }

                    result_dict['pred'] = {
                        'depth': pred_depth,
                        'depth_mask': pred_depth_mask,
                        'T_w_c': res['pred_T_w_c'][batch_idx]
                    }
                    
                    result_dict['GT'] = {
                        'undistorted_raw_image': input_view['undistorted_raw_image'][batch_idx].numpy(),
                        'input_depth': input_depth,
                        'input_depth_mask': input_depth_mask,
                        'GT_depth': GT_depth,
                        'GT_depth_mask': GT_depth_mask,
                        'T_w_c': input_view['T_w_c'][batch_idx].cpu().numpy(),
                        'intrinsics': input_view['intrinsics'][batch_idx].cpu().numpy()
                    }
   
                # result_dict['predicted_point_cloud'] = utils.undistortedDepth2Pointcloud(result_dict['pred_depth'], input_view['intrinsics'].cpu().numpy())

                   

                    result_list_per_sub_scene.append(result_dict)

                result_list.append(result_list_per_sub_scene)

            simple_postprocess(config, result_list)

        for result_list_per_sub_scene in result_list:

            for result_dict in result_list_per_sub_scene:

                pred_and_input_depth_mask = result_dict['pred']['depth_mask'] & result_dict['GT']['input_depth_mask']
                pred_and_GT_depth_mask = result_dict['pred']['depth_mask'] & result_dict['GT']['GT_depth_mask']

                GT_point_cloud = test_dataset.camera_datasets[result_dict['basic']['camera_id']].getGTPointCloud(result_dict['basic']['sample_idx'])
                pred_pts3d = result_dict['pred']['pts3d'][pred_and_GT_depth_mask]

                acc_mean, acc_median = metrics.pointcloud_accuracy(GT_point_cloud, pred_pts3d)
                comp_mean, comp_median = metrics.pointcloud_completion(GT_point_cloud, pred_pts3d)
                
                result_dict['metrics'] = {
                        'num_valid_pred': result_dict['pred']['depth_mask'].sum(),
                        'num_valid_input_depth': result_dict['GT']['input_depth_mask'].sum(),
                        'num_valid_GT_depth': result_dict['GT']['GT_depth_mask'].sum(),
                        'num_valid_pred_vs_input_depth': pred_and_input_depth_mask.sum(),
                        'num_valid_pred_vs_GT_depth': pred_and_GT_depth_mask.sum(),
                        'input_depth_vs_pred_depth_inlier_ratio':metrics.thresh_inliers(result_dict['GT']['input_depth'], result_dict['pred']['depth'], mask = pred_and_input_depth_mask),
                        'input_depth_vs_pred_depth_m_rel_ae': metrics.m_rel_ae(result_dict['GT']['input_depth'], result_dict['pred']['depth'], mask = pred_and_input_depth_mask),
                        'input_depth_vs_pred_depth_m_ae': metrics.m_ae(result_dict['GT']['input_depth'], result_dict['pred']['depth'], mask = pred_and_input_depth_mask),
                        'GT_depth_vs_pred_depth_inlier_ratio':metrics.thresh_inliers(result_dict['GT']['GT_depth'], result_dict['pred']['depth'], mask = pred_and_GT_depth_mask),
                        'GT_depth_vs_pred_depth_m_rel_ae': metrics.m_rel_ae(result_dict['GT']['GT_depth'], result_dict['pred']['depth'], mask = pred_and_GT_depth_mask),
                        'GT_depth_vs_pred_depth_m_ae': metrics.m_ae(result_dict['GT']['GT_depth'], result_dict['pred']['depth'], mask = pred_and_GT_depth_mask),
                        'GT_depth_vs_pred_depth_pointcloud_acc_mean': acc_mean,
                        'GT_depth_vs_pred_depth_pointcloud_acc_median': acc_median,
                        'GT_depth_vs_pred_depth_pointcloud_comp_mean': comp_mean,
                        'GT_depth_vs_pred_depth_pointcloud_comp_meidan': comp_median,
                    }


        saveMetricsLogAndPointCloud(config, result_list)
        print("end")


@hydra.main(
    version_base=None, config_path="../configs", config_name="dense_n_view_benchmark"
)
def execute_benchmarking(cfg: DictConfig):
    # Allow the config to be editable
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))


    # Run the testing
    benchmark(cfg)



if __name__ == '__main__':
    execute_benchmarking()
    

        


