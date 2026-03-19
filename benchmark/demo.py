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
    model.to(device)
    # priorda = PriorDepthAnything(device=device)
    # priorda_coarse_only = PriorDepthAnything(device=device, coarse_only=True)


    with torch.no_grad():

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
                    "ts"
                ]
            )
            for frame in frames:
                for view in frame:
                    for name in view.keys():  # pseudo_focal
                        if name in ignore_keys:
                            continue
                        view[name] = view[name].to(device, non_blocking=True)

            results = model(frames)

            result_list = []
            for frame_idx in range(num_frame):
                
                input_view = frames[frame_idx][0]
                res = results[frame_idx]

                for batch_idx in range(batch_size):
                    sample_idx = input_view['idx'][batch_idx].item()
                    result_dict = {
                        'sample_idx': sample_idx,
                        'pred_depth': res['pred_depth'][batch_idx].detach().cpu().numpy(),
                        'pred_depth_mask': res['pred_depth_mask'][batch_idx],
                        'GT_depth': input_view['GT_depth'][batch_idx].cpu().numpy().squeeze(),
                        'T_w_c': input_view['T_w_c'][batch_idx].cpu().numpy(),
                        'intrinsics': input_view['intrinsics'][batch_idx]
                    }
                # result_dict['predicted_point_cloud'] = utils.undistortedDepth2Pointcloud(result_dict['pred_depth'], input_view['intrinsics'].cpu().numpy())

                    result_dict['depth_inlier_ratio'] = metrics.thresh_inliers(result_dict['GT_depth'], result_dict['pred_depth'])
                    result_dict['depthm_rel_ae'] = metrics.m_rel_ae(result_dict['GT_depth'], result_dict['pred_depth'])

                    result_list.append(result_dict)

        #     # result_dict['point_cloud_inlier_ratio'] = metrics.thresh_inliers(result_dict['GT_point_cloud'], result_dict['predicted_point_cloud'])
        #     # result_dict['point_cloud_m_rel_ae'] = metrics.m_rel_ae(result_dict['GT_point_cloud'], result_dict['predicted_point_cloud'])  

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
    

        


