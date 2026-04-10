import json
import os
from benchmark.benchmark_tools.dataloader import Testdataset
from torch.utils.data import DataLoader
import torch
from benchmark.benchmark_tools import metrics
from benchmark.benchmark_tools.external import init_model
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from benchmark.benchmark_tools.postprocessing import simple_postprocess
from benchmark.benchmark_tools.utils import updateConfig, saveMetricsLogAndResults





def dense_n_view_benchmark(config):


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
        model.eval()
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
                        'T_w_c': res['pred_T_w_c'][batch_idx],
                        'runtime': res['runtime']
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
                input_depth_mask_and_GT_depth_mask = result_dict['GT']['input_depth_mask'] & result_dict['GT']['GT_depth_mask']


                input_pointcloud = test_dataset.camera_datasets[result_dict['basic']['camera_id']].getInputPointCloud(result_dict['basic']['sample_idx'])
                GT_pointcloud = test_dataset.camera_datasets[result_dict['basic']['camera_id']].getGTPointCloud(result_dict['basic']['sample_idx'])
                
                pred_pts3d = result_dict['pred']['pts3d'][pred_and_GT_depth_mask] 

                input_pointcloud_vs_input_pointcloud_acc_mean, input_pointcloud_vs_pred_pointcloud_acc_median, input_pointcloud_vs_pred_pointcloud_acc_inlier_ratio = metrics.pointcloud_accuracy(input_pointcloud, pred_pts3d)
                input_pointcloud_vs_input_pointcloud_comp_mean, input_pointcloud_vs_pred_pointcloud_comp_median, input_pointcloud_vs_pred_pointcloud_comp_inlier_ratio = metrics.pointcloud_completion(input_pointcloud, pred_pts3d)

                GT_pointcloud_vs_pred_pointcloud_acc_mean, GT_pointcloud_vs_pred_pointcloud_acc_median, GT_pointcloud_vs_pred_pointcloud_acc_inlier_ratio = metrics.pointcloud_accuracy(GT_pointcloud, pred_pts3d)
                GT_pointcloud_vs_pred_pointcloud_comp_mean, GT_pointcloud_vs_pred_pointcloud_comp_median, GT_pointcloud_vs_pred_pointcloud_comp_inlier_ratio = metrics.pointcloud_completion(GT_pointcloud, pred_pts3d)

                GT_pointcloud_vs_input_pointcloud_acc_mean, GT_pointcloud_vs_input_pointcloud_acc_median, GT_pointcloud_vs_input_pointcloud_acc_inlier_ratio = metrics.pointcloud_accuracy(GT_pointcloud, input_pointcloud)
                GT_pointcloud_vs_input_pointcloud_comp_mean, GT_pointcloud_vs_input_pointcloud_comp_median, GT_pointcloud_vs_input_pointcloud_comp_inlier_ratio = metrics.pointcloud_completion(GT_pointcloud, input_pointcloud)
                

                result_dict['metrics'] = {
                        'runtime': result_dict['pred']['runtime'],
                        'postprocess_time': result_dict['pred']['postprocess_time'],
                        'num_valid_pred': result_dict['pred']['depth_mask'].sum(),
                        'num_valid_input_depth': result_dict['GT']['input_depth_mask'].sum(),
                        'num_valid_GT_depth': result_dict['GT']['GT_depth_mask'].sum(),
                        'num_pred_pointcloud': pred_pts3d.shape[0],
                        'num_input_pointcloud': input_pointcloud.shape[0],
                        'num_GT_pointcloud': GT_pointcloud.shape[0],                        
                        'num_valid_pred_vs_input_depth': pred_and_input_depth_mask.sum(),
                        'num_valid_pred_vs_GT_depth': pred_and_GT_depth_mask.sum(),
                        'num_input_depth_vs_GT_depth': input_depth_mask_and_GT_depth_mask.sum(),

                        'input_depth_vs_pred_depth_rel_inlier_ratio':metrics.rel_thresh_inliers(result_dict['GT']['input_depth'], result_dict['pred']['depth'], mask = pred_and_input_depth_mask),
                        'input_depth_vs_pred_depth_m_rel_ae': metrics.m_rel_ae(result_dict['GT']['input_depth'], result_dict['pred']['depth'], mask = pred_and_input_depth_mask),
                        'input_depth_vs_pred_depth_m_ae': metrics.m_ae(result_dict['GT']['input_depth'], result_dict['pred']['depth'], mask = pred_and_input_depth_mask),
                        'input_pointcloud_vs_input_pointcloud_acc_mean': input_pointcloud_vs_input_pointcloud_acc_mean,
                        'input_pointcloud_vs_pred_pointcloud_acc_median': input_pointcloud_vs_pred_pointcloud_acc_median,
                        'input_pointcloud_vs_pred_pointcloud_acc_inlier_ratio': input_pointcloud_vs_pred_pointcloud_acc_inlier_ratio,        
                        'input_pointcloud_vs_input_pointcloud_comp_mean': input_pointcloud_vs_input_pointcloud_comp_mean,
                        'input_pointcloud_vs_pred_pointcloud_comp_median': input_pointcloud_vs_pred_pointcloud_comp_median,
                        'input_pointcloud_vs_pred_pointcloud_comp_inlier_ratio': input_pointcloud_vs_pred_pointcloud_comp_inlier_ratio,

                        'GT_depth_vs_input_depth_rel_inlier_ratio':metrics.rel_thresh_inliers(result_dict['GT']['GT_depth'], result_dict['GT']['input_depth'], mask = input_depth_mask_and_GT_depth_mask),
                        'GT_depth_vs_input_depth_m_rel_ae':metrics.m_rel_ae(result_dict['GT']['GT_depth'], result_dict['GT']['input_depth'], mask = input_depth_mask_and_GT_depth_mask),
                        'GT_depth_vs_input_depth_abs_inlier_ratio': metrics.abs_thresh_inliers(result_dict['GT']['GT_depth'], result_dict['GT']['input_depth'], mask = input_depth_mask_and_GT_depth_mask),                    
                        'GT_depth_vs_input_depth_m_ae': metrics.m_ae(result_dict['GT']['GT_depth'], result_dict['GT']['input_depth'], mask = input_depth_mask_and_GT_depth_mask),
                        'GT_pointcloud_vs_input_pointcloud_acc_mean': GT_pointcloud_vs_input_pointcloud_acc_mean,
                        'GT_pointcloud_vs_input_pointcloud_acc_median': GT_pointcloud_vs_input_pointcloud_acc_median,
                        'GT_pointcloud_vs_input_pointcloud_acc_inlier_ratio': GT_pointcloud_vs_input_pointcloud_acc_inlier_ratio,
                        'GT_pointcloud_vs_input_pointcloud_comp_mean': GT_pointcloud_vs_input_pointcloud_comp_mean,
                        'GT_pointcloud_vs_input_pointcloud_comp_meidan': GT_pointcloud_vs_input_pointcloud_comp_median,
                        'GT_pointcloud_vs_input_pointcloud_comp_inlier_ratio': GT_pointcloud_vs_input_pointcloud_comp_inlier_ratio,
                        
                        'GT_depth_vs_pred_depth_rel_inlier_ratio':metrics.rel_thresh_inliers(result_dict['GT']['GT_depth'], result_dict['pred']['depth'], mask = pred_and_GT_depth_mask),
                        'GT_depth_vs_pred_depth_m_rel_ae': metrics.m_rel_ae(result_dict['GT']['GT_depth'], result_dict['pred']['depth'], mask = pred_and_GT_depth_mask),
                        'GT_depth_vs_pred_depth_abs_inlier_ratio': metrics.abs_thresh_inliers(result_dict['GT']['GT_depth'], result_dict['pred']['depth'], mask = pred_and_GT_depth_mask),                                            
                        'GT_depth_vs_pred_depth_m_ae': metrics.m_ae(result_dict['GT']['GT_depth'], result_dict['pred']['depth'], mask = pred_and_GT_depth_mask),
                        'GT_pointcloud_vs_pred_pointcloud_acc_mean': GT_pointcloud_vs_pred_pointcloud_acc_mean,
                        'GT_pointcloud_vs_pred_pointcloud_acc_median': GT_pointcloud_vs_pred_pointcloud_acc_median,
                        'GT_pointcloud_vs_pred_pointcloud_acc_inlier_ratio': GT_pointcloud_vs_pred_pointcloud_acc_inlier_ratio,        
                        'GT_pointcloud_vs_pred_pointcloud_comp_mean': GT_pointcloud_vs_pred_pointcloud_comp_mean,
                        'GT_pointcloud_vs_pred_pointcloud_comp_meidan': GT_pointcloud_vs_pred_pointcloud_comp_median,
                        'GT_pointcloud_vs_pred_pointcloud_comp_inlier_ratio': GT_pointcloud_vs_pred_pointcloud_comp_inlier_ratio

                    }


        saveMetricsLogAndResults(config, result_list)
        print("end")


@hydra.main(
    version_base=None, config_path="../../configs", config_name="dense_n_view_benchmark"
)
def execute_dense_n_view_benchmark(cfg: DictConfig):
    # Allow the config to be editable
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))


    # Run the testing
    dense_n_view_benchmark(cfg)



if __name__ == '__main__':
    execute_dense_n_view_benchmark()
    

        


