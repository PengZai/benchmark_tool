import torch
import numpy as np
from dataset_tools.utils import invert_transform


def simple_postprocess(config, result_list):


    # AffineRefinefitting
    if config.model.postprocessing.isAffineRefineDepthWithInputDepth == True:
        for result_list_per_sub_scene in result_list:

            for result_dict in result_list_per_sub_scene:

                result_dict['pred']['depth'] = AffineRefinefitting(result_dict['GT']['input_depth'], result_dict['pred']['depth'], result_dict['GT']['input_depth_mask'] & result_dict['pred']['depth_mask'])
                result_dict['pred']['depth_mask'] = result_dict['pred']['depth_mask'] & (result_dict['pred']['depth'] > 0).squeeze()
                   
    for result_list_per_sub_scene in result_list:

        for result_dict in result_list_per_sub_scene:
            result_dict['pred']['pts3d'] = make_pts3d(result_dict['pred']['depth'], result_dict['GT']['intrinsics'], result_dict['pred']['depth_mask'])



    if config.model.postprocessing.isConsistencyCheck == True:

        consistency_check(result_list)

def AffineRefinefitting(gt, pred, mask):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pred.shape
    with torch.enable_grad():
        # example data
        x = torch.tensor(pred).flatten().to(device)
        y = torch.tensor(gt).flatten().to(device)
        mask = torch.tensor(mask).flatten().to(device)

        # parameters to learn
        a = torch.tensor(1.0, device=device, requires_grad=True)
        b = torch.tensor(0.0, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([a, b], lr=0.05)
        loss_fn = torch.nn.HuberLoss(delta=1.0)

        for _ in range(1000):
            optimizer.zero_grad()
            y_pred = a * x + b
            loss = loss_fn(y_pred[mask], y[mask])
            loss.backward()
            optimizer.step()
        
        refine_pred = a*x + b
        refine_pred = refine_pred.detach().cpu().numpy()

        return refine_pred.reshape(pred.shape)
    

def make_pts3d(depth, K_matrix, mask):

    pts3d = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=depth.dtype)
    
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            z =  depth[v, u][0]
            if mask[v,u] == False:
                continue                            
            x = (u - K_matrix[0,2])*z/K_matrix[0,0]
            y = (v - K_matrix[1,2])*z/K_matrix[1,1]

            pts3d[v,u] = np.array([x,y,z], dtype=depth.dtype)
    
    return pts3d



def consistency_check(result_list, tol_reproject_err = 0.03, num_consistency_num = 2):

    for result_list_per_sub_scene in result_list:

        for i, result_dict_i in enumerate(result_list_per_sub_scene):

            T_w_ci = result_dict_i['GT']['T_w_c']
            pts3d_i = result_dict_i['pred']['pts3d']
            H, W, C = result_dict_i['pred']['depth'].shape

            consistency_check_matrix = np.zeros((H,W), dtype=np.int64)

            for j, result_dict_j in enumerate(result_list_per_sub_scene):

                if result_dict_i['basic']['sample_idx'] == result_dict_j['basic']['sample_idx']: 
                    continue

                T_w_cj = result_dict_j['GT']['T_w_c']
                T_cj_ci = invert_transform(T_w_cj) @ T_w_ci
                T_cj_ci_Trans = T_cj_ci.T
                reproject_pcj_h = pts3d_i @ T_cj_ci_Trans[:-1, :] + T_cj_ci_Trans[-1:, :]
                reproject_pcj = reproject_pcj_h[:, :, :3]
                reproject_depth_j = reproject_pcj_h[:,:, 2]
                K_matrix = result_dict_j['GT']['intrinsics']
                K_matrix_Trans = K_matrix.T
                reproject_pcj_in_norm_plane = reproject_pcj @ K_matrix_Trans
                reproject_uv1 = (reproject_pcj_in_norm_plane / reproject_pcj_in_norm_plane[:,:,-1:]).round().astype(np.int64)
                reproject_u = reproject_uv1[:,:, 0]
                reproject_v = reproject_uv1[:,:, 1]
                mask_for_i = (reproject_depth_j > 0) & (0 <= reproject_u) & (reproject_u < W) & (0 <= reproject_v) & (reproject_v < H)
                mask_for_j = reproject_v[mask_for_i], reproject_u[mask_for_i]
                depth_j = result_dict_j['pred']['depth'][:,:,0]
                acceptable_mask =  np.abs(reproject_depth_j[mask_for_i] - depth_j[mask_for_j]) < tol_reproject_err
                consistency_check_matrix[mask_for_i] += acceptable_mask.astype(np.int64)

            result_dict_i['pred']['depth_mask'] = result_dict_i['pred']['depth_mask'] & (consistency_check_matrix >= num_consistency_num)



