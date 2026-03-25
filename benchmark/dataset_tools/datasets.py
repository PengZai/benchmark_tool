import numpy as np
import os
import open3d as o3d 
from dataclasses import dataclass
import pytransform3d
import cv2
from benchmark.dataset_tools import utils
from pathlib import Path
import json

from pytransform3d.transformations import (
    transform_from_pq,
    pq_from_transform,
    transform_sclerp
)


class Sensor3dData:
    
    def __init__(self, sensor3d_config):
        super().__init__()
        self.config = sensor3d_config
        self.id = sensor3d_config['id']
        self.sensor3d_names = os.listdir(self.config['sensor3dpath'])
        
        sorted(self.sensor3d_names)

class PointCloud(Sensor3dData):
    def __init__(self, sensor3d_config):
        super().__init__(sensor3d_config)


class ImageDepth(Sensor3dData):
    def __init__(self, sensor3d_config):
        super().__init__(sensor3d_config)


        self.K, self.remap1, self.remap2 = utils.calculateUndistortedRemap(sensor3d_config['distortion_model'], sensor3d_config['resolution'], sensor3d_config['intrinsics'], sensor3d_config['distortion_coeffs'])

        


class CameraData:

    def __init__(self, camera_config):
        super().__init__()
        
        self.config = camera_config
        self.id = camera_config['id']
        self.image_names = os.listdir(self.config['imagepath'])

        sorted(self.image_names)

        self.K, self.remap1, self.remap2 = utils.calculateUndistortedRemap(camera_config['distortion_model'], camera_config['resolution'], camera_config['original_intrinsics'], camera_config['distortion_coeffs'])

        camera_config["undistorted_intrinsics"] = [float(self.K[0,0]), float(self.K[1,1]), float(self.K[0,2]), float(self.K[1,2])]


class Dataset:

    def __init__(self, configs):
        super().__init__()
    
        self.configs = configs
        self.sensor3d_data_list = []
        self.camera_data_lists = []
        self.samples = []

        data_source_idx = configs['system']['use_data_source']
        self.data_source = configs['data_source'+str(data_source_idx)]

        

        if "used_sensor3d_idxes" in self.data_source:

            for used_sensord3d_id in self.data_source['used_sensor3d_idxes']:
                sensor3d_config_i = configs['sensor3ds']['sensor3d' + str(used_sensord3d_id)]
                if sensor3d_config_i['sensor3dtype'] == 'pointcloud':
                    self.sensor3d_data_list.append(PointCloud(sensor3d_config_i))

                elif sensor3d_config_i['sensor3dtype'] == 'imagedepth':
                    self.sensor3d_data_list.append(ImageDepth(sensor3d_config_i))


        if "used_camera_idxes" in self.data_source:
            
            for used_camera_id in self.data_source['used_camera_idxes']:
                camera_config_i = configs['cameras']['camera' + str(used_camera_id)]
                self.camera_data_lists.append(CameraData(camera_config_i))

        
        self.make_output_directories()
        
    def make_output_directories(self):

        output_dir = self.configs['output']['path']
        os.makedirs(output_dir, exist_ok=True)


        data_source_idx = self.configs['system']['use_data_source']
        data_source = self.configs['data_source'+str(data_source_idx)]

        save_config = {
            "system":self.configs["system"],
            "cameras":self.configs["cameras"],
            "sensor3ds":self.configs["sensor3ds"],
            "data_source":data_source
        }


        output_scene_dict = {}
        for used_camera_idx in data_source['used_camera_idxes']:
            
            output_relative_path_dict = {}

            output_relative_path_dict['sparse_depth_relative_path'] = " "
            output_relative_path_dict['sparse_pointcloud_relative_path'] = " "


            camera_config_i = self.configs['cameras']["camera"+str(used_camera_idx)]

            output_image_root_dir = camera_config_i['name']
            os.makedirs(os.path.join(output_dir, output_image_root_dir), exist_ok=True)

            output_relative_path_dict['undistorted_images_path'] = os.path.join(output_image_root_dir, "undistorted_images")
            os.makedirs(os.path.join(output_dir, output_relative_path_dict['undistorted_images_path']), exist_ok=True)

            name_for_GT_dir = ""
            for i, used_sensor3d_idx in enumerate(data_source['used_sensor3d_idxes']):
                sensor3d_config_i = self.configs['sensor3ds']["sensor3d" + str(used_sensor3d_idx)]
                name_for_GT_dir += sensor3d_config_i['name'] + "_" + "c"+ str(data_source['num_cumulation'][i])
                if i < len(data_source['used_sensor3d_idxes']) - 1:
                    name_for_GT_dir += "_"
            

            output_relative_path_dict['name_for_GT_dir'] = name_for_GT_dir
            output_relative_path_dict['GT_pose_output_relative_path'] = os.path.join(output_image_root_dir, name_for_GT_dir)
            os.makedirs(os.path.join(output_dir, output_relative_path_dict['GT_pose_output_relative_path']), exist_ok=True)
            with open(os.path.join(output_dir, output_relative_path_dict['GT_pose_output_relative_path'], "config.json"), "w", encoding="utf-8") as f:
                json.dump(save_config, f, ensure_ascii=False, indent=2)

            with open(os.path.join(output_dir, output_relative_path_dict['GT_pose_output_relative_path'], "Twc.txt"), "w") as f:
                f.write("#timestamp/index x y z q_x q_y q_z q_w\n")
            
            output_relative_path_dict['GT_depth_output_relative_path'] = os.path.join(output_image_root_dir, name_for_GT_dir, "depth")
            os.makedirs(os.path.join(output_dir, output_relative_path_dict['GT_depth_output_relative_path']), exist_ok=True)
            output_relative_path_dict['GT_depth_vis_output_relative_path'] = os.path.join(output_image_root_dir, name_for_GT_dir, "depth_vis")
            os.makedirs(os.path.join(output_dir, output_relative_path_dict['GT_depth_vis_output_relative_path']), exist_ok=True)
            output_relative_path_dict['GT_pointcloud_output_relative_path'] = os.path.join(output_image_root_dir, name_for_GT_dir, "pointcloud")
            os.makedirs(os.path.join(output_dir, output_relative_path_dict['GT_pointcloud_output_relative_path']), exist_ok=True)

        
            self.configs['cameras']['camera'+str(used_camera_idx)]['output_relative_path_dict'] = output_relative_path_dict
            output_scene_dict['camera'+str(used_camera_idx)] = self.configs['cameras']['camera'+str(used_camera_idx)]
    
        if not os.path.exists(os.path.join(output_dir, "scene.json")):
            with open(os.path.join(output_dir, "scene.json"), "w", encoding="utf-8") as f:
                json.dump(output_scene_dict, f, ensure_ascii=False, indent=2)

    def readDatasample(self):
        
        idx = 0
        with open(self.data_source['trajectory_path'], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if idx == 23:
                    print("debug")
                str_ts = line.split()[0]
                ts, x, y, z, qx, qy, qz, qw = map(float, line.split())

                T_w_p = utils.pose_to_T(x, y, z, qx, qy, qz, qw)
                
                datasample = {"id": idx, "ts": ts, "str_ts": str_ts, "T_w_p": T_w_p}
                self.samples.append(datasample)
                idx+=1

        
        # self.samples = self.samples[self.configs['system']['start_idx']:self.configs['system']['end_idx']]


    def getPoses(self):
        
        pose_list = []
        for i, datasample in enumerate(self.samples):
            T_w_p = datasample['T_w_p']
            pose_list.append(T_w_p)

        poses = np.stack(pose_list, dtype="f4")

        return poses



    def getSynchronizedPose(self, sensor_ts):
        # we try to find out time synchronized pose, if it is not perfectly synchronized
        # we will interpolate a new pose in neigghborhood pose using sclerp

        T_w_p = None

        isSync, synchronized_pose_idx = utils.getSynchronizedPoseIdx(sensor_ts, self.samples, 1e-6)
        if isSync == True:
            T_w_p = self.samples[synchronized_pose_idx]['T_w_p']
        else:
            closet_ts = self.samples[synchronized_pose_idx]['ts']
            if sensor_ts > closet_ts and synchronized_pose_idx+1 < len(self.samples):
                ts_start = closet_ts
                T_start = self.samples[synchronized_pose_idx]['T_w_p']
                ts_end = self.samples[synchronized_pose_idx+1]['ts']
                T_end = self.samples[synchronized_pose_idx+1]['T_w_p']
            elif sensor_ts <= closet_ts and synchronized_pose_idx-1 >= 0:
                ts_start = self.samples[synchronized_pose_idx-1]['ts']
                T_start = self.samples[synchronized_pose_idx-1]['T_w_p']
                ts_end = closet_ts
                T_end = self.samples[synchronized_pose_idx]['T_w_p']

            else:
                return None

            t = (sensor_ts - ts_start) / (ts_end - ts_start)
            T_w_p =  transform_sclerp(T_start, T_end, t) 

        return T_w_p

    def loadSyncrhonizedData(self, sample):
        
        pose_idx = sample['id']
        synchronized_image_data_list = []

        for i, camera_data in enumerate(self.camera_data_lists):
      
            T_pose_cam_idx = np.array(self.data_source['T_pose_used_cam_idx' + str(i)], dtype="f4")

            image_name_closest_with_sample_ts = camera_data.image_names[pose_idx]

            T_w_p_syncrhonize_with_image_ts = sample['T_w_p']
            T_w_cam_idx_syncrhonize_with_image_ts = T_w_p_syncrhonize_with_image_ts @ T_pose_cam_idx
            synchronized_image_data_list.append(
                {
                    'name': image_name_closest_with_sample_ts,
                    'ts': str(pose_idx),
                    'ts_diff_with_sample_ts': 0,
                    'camera_id': camera_data.id,
                    'T_w_cam_idx' : T_w_cam_idx_syncrhonize_with_image_ts
                }
            )

        sample['synchronized_image_data_list'] = synchronized_image_data_list

        synchronized_sensor3d_data_list_list = []
        for idx, sensor3d_data in enumerate(self.sensor3d_data_list):
            
            
            synchronized_sensor3d_data_list = []
            idx_closest_with_sample_ts = pose_idx
            sensor3d_name_closest_with_sample_ts = sensor3d_data.sensor3d_names[idx_closest_with_sample_ts]
            T_w_p_syncrhonize_with_sensor3d_ts = sample['T_w_p']
            T_p_sensor3d = np.array(self.data_source['T_pose_used_sensor3d_idx' + str(self.data_source['used_sensor3d_idxes'][idx])])
            T_w_sensor3d_syncrhonize_with_sensor3d_ts = T_w_p_syncrhonize_with_sensor3d_ts @ T_p_sensor3d

            sample['synchronized_sensor3d_data'] = {
                'name': sensor3d_name_closest_with_sample_ts,
                'ts_diff_with_sample_ts': 0,
                'T_w_sensor3d': T_w_sensor3d_syncrhonize_with_sensor3d_ts
            }

            # forward cumulation
            if self.data_source['num_cumulation'][idx] > 0:
                upper_sensor3d_idx = idx_closest_with_sample_ts + self.data_source['num_cumulation'][idx]+1
                if upper_sensor3d_idx > len(sensor3d_data.sensor3d_names) - 1:
                    upper_sensor3d_idx = len(sensor3d_data.sensor3d_names) - 1
                
                # backward cumulation
                lower_sensor3d_idx = idx_closest_with_sample_ts - self.data_source['num_cumulation'][idx]
                if lower_sensor3d_idx < 0:
                    lower_sensor3d_idx = 0
                
                if upper_sensor3d_idx > lower_sensor3d_idx:
                    cumulated_sensor3d_name_list = sensor3d_data.sensor3d_names[lower_sensor3d_idx:upper_sensor3d_idx]
                    cumulated_samples = self.samples[lower_sensor3d_idx:upper_sensor3d_idx]

                    for i, sensor3d_name in enumerate(cumulated_sensor3d_name_list):

                        T_w_p = cumulated_samples[i]['T_w_p']
                        T_w_cumulated_sensor3d_syncrhonize_with_sensor3d_ts = T_w_p @ T_p_sensor3d

                        synchronized_sensor3d_data_list.append({
                            'name': sensor3d_name,
                            'ts_diff_with_sample_ts': 0,
                            'T_w_sensor3d' : T_w_cumulated_sensor3d_syncrhonize_with_sensor3d_ts
                        })

            
            else:
                synchronized_sensor3d_data_list.append({
                    'name': sensor3d_name_closest_with_sample_ts,
                    'ts_diff_with_sample_ts': 0,
                    'T_w_sensor3d': T_w_sensor3d_syncrhonize_with_sensor3d_ts
                })

            synchronized_sensor3d_data_list_list.append(synchronized_sensor3d_data_list)
            
    
        sample['synchronized_sensor3d_data_list_list'] = synchronized_sensor3d_data_list_list

    def loadAsyncrhonizedData(self, sample):

        # if sample['str_ts'] == "1666059841.050277948":
        #     print("debug")
        synchronized_image_data_list = []

        for i, camera_data in enumerate(self.camera_data_lists):

            T_pose_cam_idx = np.array(self.data_source['T_pose_used_cam_idx' + str(i)], dtype="f4")

            image_idx_closest_with_sample_ts = utils.getSensorIdxWithClosestTimeStamp(sample['ts'], camera_data.image_names)
            if image_idx_closest_with_sample_ts == -1:
                return None

            image_name_closest_with_sample_ts = camera_data.image_names[image_idx_closest_with_sample_ts]
            image_ts = utils.timestamp_str_to_float(image_name_closest_with_sample_ts.split(".")[0])
            T_w_p_syncrhonize_with_image_ts = self.getSynchronizedPose(image_ts)
            if T_w_p_syncrhonize_with_image_ts is None:
                return None
            
            T_w_cam_idx_syncrhonize_with_image_ts = T_w_p_syncrhonize_with_image_ts @ T_pose_cam_idx

            synchronized_image_data_list.append(
                {
                    'name': image_name_closest_with_sample_ts,
                    'ts': image_ts,
                    'ts_diff_with_sample_ts': abs(sample['ts'] - image_ts),
                    'camera_id': camera_data.id,
                    'T_w_cam_idx' : T_w_cam_idx_syncrhonize_with_image_ts
                }
            )

        sample['synchronized_image_data_list'] = synchronized_image_data_list


        synchronized_sensor3d_data_list_list = []
        for idx, sensor3d_data in enumerate(self.sensor3d_data_list):
            
            idx_closest_with_sample_ts = utils.getSensorIdxWithClosestTimeStamp(sample['ts'], sensor3d_data.sensor3d_names)
            if idx_closest_with_sample_ts == -1:
                return None
            synchronized_sensor3d_data_list = []
            T_p_sensor3d = np.array(self.data_source['T_pose_used_sensor3d_idx' + str(self.data_source['used_sensor3d_idxes'][idx])])

            # forward cumulation
            if self.data_source['num_cumulation'][idx] > 0:
                upper_sensor3d_idx = idx_closest_with_sample_ts + self.data_source['num_cumulation'][idx]+1
                if upper_sensor3d_idx > len(sensor3d_data.sensor3d_names) - 1:
                    upper_sensor3d_idx = len(sensor3d_data.sensor3d_names) - 1
                
                # backward cumulation
                lower_sensor3d_idx = idx_closest_with_sample_ts - self.data_source['num_cumulation'][idx]
                if lower_sensor3d_idx < 0:
                    lower_sensor3d_idx = 0
                
                if upper_sensor3d_idx > lower_sensor3d_idx:
                    cumulated_sensor3d_name_list = sensor3d_data.sensor3d_names[lower_sensor3d_idx:upper_sensor3d_idx]
                    for sensor3d_name in cumulated_sensor3d_name_list:
                        sensor3d_ts = utils.timestamp_str_to_float(sensor3d_name.split(".")[0])
                        T_w_p = self.getSynchronizedPose(sensor3d_ts)
                        if T_w_p is None:
                            # if there are not synchronized pose, we just past it. it is just culumate point cloud
                            continue
                        T_w_cumulated_sensor3d_syncrhonize_with_sensor3d_ts = T_w_p @ T_p_sensor3d

                        synchronized_sensor3d_data_list.append({
                            'name': sensor3d_name,
                            'ts_diff_with_sample_ts': abs(sample['ts'] - sensor3d_ts),
                            'T_w_sensor3d' : T_w_cumulated_sensor3d_syncrhonize_with_sensor3d_ts
                        })

            
            else:

                sensor3d_name_closest_with_sample_ts = sensor3d_data.sensor3d_names[idx_closest_with_sample_ts]
                sensor3d_ts = utils.timestamp_str_to_float(sensor3d_name_closest_with_sample_ts.split(".")[0])
                T_w_p_syncrhonize_with_sensor3d_ts = self.getSynchronizedPose(sensor3d_ts)
                if T_w_p_syncrhonize_with_sensor3d_ts is None:
                    return None

                T_w_sensor3d_syncrhonize_with_sensor3d_ts = T_w_p_syncrhonize_with_sensor3d_ts @ T_p_sensor3d

                synchronized_sensor3d_data_list.append({
                    'name': sensor3d_name_closest_with_sample_ts,
                    'ts_diff_with_sample_ts': abs(sample['ts'] - sensor3d_ts),
                    'T_w_sensor3d': T_w_sensor3d_syncrhonize_with_sensor3d_ts
                })
                
            synchronized_sensor3d_data_list_list.append(synchronized_sensor3d_data_list)

        sample['synchronized_sensor3d_data_list_list'] = synchronized_sensor3d_data_list_list


    def loadBenchmarkData(self, sample):

        # if sample['str_ts'] == "1666059841.050277948":
        #     print("debug")

        cumulated_points_w_h_list = []
        for sensor3d_i, synchronized_sensor3d_data_list in enumerate(sample['synchronized_sensor3d_data_list_list']):
            
            sensor3d_data = self.sensor3d_data_list[sensor3d_i]
            sensor3d_config_i = self.sensor3d_data_list[sensor3d_i].config

            for synchronized_sensor3d_data in synchronized_sensor3d_data_list:
                
                sensor3d_name = synchronized_sensor3d_data['name']
                synchronized_T_w_p = sample['T_w_p']
                T_w_sensor3d = synchronized_sensor3d_data['T_w_sensor3d']
                suffix = Path(sensor3d_name).suffix

                if sensor3d_config_i['sensor3dtype'] == "pointcloud":
                    if suffix == ".pcd":
                        pcd = o3d.io.read_point_cloud(os.path.join(sensor3d_config_i['sensor3dpath'], sensor3d_name)) 
                        points_sensor3d = np.asarray(pcd.points, dtype="f4")
                        points_sensor3d_h = np.hstack([points_sensor3d, np.ones((points_sensor3d.shape[0], 1), dtype=np.float32)])  # (N,4)

                elif sensor3d_config_i['sensor3dtype'] == "imagedepth":
                    if suffix == ".tiff":
                        depth_image = cv2.imread(os.path.join(sensor3d_config_i['sensor3dpath'], sensor3d_name), cv2.IMREAD_UNCHANGED)
                    if suffix == ".npy":
                        depth_image = np.load(os.path.join(sensor3d_config_i['sensor3dpath'], sensor3d_name))
                    K = sensor3d_data.K
                    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

                    if np.sum(sensor3d_data.config['distortion_coeffs']) > 0:
                        depth_image = cv2.remap(depth_image, sensor3d_data.remap1, sensor3d_data.remap2, cv2.INTER_NEAREST) # undistorted depth
                    
                    # point_sensor3d_h_list = []
                    # for v in range(depth_image.shape[0]):
                    #     for u in range(depth_image.shape[1]):
                    #         z =  depth_image[v, u]
                    #         if z <= 1e-3 or z > 1e3:
                    #             # print(f"{v},{u} invalid z:{z}")
                    #             continue                            
                    #         x = (u - cx)*z/fx
                    #         y = (v - cy)*z/fy
                    #         point_sensor3d_h_list.append(np.array([x,y,z,1], dtype='f4'))

                    # points_sensor3d_h = np.vstack(point_sensor3d_h_list, dtype='f4')

                    v, u = np.indices(depth_image.shape, dtype='f4')
                    mask = (depth_image > 1e-3) & (depth_image <= 1e3)
                    x = (u[mask] - cx) * depth_image[mask] / fx
                    y = (v[mask] - cy) * depth_image[mask] / fy
                    z = depth_image[mask]
                    ones = np.ones_like(z, dtype='f4')
                    points_sensor3d_h = np.stack((x, y, z, ones), axis=1)

                points_w_h = (T_w_sensor3d @ points_sensor3d_h.T).T
                points_p_h = (utils.invert_transform(synchronized_T_w_p) @ points_w_h.T).T
                cumulated_points_w_h_list.append(points_w_h)
            

        cumulated_points_w_h = np.vstack(cumulated_points_w_h_list, dtype="f4")  
        cumulated_points_p_h =  (utils.invert_transform(synchronized_T_w_p) @ cumulated_points_w_h.T).T

        sample['cumulated_points_p'] = cumulated_points_p_h[:, :3]


        for camera_data_idx, synchronized_image_data in enumerate(sample['synchronized_image_data_list']):
            
            camera_config = self.camera_data_lists[camera_data_idx].config
            image_name = synchronized_image_data['name']
            image = cv2.imread(os.path.join(self.camera_data_lists[camera_data_idx].config['imagepath'], image_name)) # BGR
            height, width = image.shape[0], image.shape[1]
            K = self.camera_data_lists[camera_data_idx].K

            undistorted_image = cv2.remap(image, self.camera_data_lists[camera_data_idx].remap1, self.camera_data_lists[camera_data_idx].remap2, cv2.INTER_LINEAR) # undistorted image


            cumulated_sensor3d_depth = np.zeros((undistorted_image.shape[0], undistorted_image.shape[1]), dtype='f4')


            cumulated_p_c_h = (utils.invert_transform(synchronized_image_data['T_w_cam_idx']) @ cumulated_points_w_h.T).T
            cumulated_p_c = cumulated_p_c_h[:, :3]
            K_T = K.T
            cumulated_z = cumulated_p_c[:, 2]
            cumulated_p_c_in_norm_plane = cumulated_p_c @ K_T
            cumulated_uv1 = (cumulated_p_c_in_norm_plane / cumulated_p_c_in_norm_plane[:,-1:]).round().astype(np.int64)
            cumulated_u = cumulated_uv1[:, 0]
            cumulated_v = cumulated_uv1[:, 1]
            mask= (cumulated_z > 1e-3) & (cumulated_z <= 1e3) & (cumulated_u >=0) & (cumulated_u < width) & (cumulated_v >=0) & (cumulated_v < height)

            cumulated_u = cumulated_u[mask]
            cumulated_v = cumulated_v[mask]
            cumulated_z = cumulated_z[mask]
            cumulated_p_c = cumulated_p_c[mask]
            cumulated_p_c_color = undistorted_image[cumulated_v, cumulated_u].astype("f4")/255.
            cumulated_p_c_color = cumulated_p_c_color[:, [2, 1, 0]]

            cumulated_sensor3d_depth[cumulated_sensor3d_depth == 0] = np.inf
            np.minimum.at(cumulated_sensor3d_depth, (cumulated_v, cumulated_u), cumulated_z)
            cumulated_sensor3d_depth[np.isinf(cumulated_sensor3d_depth)] = 0


            # for u,v,z in zip(cumulated_u, cumulated_v, cumulated_z):

            #     if cumulated_sensor3d_depth[v,u] == 0:
            #         cumulated_sensor3d_depth[v,u] = z
            #     else:
            #         if cumulated_sensor3d_depth[v, u] > z:
            #             cumulated_sensor3d_depth[v, u] = z


            if self.configs['output']['isSaveVisualizationDepthImage'] == True:
                cumulated_sensor3d_depth_vis = undistorted_image.copy()
                depth_colors = utils.single_depths2colors(cumulated_z, 0.01, 50)
                cumulated_sensor3d_depth_vis[cumulated_v, cumulated_u] = depth_colors


            # for point_w_h in cumulated_points_w_h:
            #     T_w_cam_idx = synchronized_image_data['T_w_cam_idx']
            #     p_c_h = utils.invert_transform(T_w_cam_idx) @ point_w_h
            #     z = p_c_h[2]
            #     u = round(p_c_h[0] * fx / p_c_h[2] + cx)
            #     v = round(p_c_h[1] * fy / p_c_h[2] + cy)
            #     if z <= 1e-3 or z > 1e3:
            #         # print(f"{v},{u} invalid z:{z}, p_c_h:{p_c_h}")
            #         continue
            #     if utils.isInImage(u, v, z, camera_config["resolution"][0], camera_config["resolution"][1]) == True:
            #         c = undistorted_image[v,u].astype("f4")/255.
            #         c = np.array([c[2], c[1], c[0]], dtype='f4')

            #         cumulated_p_c_list.append(p_c_h[:3])
            #         cumulated_p_c_color_list.append(c)

            #         if cumulated_sensor3d_depth[v, u] == 0:
            #             cumulated_sensor3d_depth[v, u] = z
            #             if self.configs['output']['isSaveVisualizationDepthImage'] == True:
            #                 cumulated_sensor3d_depth_vis[v, u] = utils.single_depth2color(z, 0.01, 50)

            #         else:
            #             if cumulated_sensor3d_depth[v, u] > z:
            #                 cumulated_sensor3d_depth[v, u] = z
                            
            #                 if self.configs['output']['isSaveVisualizationDepthImage'] == True:
            #                     cumulated_sensor3d_depth_vis[v, u] = utils.single_depth2color(z, 0.01, 50)



            # cumulated_p_c = np.vstack(cumulated_p_c_list, dtype="f4")  
            # cumulated_p_c_color = np.vstack(cumulated_p_c_color_list, dtype="f4")  



            synchronized_image_data['undistorted_image'] = undistorted_image
            synchronized_image_data['cumulated_p_c'] = cumulated_p_c
            synchronized_image_data['cumulated_p_c_color'] = cumulated_p_c_color
            synchronized_image_data['cumulated_sensor3d_depth'] = cumulated_sensor3d_depth
            if self.configs['output']['isSaveVisualizationDepthImage'] == True:
                synchronized_image_data['cumulated_sensor3d_depth_vis'] = cumulated_sensor3d_depth_vis


        if self.configs['output']['isOutput'] == True:
            self.writeSample(sample)

        return sample

    
    def writeSample(self, sample):

        output_dir = self.configs['output']['path']

        for camera_data_idx, synchronized_image_data in enumerate(sample['synchronized_image_data_list']):
            
            name_wo_suffix, suffix = os.path.splitext(synchronized_image_data['name'])
            camera_config = self.camera_data_lists[camera_data_idx].config
            output_relative_path_dict = camera_config['output_relative_path_dict']
            path = os.path.join(output_dir, output_relative_path_dict['undistorted_images_path'], synchronized_image_data['name'])
            if not os.path.exists(path):
                cv2.imwrite(path, synchronized_image_data['undistorted_image'])
            path = os.path.join(output_dir, output_relative_path_dict['GT_depth_output_relative_path'], name_wo_suffix + ".tiff")
            if not os.path.exists(path):
                cv2.imwrite(path, synchronized_image_data['cumulated_sensor3d_depth'])
            path = os.path.join(output_dir, output_relative_path_dict['GT_depth_vis_output_relative_path'], name_wo_suffix + ".png")
            if self.configs['output']['isSaveVisualizationDepthImage'] == True and not os.path.exists(path):
                cv2.imwrite(path, synchronized_image_data['cumulated_sensor3d_depth_vis'])
            path = os.path.join(output_dir, output_relative_path_dict['GT_pointcloud_output_relative_path'], name_wo_suffix + ".pcd")
            if not os.path.exists(path):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(synchronized_image_data['cumulated_p_c'])
                pcd.colors = o3d.utility.Vector3dVector(synchronized_image_data['cumulated_p_c_color'])
                o3d.io.write_point_cloud(path, pcd, write_ascii=False)

            with open(os.path.join(output_dir, output_relative_path_dict['GT_pose_output_relative_path'], "Twc.txt"), "a") as f:
                ts = synchronized_image_data['ts']
                t, q = utils.T_to_pose(synchronized_image_data['T_w_cam_idx'])
                if isinstance(ts, str):
                    # write one line
                    f.write(f"{ts} {t[0]:.15g} {t[1]:.15g} {t[2]:.15g} "
                            f"{q[0]:.15g} {q[1]:.15g} {q[2]:.15g} {q[3]:.15g}\n")
                else:
                    # write one line
                    f.write(f"{ts:.9f} {t[0]:.15g} {t[1]:.15g} {t[2]:.15g} "
                            f"{q[0]:.15g} {q[1]:.15g} {q[2]:.15g} {q[3]:.15g}\n")



    

class BotanicGarden(Dataset):


    def __init__(self, configs):
        super().__init__(configs)
        

        self.readDatasample()

        for sample_idx, sample in enumerate(self.samples):
            if sample_idx < configs['system']['start_idx'] or sample_idx > configs['system']['end_idx']:
                continue

            sample = self.loadAsyncrhonizedData(sample)

    



class TartanAir(Dataset):
    def __init__(self, configs):
        super().__init__(configs)


        with open(self.data_source['trajectory_path'], "r", encoding="utf-8") as f:

            idx = 0
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                x, y, z, qx, qy, qz, qw = map(float, line.split())

                T_w_p = utils.pose_to_T(x, y, z, qx, qy, qz, qw)
                
                datasample = {"id":idx, "ts": idx, "T_w_p": T_w_p}
                self.samples.append(datasample)
                idx+=1
        
        for sample_idx, sample in enumerate(self.samples):
            if sample_idx < configs['system']['start_idx'] or sample_idx > configs['system']['end_idx']:
                continue

            sample = self.loadSyncrhonizedData(sample)


    


class PolyTunnel(Dataset):


    def __init__(self, configs):
        super().__init__(configs)
        

        self.readDatasample()

        for sample_idx, sample in enumerate(self.samples):
            if sample_idx < configs['system']['start_idx'] or sample_idx > configs['system']['end_idx']:
                continue

            sample = self.loadAsyncrhonizedData(sample)
