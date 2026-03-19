import numpy as np
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
import cv2







  


class LidarPointCloud:


    def __init__(self, configs, dataset):
        super().__init__()

        data_source_idx = configs['system']['use_data_source']
        data_source = configs['data_source'+str(data_source_idx)]

        self.configs = configs
        self.data_source = data_source
        self.dataset = dataset


    def run(self,):
        
        points_w_list = []
        pose_list = []
        colors_list = []
        Nsample = len(self.dataset.samples)


        for i, datasample in enumerate(self.dataset.samples):
            print(f"{i}/{Nsample}, processing {datasample.t}")
            T_w_p = datasample.pose
            T_p_l = np.array(self.data_source['T_pose_sensor3d'])
            T_w_l = T_w_p @ T_p_l
            if datasample.sensor3d_name != None:
                pcd = o3d.io.read_point_cloud(os.path.join(self.data_source['sensor3dpath'], datasample.sensor3d_name)) 
                points = np.asarray(pcd.points, dtype="f4")        
                points_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])  # (N,4)
                colors = np.tile([1.0, 1.0, 1.0], (points.shape[0], 1)).astype("f4")
                points_w_h = points_h @ T_w_l.T
                points_w_list.append(points_w_h[:, :3])
                colors_list.append(colors)
                pose_list.append(T_w_i)

        points_w = np.vstack(points_w_list, dtype="f4")   
        colors = np.vstack(colors_list, dtype="f4")   
        poses = np.stack(pose_list, dtype="f4")

        return points_w, colors, poses


class ColoarizedPointCloud:


    def __init__(self, configs, dataset):
        super().__init__()


        self.dataset = dataset


    def run(self,):
        
        points_w_list = []
        pose_list = []
        colors_list = []
        Nsample = len(self.dataset.samples)

        # fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]

        for i, datasample in enumerate(self.dataset.samples):
            print(f"{i}/{Nsample}, processing {datasample.t}")
            T_w_p = datasample.pose
            T_p_l = np.array(self.data_source['T_pose_sensor3d'])
            T_w_l = T_w_p @ T_p_l
            if datasample.sensor3d_name != None and len(datasample.image_name_list) > 0 :
                pcd = o3d.io.read_point_cloud(os.path.join(self.data_source['sensor3dpath'], datasample.sensor3d_name)) 
                points = np.asarray(pcd.points, dtype="f4")        
                points_h = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])  # (N,4)

                for datasample.image_name in datasample.image_name_list:
                    image = cv2.imread(os.path.join(self.data_source['imagepath'], datasample.image_name)) # BGR
                    image = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR) # undistorted image
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB
                    # cv2.imshow("image", image)
                    # cv2.waitKey(0)
                    point_list = []
                    color_list = []
                    for p_l_h in points_h:
                        T_lidar_cam = np.array(self.data_source['T_lidar_cam'], dtype="f4")
                        p_c_h = T_lidar_cam.T @ p_l_h
                        p_w_h = T_w_l @ p_l_h
                        z = p_c_h[2]
                        if z > 0:
                            u = round(p_c_h[0] * fx / p_c_h[2] + cx)
                            v = round(p_c_h[1] * fy / p_c_h[2] + cy)
                            if isInImage(u, v, z, self.data_source["resolution"][0], self.data_source["resolution"][1]) == True:
                                c = image[v,u].astype("f4")/255.
                                color = np.array([c[2], c[1], c[0]], dtype='f4')
                                point_list.append(p_w_h)
                                color_list.append(color)
                    
                    pose_list.append(T_w_p)    
                    colors = np.vstack(color_list, dtype="f4")   
                    points_w_h = np.vstack(point_list, dtype="f4") 
                    points_w_list.append(points_w_h[:, :3])
                    
                    # else:
                    #     colors = np.tile([1.0, 1.0, 1.0], (points.shape[0], 1)).astype("f4")


                    colors_list.append(colors)



        points_w = np.vstack(points_w_list, dtype="f4")   
        colors = np.vstack(colors_list, dtype="f4")   
        poses = np.stack(pose_list, dtype="f4")

        return points_w, colors, poses
            



class ImageDepthPointCloud:


    def __init__(self, configs, dataset):
        super().__init__()

        self.dataset = dataset

    def run(self,):
        
        
        points_w_list = []
        pose_list = []
        point_list = []
        colors_list = []
        Nsample = len(self.dataset.samples)

        fx, fy, cx, cy = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]

        for i, datasample in enumerate(self.dataset.samples):
            print(f"{i}/{Nsample}, processing {datasample.t}")
            T_w_p = datasample.pose
            T_p_c = np.array(self.data_source['T_pose_sensor3d'])
            T_w_c = T_w_p @ T_p_c
            if datasample.sensor3d_name != None and datasample.image_name != None :
                
                if datasample.sensor3d_name.endswith(".tiff"):
                    depth_image = cv2.imread(os.path.join(self.data_source['sensor3dpath'], datasample.sensor3d_name), cv2.IMREAD_UNCHANGED)
                elif datasample.sensor3d_name.endswith(".npy"):
                    depth_image = np.load(os.path.join(self.data_source['sensor3dpath'], datasample.sensor3d_name))

                image = cv2.imread(os.path.join(self.data_source['imagepath'], datasample.image_name)) # BGR
                image = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR) # undistorted image
                for v in range(self.image_h):
                    for u in range(self.image_w):
                        
                        z = depth_image[v,u]
                        if z <= 0 and z > 5:
                            continue
                        x = (u - cx)*z/fx
                        y = (v - cy)*z/fy
                        c = image[v, u].astype("f4")/255.
                        color = np.array([c[2], c[1], c[0]], dtype='f4')
                        p_c_h = np.array([x, y, z, 1], dtype='f4')
                        p_w_h = T_w_c @ p_c_h
                        point_list.append(p_w_h)
                        colors_list.append(color)
            pose_list.append(T_w_p)

        
        colors = np.vstack(colors_list, dtype="f4")   
        points_w_h = np.vstack(point_list, dtype="f4") 
        points_w_list.append(points_w_h[:, :3])


        points_w = np.vstack(points_w_list, dtype="f4")   
        colors = np.vstack(colors_list, dtype="f4")   
        poses = np.stack(pose_list, dtype="f4")

        return points_w, colors, poses
            