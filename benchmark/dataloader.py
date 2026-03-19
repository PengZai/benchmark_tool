import torch
from dataclasses import dataclass
from benchmark.dataset_tools import utils
import torchvision
import cv2
import open3d as o3d 
import os
import numpy as np
from PIL import Image
import PIL
from benchmark.utils.cropping import (
    bbox_from_intrinsics_in_out,
    camera_matrix_of_crop,
    crop_image_and_other_optional_info,
    rescale_image_and_other_optional_info,
)

@dataclass
class ImageNormalization:
    mean: torch.Tensor
    std: torch.Tensor


IMAGE_NORMALIZATION_DICT = {
    "dummy": ImageNormalization(mean=torch.tensor([0.0, 0.0, 0.0]), std=torch.tensor([1.0, 1.0, 1.0])),
    "dinov2": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "dinov3": ImageNormalization(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    "dust3r": ImageNormalization(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])),
}


class CameraDataset:

    def __init__(self, camera_config):
        super().__init__()

        self.config = camera_config
        self.samples = []

        self.undistorted_image_names = os.listdir(camera_config['datapath']['undistorted_images'])
        sorted(self.undistorted_image_names)
        self.input_depth_names = os.listdir(camera_config['datapath']['input_depth'])
        sorted(self.input_depth_names)
        self.GT_depth_names = os.listdir(camera_config['datapath']['GT_depth'])
        sorted(self.GT_depth_names)
        self.GT_pointcloud_names = os.listdir(camera_config['datapath']['GT_pointcloud'])
        sorted(self.GT_pointcloud_names)

        self.readDatasample()


    def readDatasample(self):

        idx = 0
        with open(self.config['datapath']['input_pose'], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ts, x, y, z, qx, qy, qz, qw = map(float, line.split())

                T_w_c = utils.pose_to_T(x, y, z, qx, qy, qz, qw)
                
                datasample = {
                    "ts": ts, 
                    "T_w_c": T_w_c,
                    "undistorted_image_name": self.undistorted_image_names[idx],
                    "input_depth_name": self.input_depth_names[idx],
                    "GT_depth_name": self.GT_depth_names[idx],
                    "GT_pointcloud_name": self.GT_pointcloud_names[idx],

                }
                self.samples.append(datasample)
                idx+=1
      


    def getGTPointCloud(self, sample_idx):
        
        pcd = o3d.io.read_point_cloud(os.path.join(self.config['datapath']['GT_pointcloud'], self.samples[sample_idx]['GT_pointcloud_name'])) 
        GT_pointcloud = np.asarray(pcd.points, dtype="f4")
        return GT_pointcloud




class Testdataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.principal_point_centered = config['dataset_test']['principal_point_centered']
        self.aug_crop = config['dataset_test']['aug_crop']
        self.is_metric_scale = config['dataset_test']['is_metric_scale']

        self.resolution = config['model']['test_resolution']
        self.camera_datasets = []
        self.sample_indexes_per_views_list = []
        self.depth_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
        )


        self.data_norm_type = config['model']['data_norm_type']

        if config['model']['data_norm_type'] in IMAGE_NORMALIZATION_DICT.keys():
            image_norm = IMAGE_NORMALIZATION_DICT[self.data_norm_type]
            self.image_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=image_norm.mean, std=image_norm.std),
                ]
            )

        if config['model']['data_norm_type'] == 'unchange':

            self.image_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.PILToTensor(),
                ]
            )

        for camera_config in config['cameras']:
            self.camera_datasets.append(CameraDataset(config['cameras'][camera_config]))


        self.makeSampleIndexPerViewsInSequential()

    def _crop_resize_if_necessary(
        self,
        image,
        resolution,
        depthmap,
        intrinsics,
        additional_quantities=None,
    ):
        """
        Process an image by downsampling and cropping as needed to match the target resolution.

        This method performs the following operations:
        1. Converts the image to PIL.Image if necessary
        2. Crops the image centered on the principal point if requested
        3. Downsamples the image using high-quality Lanczos filtering
        4. Performs final cropping to match the target resolution

        Args:
            image (numpy.ndarray or PIL.Image.Image): Input image to be processed
            resolution (tuple): Target resolution as (width, height)
            depthmap (numpy.ndarray): Depth map corresponding to the image
            intrinsics (numpy.ndarray): Camera intrinsics matrix (3x3)
            additional_quantities (dict, optional): Additional image-related data to be processed
                                                   alongside the main image with nearest interpolation. Defaults to None.

        Returns:
            tuple: Processed image, depthmap, and updated intrinsics matrix.
                  If additional_quantities is provided, it returns those as well.
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # Cropping centered on the principal point if necessary
        if self.principal_point_centered:
            W, H = image.size
            cx, cy = intrinsics[:2, 2].round().astype(int)
            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                # Skip centered cropping if principal point is outside image bounds
                pass
            else:
                min_margin_x = min(cx, W - cx)
                min_margin_y = min(cy, H - cy)
                left, top = cx - min_margin_x, cy - min_margin_y
                right, bottom = cx + min_margin_x, cy + min_margin_y
                crop_bbox = (left, top, right, bottom)
                # Only perform the centered crop if the crop_bbox is larger than the target resolution
                crop_width = right - left
                crop_height = bottom - top
                if crop_width > resolution[0] and crop_height > resolution[1]:
                    image, depthmap, intrinsics, additional_quantities = (
                        crop_image_and_other_optional_info(
                            image=image,
                            crop_bbox=crop_bbox,
                            depthmap=depthmap,
                            camera_intrinsics=intrinsics,
                            additional_quantities=additional_quantities,
                        )
                    )

        # Get the target resolution for re-scaling
        target_rescale_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_rescale_resolution += self._rng.integers(0, self.aug_crop)

        # High-quality Lanczos down-scaling if necessary
        image, depthmap, intrinsics, additional_quantities = (
            rescale_image_and_other_optional_info(
                image=image,
                output_resolution=target_rescale_resolution,
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                additional_quantities_to_be_resized_with_nearest=additional_quantities,
            )
        )

        # Actual cropping (if necessary)
        new_intrinsics = camera_matrix_of_crop(
            input_camera_matrix=intrinsics,
            input_resolution=image.size,
            output_resolution=resolution,
            offset_factor=0.5,
        )
        crop_bbox = bbox_from_intrinsics_in_out(
            input_camera_matrix=intrinsics,
            output_camera_matrix=new_intrinsics,
            output_resolution=resolution,
        )
        image, depthmap, new_intrinsics, additional_quantities = (
            crop_image_and_other_optional_info(
                image=image,
                crop_bbox=crop_bbox,
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                additional_quantities=additional_quantities,
            )
        )

        # Return the output
        if additional_quantities is not None:
            return image, depthmap, new_intrinsics, additional_quantities
        else:
            return image, depthmap, new_intrinsics

    @staticmethod
    def get_views(views, start_index, require_num_view, wrap=False):

        n = len(views)
        indexes_views = list(range(n))
    
        if n == 0:
            return []
        
        if not wrap:
            return indexes_views[start_index:start_index + require_num_view]


        result = []
        for i in range(require_num_view):
            index = (start_index + i) % n
            result.append(indexes_views[index])
        return result
        
    def makeSampleIndexPerViewsInSequential(self):
        
        
        num_view = self.config['num_view']
        is_scene_back_to_origin = False
        Nsamples = len(self.camera_datasets[0].samples)
        len_sample_indexes_per_views_list = Nsamples if is_scene_back_to_origin else Nsamples - num_view + 1

        if num_view < Nsamples:
            for i in range(len_sample_indexes_per_views_list):
                self.sample_indexes_per_views_list.append(self.get_views(self.camera_datasets[0].samples, i, num_view, wrap = is_scene_back_to_origin))

        else:
            self.sample_indexes_per_views_list.append(self.get_views(self.camera_datasets[0].samples, 0, num_view, wrap = is_scene_back_to_origin))
             
   

    
    def __len__(self):
        return len(self.sample_indexes_per_views_list)
    
      

    def __getitem__(self, idx):
        
        output_frames = []
        sample_indexes_per_views = self.sample_indexes_per_views_list[idx]
        
        for sample_idx in sample_indexes_per_views:

            frame = []

            for camera_dataset in self.camera_datasets:

                undistorted_image = cv2.imread(os.path.join(camera_dataset.config['datapath']['undistorted_images'], camera_dataset.samples[sample_idx]['undistorted_image_name']))
                input_depth = cv2.imread(os.path.join(camera_dataset.config['datapath']['input_depth'], camera_dataset.samples[sample_idx]['input_depth_name']), cv2.IMREAD_UNCHANGED)
                GT_depth = cv2.imread(os.path.join(camera_dataset.config['datapath']['GT_depth'], camera_dataset.samples[sample_idx]['GT_depth_name']), cv2.IMREAD_UNCHANGED)

                intrinsics = camera_dataset.config['undistorted_intrinsics']
                K = np.array([
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1]
                ], dtype=np.float32)

                _, input_depth, _ = self._crop_resize_if_necessary(
                    image=undistorted_image,
                    resolution=self.resolution,
                    depthmap=input_depth,
                    intrinsics=K,
                    additional_quantities=None,
                )

                undistorted_image, GT_depth, K = self._crop_resize_if_necessary(
                    image=undistorted_image,
                    resolution=self.resolution,
                    depthmap=GT_depth,
                    intrinsics=K,
                    additional_quantities=None,
                )

                undistorted_image = self.image_transform(undistorted_image)
                input_depth = self.depth_transform(input_depth)
                GT_depth = self.depth_transform(GT_depth)

                
                # input_depth_mask = input_depth > 0
                # GT_depth = cv2.imread(os.path.join(camera_dataset.config['datapath']['GT_depth'], camera_dataset.samples[idx]['GT_depth_name']), cv2.IMREAD_UNCHANGED)
                # pcd = o3d.io.read_point_cloud(os.path.join(camera_dataset.config['datapath']['GT_pointcloud'], camera_dataset.samples[idx]['GT_pointcloud_name'])) 
                # GT_pointcloud = np.asarray(pcd.points, dtype="f4")

                view_data = {
                    "idx": sample_idx,
                    'name': camera_dataset.samples[sample_idx]['undistorted_image_name'],
                    "camera_id": camera_dataset.config['id'],
                    "camera_name": camera_dataset.config['name'],
                    "dataset": self.config['dataset'],
                    "scene_name": self.config['scene_name'],
                    "ts": camera_dataset.samples[sample_idx]["ts"], 
                    "T_w_c": camera_dataset.samples[sample_idx]["T_w_c"],
                    'data_norm_type': self.data_norm_type,
                    'is_metric_scale': self.is_metric_scale,
                    "intrinsics": torch.tensor(K),
                    "undistorted_image": undistorted_image,
                    "input_depth": input_depth,
                    'GT_depth': GT_depth
                    # "input_depth_mask": input_depth_mask,
                    # "GT_depth": GT_depth,
                    # "GT_pointcloud": GT_pointcloud,
                }

                frame.append(view_data)

            output_frames.append(frame)    

        return output_frames
