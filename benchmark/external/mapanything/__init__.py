import torch
from mapanything.models import MapAnything




class MapAnythingWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        **kwargs,
    ):
        super().__init__()

        self.name = name

        self.model = MapAnything(name, **kwargs)

    
    def forward(self, frames):

        # convert multi camera in single frame to views
        num_frame = len(frames)

        num_views_per_frame = len(frames[0])
        views = [view for frame in frames for view in frame]

        batch_size_per_view, _, height, width = views[0]["undistorted_image"].shape

        input_views = []
        for view in views:
            input_views.append({
                'img':view['undistorted_image'],
                'intrinsics': view['intrinsics'],
                'camera_poses': view['T_w_c'],
                'depth_z': view['input_depth'].squeeze(1),
                "data_norm_type": view['data_norm_type'],
                'is_metric_scale': view['is_metric_scale'], 
            })
        

        outputs = self.model.infer(
                input_views,
                memory_efficient_inference=True,
                minibatch_size=batch_size_per_view,
                ignore_calibration_inputs=False,  # Whether to use COLMAP calibration or not
                ignore_depth_inputs=False,  # COLMAP doesn't provide depth (can recover from sparse points but convoluted)
                ignore_pose_inputs=False,  # Whether to use COLMAP poses or not
                ignore_depth_scale_inputs=False,  # No depth data
                ignore_pose_scale_inputs=False,  # COLMAP poses are non-metric
                # Use amp for better performance
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
            )
        
        res = []
        for frame_idx in range(num_frame):

            pred_idx = frame_idx*num_views_per_frame
            depth_z = outputs[pred_idx]['depth_z'].squeeze(-1)
            mask = outputs[pred_idx]['mask'].squeeze(-1)
            valid_mask = depth_z > 0.0

            res.append(
                {
                    'pred_depth':depth_z,
                    'pred_depth_mask': mask & valid_mask  # this 1 threshold according to scene.show() visualization setting
                }
            )

        return res