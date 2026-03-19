import torch
from prior_depth_anything import PriorDepthAnything





class PriorDepthAnythingWrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        version,
        ckpt_dir,
        conditioned_model_size,
        coarse_only,
        **kwargs,
    ):
        super().__init__()

        self.name = name
        self.model = PriorDepthAnything(
            version=version,
            ckpt_dir=ckpt_dir,
            conditioned_model_size=conditioned_model_size,
            coarse_only=coarse_only,
        )

    def forward(self, frames):

        num_frame = len(frames)

        num_views_per_frame = len(frames[0])
        views = [view for frame in frames for view in frame]

        batch_size_per_view, _, height, width = views[0]["undistorted_image"].shape



        res = []
        for view in views:

            sparse_mask  = view['input_depth'] > self.model.sampler.min_depth
            cover_mask = torch.zeros_like(sparse_mask)
            
            input_view = {
                'images':view['undistorted_image'],
                'prior_depths': view['input_depth'],
                'sparse_depths': view['input_depth'],
                'sparse_masks': sparse_mask,
                'cover_masks': cover_mask,
                'pattern': None,
                'geometric_depths': None,
            }



            output = self.model.forward(
                **input_view
            ) 

            # output = output.reshape((height, width))
            res.append({
                'pred_depth': output,
                'pred_depth_mask': output > 0,
            })

        return res

