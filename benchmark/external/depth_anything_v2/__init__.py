from depth_anything_v2.dpt import DepthAnythingV2
import torch
import time





class DepthAnythingV2Wrapper(torch.nn.Module):
    def __init__(
        self,
        name,
        ckpt_dir,
        encoder,
        **kwargs,
    ):
        super().__init__()
        self.name = name

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.name = name
        self.model = DepthAnythingV2(
            **model_configs[encoder]
        )
        
        self.model.load_state_dict(torch.load(f'{ckpt_dir}/depth_anything_v2_{encoder}.pth', map_location='cpu'))


    def forward(self, frames):

        
        views = [view for frame in frames for view in frame]


        res = []
        for view in views:
            
            input_view = {
                'x':view['undistorted_image'],
            }

            start = time.time()

            inv_depth = self.model.forward(
                **input_view
            ) 

            # detph = 1/(inv_depth+1e-9)
            mask = inv_depth > 0
            depth = torch.zeros_like(inv_depth)
            depth[mask] = 1.0 / inv_depth[mask]
        
            depth = depth.detach().cpu().squeeze(1).numpy()
             
            end = time.time()
            runtime = end - start
            res.append({
                'pred_depth': depth,
                'pred_depth_mask': depth > 0,
                'pred_T_w_c': view['T_w_c'].cpu().numpy(),
                'runtime': runtime,
            })

        

        return res

