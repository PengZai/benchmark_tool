import weakref
import moderngl_window
from visualizer import Visualizer
import argparse
from moderngl_window.timers.clock import Timer
import yaml
from dense_slam_benchmark.dataset_tools.datasets import build_dataset
import open3d as o3d
import os 
import numpy as np





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--configdir", default="configs/BotanicGarden.yaml", type=str, help="path to configure directory")
    args = parser.parse_args()


    VisualizerConfig = Visualizer

    with open(args.configdir, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

  
    data_source_idx = configs['system']['use_data_source']
    data_source = configs['data_source'+str(data_source_idx)]

    dataset = build_dataset(configs, config_path=args.configdir)








    # if "sensor3dtype" in data_source:
    #     if data_source["sensor3dtype"] == 'lidarpointcloud':
    #         poincloud_processor = LidarPointCloud(configs, dataset)
    #     elif data_source["sensor3dtype"] == 'colorizedpointcloud':
    #         poincloud_processor = ColoarizedPointCloud(configs, dataset)
    #     elif data_source["sensor3dtype"] == 'imagedepth':
    #         poincloud_processor = ImageDepthPointCloud(configs, dataset)

    #     points_w, colors, poses = poincloud_processor.run()

    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points_w)
    #     pcd.colors = o3d.utility.Vector3dVector(colors)
    #     o3d.io.write_point_cloud(os.path.join(configs['system']['output_path'], "out.pcd"), pcd, write_ascii=False)  # binary (smaller/faster)
   
    #     ones = np.ones((colors.shape[0], 1), dtype=colors.dtype)
    #     colors = np.hstack([colors, ones])   # shape (N, 4)
    #     VisualizerConfig._points = points_w
    #     VisualizerConfig._colors = colors

    # else:
    #     poses = dataset.getPoses()

   
    # VisualizerConfig._poses = poses

    # window_cls = moderngl_window.get_local_window_cls("glfw")
    # window = window_cls(
    #     title=VisualizerConfig._title,
    #     size=VisualizerConfig._window_size,
    #     fullscreen=False,
    #     resizable=True,
    #     visible=True,
    #     gl_version=(3, 3),
    #     aspect_ratio=None,
    #     vsync=True,
    #     samples=4,
    #     cursor=True,
    #     backend="glfw",
    # )
    # window.print_context_info()
    # moderngl_window.activate_context(window=window)
    # window.ctx.gc_mode = "auto"
    # timer = Timer()

    # window_config = VisualizerConfig(
    #     ctx=window.ctx,
    #     wnd=window,
    #     timer=timer,
    # )
    # window._config = weakref.ref(window_config)


    # window.swap_buffers()
    # window.set_default_viewport()

    # timer.start()

    # while not window.is_closing:
    #     current_time, delta = timer.next_frame()

    #     if window_config.clear_color is not None:
    #         window.clear(*window_config.clear_color)

    #     # Always bind the window framebuffer before calling render
    #     window.use()


    #     window.render(current_time, delta)
    #     window.swap_buffers()

    # print("end")
