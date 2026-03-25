import weakref
import moderngl_window
from benchmark.dataset_tools.visualizer import Visualizer
import argparse
from moderngl_window.timers.clock import Timer
import yaml
from benchmark.dataset_tools.datasets import BotanicGarden, TartanAir, PolyTunnel
import open3d as o3d
import os 
import numpy as np
from benchmark.dataset_tools import utils 
import trimesh





    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--configdir", default="configs/dataset_tools/BotanicGarden.yaml", type=str, help="path to configure directory")
    args = parser.parse_args()


    VisualizerConfig = Visualizer

    with open(args.configdir, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

  
    data_source_idx = configs['system']['use_data_source']
    data_source = configs['data_source'+str(data_source_idx)]

    if 'BotanicGarden' in args.configdir:
        dataset = BotanicGarden(configs)
    elif 'TartanAir' in args.configdir:
        dataset = TartanAir(configs)
    elif 'PolyTunnel' in args.configdir:
        dataset = PolyTunnel(configs)


    invalid_data_count = 0
    for sample_idx, sample in enumerate(dataset.samples):
        if sample_idx < configs['system']['start_idx'] or sample_idx > configs['system']['end_idx']:
            continue
        print("processing sample ", sample_idx)

        sample = dataset.loadBenchmarkData(sample)
        if sample == None:
           invalid_data_count+=1
    
    dataset.samples = dataset.samples[configs['system']['start_idx']: configs['system']['end_idx']]
    print("invalid_data_count:",invalid_data_count)   
    


    # if "sensor3dtype" in data_source:
    #     if data_source["sensor3dtype"] == 'lidarpointcloud':
    #         poincloud_processor = LidarPointCloud(configs, dataset)
    #     elif data_source["sensor3dtype"] == 'colorizedpointcloud':
    #         poincloud_processor = ColoarizedPointCloud(configs, dataset)
    #     elif data_source["sensor3dtype"] == 'imagedepth':
    #         poincloud_processor = ImageDepthPointCloud(configs, dataset)

    #     points_w, colors, poses = poincloud_processor.run()

    points_first_image_list = []
    colors_list = []
    pose_list = []
    T_w_first_image = dataset.samples[0]['synchronized_image_data_list'][0]['T_w_cam_idx']
    
    for sample_idx, sample in enumerate(dataset.samples):
        for synchronized_image_data in sample['synchronized_image_data_list']:
            T_w_cam_idx = synchronized_image_data['T_w_cam_idx']
            synchronized_p_c = synchronized_image_data['cumulated_p_c']
            points_c = synchronized_p_c
            points_c_h = np.hstack([points_c, np.ones((points_c.shape[0], 1), dtype=np.float32)])  # (N,4)
            T_first_image_cam_idx = utils.invert_transform(T_w_first_image) @ T_w_cam_idx
            points_first_image_h = (T_first_image_cam_idx @ points_c_h.T).T
            colors = synchronized_image_data['cumulated_p_c_color']
            points_first_image_list.append(points_first_image_h[:, :3])
            colors_list.append(colors)
            pose_list.append(T_first_image_cam_idx)

        # pose_list.append(sample['T_w_p'])


    points_first_image = np.vstack(points_first_image_list, dtype='f4')
    points_first_image = np.ascontiguousarray(points_first_image)

    colors = np.vstack(colors_list, dtype='f4')
    colors = np.ascontiguousarray(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_first_image)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(configs['output']['path'], configs['cameras']['camera'+str(data_source['used_camera_idxes'][0])]['output_relative_path_dict']['name_for_GT_dir'])+'.pcd', pcd, write_ascii=False)
    poses = np.stack(pose_list, dtype="f4")
    
    rgbas = np.hstack([colors, np.ones((colors.shape[0], 1), dtype=np.float32)])  # (N,4)



    if configs['visualization']['isVisualization'] == True:
        VisualizerConfig._points = points_first_image
        VisualizerConfig._rgbas = rgbas
        VisualizerConfig._poses = poses

        window_cls = moderngl_window.get_local_window_cls("glfw")
        window = window_cls(
            title=VisualizerConfig._title,
            size=VisualizerConfig._window_size,
            fullscreen=False,
            resizable=True,
            visible=True,
            gl_version=(3, 3),
            aspect_ratio=None,
            vsync=True,
            samples=4,
            cursor=True,
            backend="glfw",
        )
        window.print_context_info()
        moderngl_window.activate_context(window=window)
        window.ctx.gc_mode = "auto"
        timer = Timer()

        window_config = VisualizerConfig(
            ctx=window.ctx,
            wnd=window,
            timer=timer,
        )
        window._config = weakref.ref(window_config)


        window.swap_buffers()
        window.set_default_viewport()

        timer.start()

        while not window.is_closing:
            current_time, delta = timer.next_frame()

            if window_config.clear_color is not None:
                window.clear(*window_config.clear_color)

            # Always bind the window framebuffer before calling render
            window.use()


            window.render(current_time, delta)
            window.swap_buffers()

    print("end")