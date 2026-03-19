import moderngl
import moderngl_window
import numpy as np
from camera import OrbitDragCameraWindow
from moderngl_window.opengl.vao import VAO
import open3d as o3d




class Visualizer(OrbitDragCameraWindow):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.wnd.mouse_exclusivity = False

        CAM_POINTS =  np.array(
        [
            [0, 0, 0],
            [-1, -1, 1.5],
            [1, -1, 1.5],
            [1, 1, 1.5],
            [-1, 1, 1.5],
            [-0.5, 1, 1.5],
            [0.5, 1, 1.5],
            [0, 1.2, 1.5],
        ]
        ).astype("f4")

        CAM_LINES = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
        )

        CAM_SEGMENTS = []
        for i, j in CAM_LINES:
            CAM_SEGMENTS.append(CAM_POINTS[i])
            CAM_SEGMENTS.append(CAM_POINTS[j])

        CAM_SEGMENTS = np.stack(CAM_SEGMENTS, axis=0)

        # POINTS_PATH = "/media/spiderman/zhipeng_8t1/datasets/BotanicGarden/1018-00/1018_00_img10hz600p/velodyne_lidar/1666059817633529000.pcd"
        # pcd = o3d.io.read_point_cloud(POINTS_PATH)
        # points = np.asarray(pcd.points, dtype="f4") 

        points = np.array([
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, -5.0, 0.0],
                [-5.0, 0.0, 0.0],
            ],
            dtype="f4"
        )

        n_points = points.shape[0]
        colors = np.tile([1.0, 1.0, 1.0], (n_points, 1)).astype("f4")
        alphas = np.ones((n_points, ), dtype="f4")


        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;
                in vec3 in_color0;
                in float in_alpha0;

                uniform mat4 m_proj;
                uniform mat4 m_cam;

                out vec3 color;
                out float alpha;

                void main() {
                    gl_Position = m_proj * m_cam * vec4(in_position, 1.0);
                    color = in_color0;
                    alpha = in_alpha0;
                }
            """,
            fragment_shader="""
                #version 330
                out vec4 fragColor;
                in vec3 color;
                in float alpha;

                void main()
                {

                    if (alpha <= 0)
                        discard;

                    fragColor = vec4(color, 1.0);
                }
            """,
        )

        self.cam_prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec3 in_position;

                uniform mat4 m_proj;
                uniform mat4 m_cam;

                void main() {
                    gl_Position = m_proj * m_cam * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330

                out vec4 fragColor;
                uniform vec4 color;

                void main()
                {
                    fragColor = color;
                }
            """,
        )


        self.pts_buffer = self.ctx.buffer(points.tobytes())
        self.clr_buffer = self.ctx.buffer(colors.tobytes())
        self.valid_buffer = self.ctx.buffer(alphas.tobytes())


        self.points = self.ctx.vertex_array(
            self.prog,
            [
                (self.pts_buffer, "3f", "in_position"),
                (self.clr_buffer, "3f", "in_color0"),
                (self.valid_buffer, "1f", "in_alpha0"),
            ],
        )



        self.cam_prog["color"].value = (1.0, 1.0, 1.0, 1.0)
        cam_segments = CAM_SEGMENTS.astype("f4")
        # cam_segments = np.tile(cam_segments, (n, 1))
        self.cam_buffer = self.ctx.buffer(cam_segments.tobytes())
        self.cams = self.ctx.vertex_array(
            self.cam_prog,
            [
                (self.cam_buffer, "3f", "in_position"),
            ],
        )

        # self.camera.projection.update(near=0.1, far=100.0)
        self.camera.mouse_sensitivity = 0.75
        self.camera.zoom = 1.0

        

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.CULL_FACE | moderngl.DEPTH_TEST)

        # self.ctx.point_size = 2.0

        self.prog["m_proj"].write(self.camera.projection.matrix)
        self.prog["m_cam"].write(self.camera.matrix)

        self.cam_prog["m_proj"].write(self.camera.projection.matrix)
        self.cam_prog["m_cam"].write(self.camera.matrix)

        self.cams.render(mode=moderngl.LINES)
        self.points.render(mode=moderngl.POINTS)



if __name__ == '__main__':



    moderngl_window.run_window_config(Visualizer)



    print("end")