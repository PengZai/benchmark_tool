import dataclasses
from pathlib import Path

import math

import moderngl
import moderngl_window
import numpy as np
# from camera import OrbitDragCameraWindow, CameraWindow
# from moderngl_window.scene import Camera, KeyboardCamera
from in3d.window import WindowEvents
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.viewport_window import ViewportWindow
from in3d.geometry import Axis, AxisList, Pointcloud, LineGeometry, PointGeometry, Frustum
from moderngl_window import resources
import imgui




class Visualizer(WindowEvents):

    _title = "Visualizer"
    _window_size = (1960, 1080)
    _points = None
    _rgbas = None
    _poses = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx.gc_mode = "auto"
        self.wnd.mouse_exclusivity = False
        self.scale = 1.0

        width, height = self.wnd.size
        self.camera = Camera(
            ProjectionMatrix(width, height, 120, width // 2, height // 2, 0.05, 1000),
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        )
        self.viewport = ViewportWindow("Scene", self.camera)
        resources.register_dir((Path(__file__).parent.parent.parent / "thirdparty/in3d/resources").resolve())

        self.line_prog = self.load_program("programs/lines.glsl")
        self.axis = Axis(self.line_prog, 0.1 * self.scale,  self.scale)
        self.axis_list = AxisList(self.line_prog, 0.1 * self.scale, self.scale)

        self.point_prog = self.load_program("programs/points.glsl") 
        # self.points = Pointcloud(self.point_prog)
        self.points = PointGeometry()
        self.points.program = self.point_prog


        self.frustum = Frustum(self.line_prog, np.eye(4, dtype=np.float32), 0.01, 0.2, 60, 1.777, 1.0)

        # self.points.add(points, point_colors)
        
        # colors = np.ones_like(self._points, dtype='f4')
        # colors[:, 1] = 0.0
        # colors[:, 2] = 0.0


        if self._points is not None:
            self.points.points = self.scale * self._points
            self.points.colors = self._rgbas


        

    def render(self, time: float, frametime: float):

        self.viewport.use()
        self.ctx.enable(moderngl.CULL_FACE | moderngl.DEPTH_TEST)
        # self.ctx.point_size = 2.0
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.point_size = 2


        self.axis.render(self.camera)

        for Twp in self._poses:
            self.axis_list.add(Twp)

        # self.axis.render(self.camera)
        # self.cam_prog["m_proj"].write(self.camera.proj_mat.gl_matrix())
        # self.cam_prog["m_cam"].write(self.camera.gl_matrix())
        self.axis_list.render(self.camera)

        if self.points.points is not None:
            self.points.render(self.camera)
        # self.frustum.render(self.camera)
        # self.cams.render(mode=moderngl.LINES)
        self.render_ui()



    def render_ui(self):

        self.wnd.use()
        imgui.new_frame()
        io = imgui.get_io()
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()
        imgui.set_next_window_size(
            window_size[0] / 4, 15 * window_size[1] / 16, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_position(
            32 * self.scale, 32 * self.scale, imgui.FIRST_USE_EVER
        )
        imgui.begin("GUI", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        imgui.end()


        # self.prog["m_proj"].write(self.camera.proj_mat.gl_matrix())
        # self.prog["m_cam"].write(self.camera.gl_matrix())
        # print("self.camera.gl_matrix()", self.camera.gl_matrix())
        # print("self.camera.proj_mat.gl_matrix()", self.camera.proj_mat.gl_matrix())

        imgui.render()
        self.imgui.render(imgui.get_draw_data())
