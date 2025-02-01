import open3d as o3d
import win32api
import win32gui
import win32con
import time
from typing import Optional, Tuple
import numpy as np

class Scene:
    def __init__(self, point_cloud: Optional[o3d.geometry.PointCloud] = None, points: Optional[int] = None, colors: Optional[int] = None):
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self.window: Optional[o3d.visualization.VisualizerWithKeyCallback] = None
        self.windowWidth: int = 800
        self.windowHeight: int = 600
        self.camera: Optional[Scene.Camera] = None

        if point_cloud is not None:
            self.pcd = point_cloud
        elif points is not None and colors is not None:
            self.createPointCloud(points, colors)

        self.createWindow()
        self.camera = Scene.Camera(self)

    def createPointCloud(self, points: int, colors: int) -> None:
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.random.rand(points, 3))
        self.pcd.colors = o3d.utility.Vector3dVector(np.random.rand(colors, 3))

    def createWindow(self) -> None:
        self.window = o3d.visualization.VisualizerWithKeyCallback()
        self.window.create_window(width=self.windowWidth, height=self.windowHeight)

        screenWidth = win32api.GetSystemMetrics(0)
        screenHeight = win32api.GetSystemMetrics(1)
        centerX = (screenWidth - self.windowWidth) // 2
        centerY = (screenHeight - self.windowHeight) // 2

        hwnd = win32gui.FindWindow(None, "Open3D")
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, centerX, centerY, self.windowWidth, self.windowHeight, win32con.SWP_SHOWWINDOW)

        if self.pcd:
            self.window.add_geometry(self.pcd)

        self.windowHeight -= 48
        self.windowWidth -= 19

    class Camera:
        def __init__(self, scene: 'Scene'):
            self.scene = scene
            self.fps: float = 1 / 60
            self.sensitivity: float = self.fps * 2
            self.forward: float = 0
            self.right: float = 0
            self.up: float = 0
            self.mouseLookX: float = 0
            self.mouseLookY: float = 0
            self.mouseScrollX: float = 0
            self.mouseScrollY: float = 0
            self.mouseYMoreThan180: bool = False
            self.ctr = self.scene.window.get_view_control()
            self.keyboard_controls = Scene.Camera.CameraKeyboardControls(self)
            self.touchpad_controls = Scene.Camera.CameraTouchPadControls(self)
            self.setRenderDistance(0.1, 1000)

        def cameraFront(self, x: float, y: float, z: float) -> None:
            self.ctr.set_front([x, y, z])

        def cameraLookat(self, x: float, y: float, z: float) -> None:
            self.ctr.set_lookat([x, y, z])

        def cameraSetUp(self, x: float, y: float, z: float) -> None:
            self.ctr.set_up([x, y, z])

        def cameraSetZoom(self, zoom: float) -> None:
            self.ctr.set_zoom(zoom)

        def setRenderDistance(self, near: float, far: float) -> None:
            self.ctr.set_constant_z_near(near)
            self.ctr.set_constant_z_far(far)

        def getOriginAndDirection(self) -> dict:
            extrinsicMatrix = self.ctr.convert_to_pinhole_camera_parameters().extrinsic
            rotationMatrix = extrinsicMatrix[:3, :3]
            origin = extrinsicMatrix[:3, 3]

            rightDirection = rotationMatrix[:, 0]
            rightDirection = rightDirection / np.linalg.norm(rightDirection)

            upDirection = rotationMatrix[:, 1]
            upDirection = upDirection / np.linalg.norm(upDirection)

            forwardDirection = -rotationMatrix[:, 2]
            forwardDirection = forwardDirection / np.linalg.norm(forwardDirection)

            return {
                "origin": origin,
                "right": rightDirection,
                "up": upDirection,
                "forward": forwardDirection
            }
        def normalize(self,vec):
            norm = np.linalg.norm(vec)
            if norm == 0: 
                return vec
            return vec / norm
            
        def quatMul(self,q1, q2):
            w = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
            x = q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1]
            y = q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0]
            z = q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3]
            return np.ndarray[(x, y, z, w)]

        def angled_axis(self,angle, point, origin):
            normalized = self.normalize(point - origin)
            half_angle = angle * 0.5
            sin_half_angle = np.sin(half_angle)
            w = np.cos(half_angle)
            x = normalized[0] * sin_half_angle
            y = normalized[1] * sin_half_angle
            z = normalized[2] * sin_half_angle
            return np.array([x, y, z, w])
        
        def inverseQuat(self,q:np.ndarray) :
            return np.ndarray(-q[0],-q[1],-q[2],q[3])
        
        def quatRotate(self,point, origin:np.ndarray, angledAxis):
            point_direction = np.ndarray([point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]])
            rotating_vector = np.ndarray(point_direction[0], point_direction[1], point_direction[2],0)
            rotated_quaternion = self.quatMul(self.quatMul(angledAxis, rotating_vector), self.inverseQuat(angledAxis))
            return np.ndarray(rotated_quaternion[0] + origin[0], rotated_quaternion[1] + origin[1], rotated_quaternion[2] + origin[2])

        class CameraKeyboardControls:
            def __init__(self, camera: 'Scene.Camera'):
                self.camera = camera
                key_actions = {
                    ord('W'): lambda vis: self.key_callback(vis, ord('W')),
                    ord('S'): lambda vis: self.key_callback(vis, ord('S')),
                    ord('A'): lambda vis: self.key_callback(vis, ord('A')),
                    ord('D'): lambda vis: self.key_callback(vis, ord('D')),
                    ord('E'): lambda vis: self.key_callback(vis, ord('E')),
                    ord('Q'): lambda vis: self.key_callback(vis, ord('Q')),
                }
                for key, action in key_actions.items():
                    self.camera.scene.window.register_key_callback(key, action)

            def key_callback(self, vis, key_code: int) -> bool:
                if key_code == ord('W'):
                    self.camera.forward = 3
                elif key_code == ord('S'):
                    self.camera.forward = -3
                elif key_code == ord('D'):
                    self.camera.right = 3
                elif key_code == ord('A'):
                    self.camera.right = -3
                elif key_code == ord('E'):
                    self.camera.up = 3
                elif key_code == ord('Q'):
                    self.camera.up = -3

                self.camera.ctr.camera_local_translate(self.camera.forward, self.camera.right, self.camera.up)
                self.camera.right = 0
                self.camera.forward = 0
                self.camera.up = 0
                return False

        class CameraTouchPadControls:
            def __init__(self, camera: 'Scene.Camera'):
                self.camera = camera
                self.camera.scene.window.register_mouse_move_callback(self.on_mouse_move)
                self.camera.scene.window.register_mouse_scroll_callback(self.on_mouse_scroll)

            def on_mouse_move(self, obj, x: int, y: int) -> None:
                if x > self.camera.scene.windowWidth / 2:
                    self.camera.mouseLookX = (x-(self.camera.scene.windowWidth/2))*self.camera.sensitivity
                else:
                    self.camera.mouseLookX = -((self.camera.scene.windowWidth/2)-x)*self.camera.sensitivity

                if y > self.camera.scene.windowHeight / 2:
                    self.camera.mouseLookY = (y-(self.camera.scene.windowHeight/2))*self.camera.sensitivity
                else:
                    self.camera.mouseLookY = -((self.camera.scene.windowHeight/2)-y)*self.camera.sensitivity

            def on_mouse_scroll(self, obj, x: int, y: int) -> None:
                self.camera.mouseScrollX -= x*self.camera.sensitivity/10
                self.camera.mouseScrollY -= y*self.camera.sensitivity/10
                print(self.camera.getOriginAndDirection())


# Example usage
scene = Scene(o3d.geometry.PointCloud(o3d.io.read_point_cloud(o3d.data.PLYPointCloud().path)))

while True:
    scene.window.poll_events()
    scene.window.update_renderer()
    scene.camera.ctr.camera_local_rotate(scene.camera.mouseLookX, scene.camera.mouseLookY)
    scene.camera.ctr.camera_local_translate(scene.camera.mouseScrollY, scene.camera.mouseScrollX, 0)
    time.sleep(scene.camera.fps)