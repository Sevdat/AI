import open3d as o3d
import win32api
import win32gui
import win32con
import time
from typing import Optional, Tuple
import numpy as np
import numba

@numba.jit(nopython=True)
def normalize(vec:np.array) -> np.array:
    norm = length(vec)
    if norm == 0:
        return vec
    return vec / norm


@numba.jit(nopython=True)
def quatMul(q1:np.array, q2:np.array) -> np.array:
    w = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
    x = q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1]
    y = q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0]
    z = q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3]
    return np.array([x, y, z, w])


@numba.jit(nopython=True)
def angledAxis(angle:float, point:np.array, origin:np.array) -> np.array:
    normalized = normalize(point - origin)  # Use cls to call static methods
    halfAngle = angle * 0.5
    sinHalfAngle = np.sin(halfAngle)
    w = np.cos(halfAngle)
    x = normalized[0] * sinHalfAngle
    y = normalized[1] * sinHalfAngle
    z = normalized[2] * sinHalfAngle
    return np.array([x, y, z, w])


@numba.jit(nopython=True)
def inverseQuat(q:np.array) -> np.array:
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quatRotate(point:np.array, origin:np.array, angledAxis:np.array) -> np.array:
    pointDirection = point - origin
    rotatingVector = np.array([pointDirection[0], pointDirection[1], pointDirection[2], 0.0])
    rotatedQuaternion = quatMul(quatMul(angledAxis, rotatingVector), inverseQuat(angledAxis))
    return np.array([rotatedQuaternion[0] + origin[0],rotatedQuaternion[1] + origin[1],rotatedQuaternion[2] + origin[2]])

@numba.jit(nopython=True)        
def convertTo360(angle:float) -> float:
    if angle<0 : 
        return 2*np.pi - (np.abs(angle) % (2*np.pi))
    else: 
        return np.abs(angle) % (2*np.pi)

@numba.jit(nopython=True)      
def length(vectorDirections:np.array) -> float:
    return np.sqrt(np.sum(vectorDirections**2))

@numba.jit(nopython=True) 
def direction(point:np.array, origin:np.array) -> np.array:
    vec = point-origin
    return vec/length(vec)

@numba.jit(nopython=True) 
def distanceFromOrigin(point:np.array, origin:np.array, distance:float) -> np.array:
    return direction(point,origin)*distance

def setPointAroundOrigin(originAndDirection:np.array,angleY:float,angleX:float) -> np.array:
    point = originAndDirection[2]
    angleY = convertTo360(angleY)
    angleX = convertTo360(angleX)
    rotY = angledAxis(angleY,np.array([10,0,0])+originAndDirection[1],originAndDirection[0])
    rotX = angledAxis(angleX,np.array([0,10,0])+originAndDirection[2],originAndDirection[0])
    point = quatRotate(point,originAndDirection[0],rotY)
    point = quatRotate(point,originAndDirection[0],rotX)
    point = originAndDirection[0] + distanceFromOrigin(point,originAndDirection[0],10)
    return point

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
            self.mouseLookX: float = 0
            self.mouseLookY: float = 0
            self.mouseScrollX: float = 0
            self.mouseScrollY: float = 0
            self.ctr = self.scene.window.get_view_control()
            self.keyboard_controls = Scene.Camera.CameraKeyboardControls(self)
            self.touchpad_controls = Scene.Camera.CameraTouchPadControls(self)
            self.setRenderDistance(0.1, 1000)
            self.angleX = 0
            self.angleY = 0
            self.flipped = False

        def cameraFront(self, vec:np.array) -> None:
            self.ctr.set_front(vec)

        def cameraLookat(self, vec:np.array) -> None:
            self.ctr.set_lookat(vec)

        def cameraSetUp(self, vec:np.array) -> None:
            self.ctr.set_up(vec)

        def cameraSetZoom(self, zoom: float) -> None:
            self.ctr.set_zoom(zoom)

        def setRenderDistance(self, near: float, far: float) -> None:
            self.ctr.set_constant_z_near(near)
            self.ctr.set_constant_z_far(far)

        def getOriginAndDirection(self) -> np.array:
            extrinsicMatrix = self.ctr.convert_to_pinhole_camera_parameters().extrinsic
            rotationMatrix = extrinsicMatrix[:3, :3]
            origin = extrinsicMatrix[:3, 3]
            rightDirection = origin+rotationMatrix[:, 0]
            upDirection = origin+rotationMatrix[:, 1]
            forwardDirection = origin-rotationMatrix[:, 2]
            return np.array([origin,rightDirection,upDirection,forwardDirection])
        
        def getCameraForwardVector(self,):
            extrinsicMatrix = self.ctr.convert_to_pinhole_camera_parameters().extrinsic
            forwardVector = extrinsicMatrix[2, :3]
            return forwardVector / np.linalg.norm(forwardVector)

        def getCameraPitch(self):
            extrinsicMatrix = self.ctr.convert_to_pinhole_camera_parameters().extrinsic
            rotationMatrix = extrinsicMatrix[:3, :3]
            pitch = np.arcsin(-rotationMatrix[2, 1])  # Simplified for this example
            return pitch
        
        def flipCameraUpsideDown(self):
            cameraParam = self.ctr.convert_to_pinhole_camera_parameters()
            extrinsicMatrix = cameraParam.extrinsic
            flip_matrix = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            rotate_y_matrix = np.array([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            combined_matrix = np.dot(rotate_y_matrix, flip_matrix)
            cameraParam.extrinsic = np.dot(combined_matrix, extrinsicMatrix)
            self.ctr.convert_from_pinhole_camera_parameters(cameraParam)
            self.flipped = not self.flipped

        class CameraKeyboardControls:
            def __init__(self, camera: 'Scene.Camera'):
                self.camera = camera
                key_actions = {
                    ord('A'): lambda vis: self.key_callback(vis, ord('A')),
                    32: lambda vis: self.key_callback(vis, 32),  # Space bar key
                }
                for key, action in key_actions.items():
                    self.camera.scene.window.register_key_callback(key, action)

            def key_callback(self, vis, key_code: int) -> bool:
                if key_code == ord('W'):
                    ""
                elif key_code == 32:
                    self.camera.flipCameraUpsideDown()
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
                self.camera.mouseScrollX += x*self.camera.sensitivity/10
                self.camera.mouseScrollY -= y*self.camera.sensitivity/10
                


# Example usage
scene2 = Scene(o3d.geometry.PointCloud(o3d.io.read_point_cloud(o3d.data.PLYPointCloud().path)))
PITCH_LIMIT_MIN = np.radians(-60)  # -70 degrees
PITCH_LIMIT_MAX = np.radians(60)   # 70 degrees
while True:
    scene2.window.poll_events()
    scene2.window.update_renderer()
    current = scene2.camera.getCameraPitch()
    print(current*180/np.pi)
    if (scene2.camera.flipped) :
        if current < PITCH_LIMIT_MIN and scene2.camera.mouseLookY>0: scene2.camera.mouseLookY = 0
        if current > PITCH_LIMIT_MAX  and scene2.camera.mouseLookY<0:  scene2.camera.mouseLookY = 0
    else:
        if current < PITCH_LIMIT_MIN and scene2.camera.mouseLookY<0: scene2.camera.mouseLookY = 0
        if current > PITCH_LIMIT_MAX  and scene2.camera.mouseLookY>0:  scene2.camera.mouseLookY = 0
    scene2.camera.ctr.camera_local_translate(scene2.camera.mouseScrollY, scene2.camera.mouseScrollX, 0)
    scene2.camera.ctr.camera_local_rotate(scene2.camera.mouseLookX,scene2.camera.mouseLookY)
    time.sleep(scene2.camera.fps)