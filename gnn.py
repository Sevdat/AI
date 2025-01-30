import open3d as o3d
import numpy as np
import win32api
import win32gui
import win32con
import time
from scipy.spatial.transform import Rotation as R


class Scene:

    def __init__(self, pointCloud = None, points = None, colors = None):
        global pcd
        global camera
        if pointCloud is not None :
            pcd = o3d.geometry.PointCloud(pointCloud)
        elif points is not None and colors is not None:
            self.create_point_cloud(points,colors)
        self.create_window()
        camera = self.Camera()
    
    def create_point_cloud(self,points,colors):
        global pcd
        pointCloud = o3d.geometry.PointCloud()
        pointCloud.points = o3d.utility.Vector3dVector(np.random.rand(points, 3))
        pointCloud.colors = o3d.utility.Vector3dVector(np.random.rand(colors, 3))
        pcd = o3d.geometry.PointCloud(pointCloud)

    def create_window(self):
        global window
        global window_width
        global window_height
        window_width = 800
        window_height = 600
        window = o3d.visualization.VisualizerWithKeyCallback()
        window.create_window(width=window_width, height=window_height)
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)
        center_x = (screen_width - window_width) // 2
        center_y = (screen_height - window_height) // 2
        hwnd = win32gui.FindWindow(None, "Open3D")
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, center_x, center_y, window_width, window_height, win32con.SWP_SHOWWINDOW)
        window.add_geometry(pcd)
        window_height -= 48
        window_width -= 19

    class Camera:

        def __init__(self):
            global fps 
            global sensitivity 
            global forward 
            global right 
            global up 
            global mouseLookX 
            global mouseLookY 
            global mouseScrollY
            global mouseScrollX
            global mouseYMoreThan180 
            global ctr
            fps = 1/60
            sensitivity = fps*2
            forward = 0
            right = 0
            up = 0
            mouseLookX = 0
            mouseLookY = 0
            mouseScrollX = 0
            mouseScrollY = 0
            mouseYMoreThan180 = 0
            self.camera_controls()
            self.keyboard_controls = self.Camera_Keyboard_Controls()
            self.touchpad_controls = self.Camera_TouchPad_Controls()
            self.set_Render_Distance(0.1,1000)

        def camera_controls(self):
            global ctr
            ctr = window.get_view_control()

        def camera_front(self,x,y,z):
            ctr.set_front([x, y, z])

        def camera_lookat(self,x,y,z):
            ctr.set_lookat([x,y,z])

        def camera_set_Up(self,x,y,z):
            ctr.set_up([x,y,z])

        def camera_set_Up(self,zoom):
            ctr.set_zoom(zoom)
        
        def set_Render_Distance(self,near,far):
            ctr.set_constant_z_near(near)
            ctr.set_constant_z_far(far)
        
        def get_camera_origin_and_orientation(self):
            extrinsic_matrix = ctr.convert_to_pinhole_camera_parameters().extrinsic

            rotation_matrix = extrinsic_matrix[:3, :3]
            origin = extrinsic_matrix[:3, 3]

            right_direction = rotation_matrix[:, 0]
            right_direction = right_direction / np.linalg.norm(right_direction)

            up_direction = rotation_matrix[:, 1]
            up_direction = up_direction / np.linalg.norm(up_direction)

            forward_direction = -rotation_matrix[:, 2]
            forward_direction = forward_direction / np.linalg.norm(forward_direction)

            return {
                "origin": origin,
                "right": right_direction,
                "up": up_direction,
                "forward": forward_direction
            }
        class Camera_Keyboard_Controls:
            
            def __init__(self):
                key_actions = {
                    ord('W'): lambda vis: self.key_callback(vis, ord('W')),
                    ord('S'): lambda vis: self.key_callback(vis, ord('S')),
                    ord('A'): lambda vis: self.key_callback(vis, ord('A')),
                    ord('D'): lambda vis: self.key_callback(vis, ord('D')),
                    ord('E'): lambda vis: self.key_callback(vis, ord('E')),
                    ord('Q'): lambda vis: self.key_callback(vis, ord('Q')),
                }
                for key, action in key_actions.items():
                    window.register_key_callback(key, action)
                
            def key_callback(self,vis, key_code):
                global forward
                global right
                global up
                if key_code == ord('W'):
                    forward = 3
                elif key_code == ord('S'):
                    forward = -3
                elif key_code == ord('D'):
                    right = 3
                elif key_code == ord('A'):
                    right = -3
                elif key_code == ord('E'):
                    up = 3
                elif key_code == ord('Q'):
                    up = -3
                ctr.camera_local_translate(forward,right,up)
                right = 0
                forward = 0
                up = 0
                return False
            
        class Camera_TouchPad_Controls:

            def __init__(self):
                window.register_mouse_move_callback(self.on_mouse_move)
                window.register_mouse_scroll_callback(self.on_mouse_scroll)
            
            def on_mouse_move(self,obj, x, y):
                global mouseLookY
                global mouseLookX

                if (x>window_width/2): mouseLookX = (x-(window_width/2))*sensitivity
                else :  mouseLookX = -((window_width/2)-x)*sensitivity

                if (y>window_height/2): mouseLookY = (y-(window_height/2))*sensitivity
                else : mouseLookY = -((window_height/2)-y)*sensitivity

            def on_mouse_scroll(self,obj, x, y):
                global mouseScrollY
                global mouseScrollX
                mouseScrollX += x*sensitivity/10
                mouseScrollY += y*sensitivity/10
                print(y)
                


Scene(o3d.geometry.PointCloud(o3d.io.read_point_cloud(o3d.data.PLYPointCloud().path)))


while True:
    window.poll_events()
    window.update_renderer()
    ctr.camera_local_rotate(mouseLookX,mouseLookY)
    ctr.camera_local_translate(mouseScrollY,mouseScrollX,0)
    time.sleep(fps)
