import open3d as o3d
import numpy as np
import win32api
import win32gui
import win32con
import time

ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.geometry.PointCloud(o3d.io.read_point_cloud(ply_point_cloud.path))

# Create a visualizer with key callback support
vis = o3d.visualization.VisualizerWithKeyCallback()
window_width = 800  # Width in pixels
window_height = 600  # Height in pixels
vis.create_window(width=window_width, height=window_height)

# Get the screen center
screen_width = win32api.GetSystemMetrics(0)
screen_height = win32api.GetSystemMetrics(1)
center_x = (screen_width - window_width) // 2
center_y = (screen_height - window_height) // 2

# Get the window handle
hwnd = win32gui.FindWindow(None, "Open3D")
if hwnd:
    # Move the window to the center of the screen
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, center_x, center_y, window_width, window_height, win32con.SWP_SHOWWINDOW)

# Add the geometry to the visualizer
vis.add_geometry(pcd)

# Get the view control
ctr = vis.get_view_control()

# This line will obtain the default camera parameters.
camera_params = ctr.convert_to_pinhole_camera_parameters()

# Set camera parameters
ctr.set_front([0.4257, -0.2125, -0.8795])  # Set the front direction of the camera
ctr.set_lookat([2.6172, 2.0475, 1.532])  # Set the point the camera is looking at
ctr.set_up([-0.0694, -0.9768, 0.2024])  # Set the up direction of the camera
ctr.set_zoom(0.3412)  # Set the zoom level


fps = 1/60
sensitivity = fps*5
moveX = 0
moveY = 0
movez = 0

def key_callback(vis, key_code):
    global moveX
    global moveY
    if key_code == ord('W'):  # 'W' key
        moveY = 3
    elif key_code == ord('S'):  # 'S' key
        moveY = -3
    elif key_code == ord('A'):  # 'A' key
        moveX = 3
    elif key_code == ord('D'):  # 'D' key
        moveX = -3
    ctr.camera_local_translate(moveX,moveY)
    moveY = 0
    moveX = 0
    return False

key_actions = {
    ord('W'): lambda vis: key_callback(vis, ord('W')),  # 'W' key
    ord('S'): lambda vis: key_callback(vis, ord('S')),  # 'S' key
    ord('A'): lambda vis: key_callback(vis, ord('A')),  # 'A' key
    ord('D'): lambda vis: key_callback(vis, ord('D')),  # 'D' key
}

# Register all key callbacks using a loop
for key, action in key_actions.items():
    vis.register_key_callback(key, action)

mouseX = 0
mouseY = 0
def mouse_callback(vis, x,y):
    ctr.camera_local_rotate(x,y)
    return False
window_height -= 48
window_width -= 19
def on_mouse_move(vis, x, y):
    global mouseY
    global mouseX
    if (x>window_width/2): mouseX = (x-(window_width/2))*sensitivity
    else :  mouseX = -((window_width/2)-x)*sensitivity

    if (y>window_height/2): mouseY = (y-(window_height/2))*sensitivity
    else : mouseY = -((window_height/2)-y)*sensitivity
    print(f"mouseX {mouseX}, mouseY {mouseY}")

vis.register_mouse_move_callback(on_mouse_move)

while True:
    vis.poll_events()
    vis.update_renderer()
    ctr.camera_local_rotate(mouseX,mouseY)
    time.sleep(fps)
