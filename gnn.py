import open3d as o3d
import numpy as np
import win32api
import win32gui
import win32con

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

value = 0

def key_callback(vis, key_code):
    global value
    if key_code == ord('W'):  # 'W' key
        value += 1
        print(f"W pressed. Value increased to: {value}")
    elif key_code == ord('S'):  # 'S' key
        value -= 1
        print(f"S pressed. Value decreased to: {value}")
    elif key_code == ord('A'):  # 'A' key
        value -= 10
        print(f"A pressed. Value decreased to: {value}")
    elif key_code == ord('D'):  # 'D' key
        value += 10
        print(f"D pressed. Value increased to: {value}")
    return False

# Register the key callback
vis.register_key_callback(ord('W'), lambda vis: key_callback(vis, ord('W')))
vis.register_key_callback(ord('S'), lambda vis: key_callback(vis, ord('S')))
vis.register_key_callback(ord('A'), lambda vis: key_callback(vis, ord('A')))
vis.register_key_callback(ord('D'), lambda vis: key_callback(vis, ord('D')))

# Run the visualization loop
print("Press W, A, S, D to modify the value.")
while True:
    vis.poll_events()
    vis.update_renderer()
    if not vis.poll_events():
        break

vis.destroy_window()