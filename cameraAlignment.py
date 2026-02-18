import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation as R
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import cv2

# --- INPUT ---
INPUT_FILE = "newestScan.mcap"

# --- INITIAL GUESSES ---
INIT_ROLL = 0.0
INIT_PITCH = -90.0
INIT_YAW = 0.0
INIT_X = 0.0
INIT_Y = 0.0
INIT_Z = 0.05
CAM_FOV = 70.0


def get_data(filename):
    print(f"Loading {filename}...")
    reader = make_reader(open(filename, "rb"), decoder_factories=[DecoderFactory()])
    scan_points = None
    image_data = None
    found_img = False
    found_scan = False

    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic == "/scan" and not found_scan:
            angles = np.arange(ros_msg.angle_min, ros_msg.angle_max, ros_msg.angle_increment)
            count = min(len(angles), len(ros_msg.ranges))
            r = np.array(ros_msg.ranges[:count])
            a = angles[:count]
            # Filter for a slice of the room (1m to 10m)
            valid = (r > 1.0) & (r < 10.0)
            x = r[valid] * np.cos(a[valid])
            y = r[valid] * np.sin(a[valid])
            z = np.zeros_like(x)
            # Standard Lidar Fix (Upright)
            LIDAR_FIX = R.from_euler('x', 90, degrees=True).as_matrix()
            scan_points = np.column_stack((x, y, z)) @ LIDAR_FIX.T
            found_scan = True

        elif channel.topic == "/camera/image_raw" and not found_img and found_scan:
            width = getattr(ros_msg, "width", 640)
            height = getattr(ros_msg, "height", 480)
            np_arr = np.frombuffer(ros_msg.data, dtype=np.uint8)
            image_data = np_arr.reshape((height, width, 3))
            # Convert BGR to RGB for Matplotlib
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            found_img = True

        if found_img and found_scan:
            break
    return scan_points, image_data


def main():
    scan_points, image_data = get_data(INPUT_FILE)
    if scan_points is None: return

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.35)  # Make room for sliders

    # Show Image
    ax.imshow(image_data)
    ax.set_title("Align Red Dots (Lidar) to Photo")

    # Initial Scatter (Empty)
    scatter = ax.scatter([], [], c='r', s=2, alpha=0.5)

    # Camera Matrix
    h, w, _ = image_data.shape
    f = (w / 2) / np.tan(np.deg2rad(CAM_FOV) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

    # --- SLIDERS ---
    ax_roll = plt.axes([0.15, 0.25, 0.65, 0.03])
    ax_pitch = plt.axes([0.15, 0.20, 0.65, 0.03])
    ax_yaw = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_z = plt.axes([0.15, 0.10, 0.65, 0.03])

    s_roll = Slider(ax_roll, 'Roll', -180, 180, valinit=INIT_ROLL)
    s_pitch = Slider(ax_pitch, 'Pitch', -180, 180, valinit=INIT_PITCH)
    s_yaw = Slider(ax_yaw, 'Yaw', -180, 180, valinit=INIT_YAW)
    s_z = Slider(ax_z, 'Height (Z)', -0.5, 0.5, valinit=INIT_Z)

    def update(val):
        # 1. Build Transform
        r = s_roll.val
        p = s_pitch.val
        y = s_yaw.val
        z_offset = s_z.val

        rot = R.from_euler('xyz', [r, p, y], degrees=True).as_matrix()

        # Base Align (Robot -> Camera Optical)
        base_fix = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        full_rot = rot @ base_fix

        # 2. Transform Points
        # Apply Translation (Offset) FIRST, then Rotate
        # (This matches your main script logic: pts_robot - OFFSET)
        pts_local = scan_points - [INIT_X, INIT_Y, z_offset]
        pts_cam = pts_local @ full_rot.T

        # 3. Project
        z_vals = pts_cam[:, 2]
        valid = z_vals > 0.1

        u = (pts_cam[valid, 0] * K[0, 0] / z_vals[valid]) + K[0, 2]
        v = (pts_cam[valid, 1] * K[1, 1] / z_vals[valid]) + K[1, 2]

        # Filter to image bounds
        in_view = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u_final = u[in_view]
        v_final = v[in_view]

        # Update Plot
        scatter.set_offsets(np.column_stack((u_final, v_final)))
        fig.canvas.draw_idle()

    # Attach updaters
    s_roll.on_changed(update)
    s_pitch.on_changed(update)
    s_yaw.on_changed(update)
    s_z.on_changed(update)

    # Initial Draw
    update(0)

    print("Adjust sliders until red dots align with walls.")
    print("Close window to see final values.")
    plt.show()

    print("\n--- FINAL VALUES ---")
    print(f"CAM_ROLL  = {s_roll.val}")
    print(f"CAM_PITCH = {s_pitch.val}")
    print(f"CAM_YAW   = {s_yaw.val}")
    print(f"CAM_OFFSET = [{INIT_X}, {INIT_Y}, {s_z.val}]")


if __name__ == "__main__":
    main()