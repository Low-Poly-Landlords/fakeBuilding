import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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


class CalibrationApp:
    def __init__(self, frames):
        self.frames = frames  # List of (scan_points, image_data)
        self.current_idx = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.35)

        # Sliders
        self.ax_roll = plt.axes([0.15, 0.25, 0.65, 0.03])
        self.ax_pitch = plt.axes([0.15, 0.20, 0.65, 0.03])
        self.ax_yaw = plt.axes([0.15, 0.15, 0.65, 0.03])
        self.ax_z = plt.axes([0.15, 0.10, 0.65, 0.03])

        self.s_roll = Slider(self.ax_roll, 'Roll', -180, 180, valinit=INIT_ROLL)
        self.s_pitch = Slider(self.ax_pitch, 'Pitch', -180, 180, valinit=INIT_PITCH)
        self.s_yaw = Slider(self.ax_yaw, 'Yaw', -180, 180, valinit=INIT_YAW)
        self.s_z = Slider(self.ax_z, 'Height (Z)', -0.5, 0.5, valinit=INIT_Z)

        # Buttons
        self.ax_prev = plt.axes([0.15, 0.025, 0.1, 0.04])
        self.ax_next = plt.axes([0.26, 0.025, 0.1, 0.04])
        self.b_prev = Button(self.ax_prev, 'Previous')
        self.b_next = Button(self.ax_next, 'Next')

        self.b_prev.on_clicked(self.prev_frame)
        self.b_next.on_clicked(self.next_frame)

        # Attach updates
        self.s_roll.on_changed(self.update)
        self.s_pitch.on_changed(self.update)
        self.s_yaw.on_changed(self.update)
        self.s_z.on_changed(self.update)

        self.scatter = None
        self.draw_frame()

    def prev_frame(self, event):
        self.current_idx = (self.current_idx - 1) % len(self.frames)
        self.draw_frame()

    def next_frame(self, event):
        self.current_idx = (self.current_idx + 1) % len(self.frames)
        self.draw_frame()

    def draw_frame(self):
        scan_points, image_data = self.frames[self.current_idx]

        self.ax.clear()
        self.ax.imshow(image_data)
        self.ax.set_title(f"Frame {self.current_idx + 1} / {len(self.frames)} - Check Alignment")
        self.scatter = self.ax.scatter([], [], c=[], cmap='jet', s=3, alpha=0.6)

        self.update(0)

    def update(self, val):
        scan_points, image_data = self.frames[self.current_idx]
        h, w, _ = image_data.shape
        f = (w / 2) / np.tan(np.deg2rad(CAM_FOV) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

        # 1. Build Transform
        r, p, y = self.s_roll.val, self.s_pitch.val, self.s_yaw.val
        z_offset = self.s_z.val

        rot = R.from_euler('xyz', [r, p, y], degrees=True).as_matrix()
        base_fix = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        full_rot = rot @ base_fix

        # 2. Transform Points
        pts_local = scan_points - [INIT_X, INIT_Y, z_offset]
        pts_cam = pts_local @ full_rot.T

        # 3. Project
        z_vals = pts_cam[:, 2]
        valid = z_vals > 0.1

        u = (pts_cam[valid, 0] * K[0, 0] / z_vals[valid]) + K[0, 2]
        v = (pts_cam[valid, 1] * K[1, 1] / z_vals[valid]) + K[1, 2]

        in_view = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u_final = u[in_view]
        v_final = v[in_view]
        z_final = z_vals[valid][in_view]

        # Update Plot
        self.scatter.set_offsets(np.column_stack((u_final, v_final)))
        # Color by depth (Helpful for 3D checking)
        self.scatter.set_array(z_final)
        self.scatter.set_clim(0, 5)  # 0 to 5 meters range

        self.fig.canvas.draw_idle()


def get_frames(filename, num_frames=3):
    print(f"Loading {num_frames} frames from {filename}...")
    reader = make_reader(open(filename, "rb"), decoder_factories=[DecoderFactory()])

    frames = []

    # We want to grab frames spread out over time
    # This is a simple logic: Grab a frame every time we see a 'camera' msg
    # provided we have a recent scan.

    last_scan = None
    frame_interval = 20  # Skip frames to get variety
    count = 0

    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic == "/scan":
            # Store most recent scan
            angles = np.arange(ros_msg.angle_min, ros_msg.angle_max, ros_msg.angle_increment)
            count_pts = min(len(angles), len(ros_msg.ranges))
            r = np.array(ros_msg.ranges[:count_pts])
            a = angles[:count_pts]
            valid = (r > 1.0) & (r < 10.0)
            x = r[valid] * np.cos(a[valid])
            y = r[valid] * np.sin(a[valid])
            z = np.zeros_like(x)
            LIDAR_FIX = R.from_euler('x', 90, degrees=True).as_matrix()
            last_scan = np.column_stack((x, y, z)) @ LIDAR_FIX.T

        elif channel.topic == "/camera/image_raw" and last_scan is not None:
            count += 1
            if count % frame_interval == 0:
                width = getattr(ros_msg, "width", 640)
                height = getattr(ros_msg, "height", 480)
                np_arr = np.frombuffer(ros_msg.data, dtype=np.uint8)
                img = np_arr.reshape((height, width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                frames.append((last_scan, img))
                print(f"  Captured Frame {len(frames)}")

                if len(frames) >= num_frames:
                    break
    return frames


def main():
    frames = get_frames(INPUT_FILE, num_frames=4)
    if not frames:
        print("Error: Could not find frames.")
        return

    print("\n--- INSTRUCTIONS ---")
    print("1. Align dots on Frame 1.")
    print("2. Click 'Next'. The dots will likely be misaligned.")
    print("3. Adjust sliders until it looks good on BOTH Frame 1 and Frame 2.")
    print("4. This 'locks' the 3rd axis.")

    app = CalibrationApp(frames)
    plt.show()

    print("\n--- FINAL VALUES ---")
    print(f"CAM_ROLL  = {app.s_roll.val}")
    print(f"CAM_PITCH = {app.s_pitch.val}")
    print(f"CAM_YAW   = {app.s_yaw.val}")
    print(f"CAM_OFFSET = [{INIT_X}, {INIT_Y}, {app.s_z.val}]")


if __name__ == "__main__":
    main()