import os
import numpy as np
import cv2
import bisect
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# --- CONFIGURATION ---
INPUT_FILE = "newestScan.mcap"

# Base settings (Keep your existing offsets here)
LIDAR_ROLL_OFFSET = 0.0
LIDAR_PITCH_OFFSET = -20.0
LIDAR_YAW_OFFSET = 0.0

CAM_OFFSET = [0.0, 0.0, 0.05]
CAM_FOV_DEG = 70.0
CAM_ROLL = -17.0
CAM_YAW = 0.0

LIDAR_FIX = R.from_euler('xyz', [90 + LIDAR_ROLL_OFFSET, LIDAR_PITCH_OFFSET, LIDAR_YAW_OFFSET],
                         degrees=True).as_matrix()
IMU_FIX = R.from_euler('x', 90, degrees=True)
BASE_ROBOT_TO_CAM = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])


def get_data(filename):
    print(f"Extracting 1 Scan and 1 Image from {filename}...")
    reader = make_reader(open(filename, "rb"), decoder_factories=[DecoderFactory()])

    imu_data = []
    scan_pts = None
    img = None
    found_scan = False

    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic == "/imu/data":
            q = [ros_msg.orientation.x, ros_msg.orientation.y, ros_msg.orientation.z, ros_msg.orientation.w]
            imu_data.append((message.log_time, q))

        elif channel.topic == "/scan" and not found_scan:
            angles = np.arange(ros_msg.angle_min, ros_msg.angle_max, ros_msg.angle_increment)
            count = min(len(angles), len(ros_msg.ranges))
            r = np.array(ros_msg.ranges[:count])
            a = angles[:count]

            valid = (r > 1.0) & (r < 10.0)
            x = r[valid] * np.cos(a[valid])
            y = r[valid] * np.sin(a[valid])
            z = np.zeros_like(x)

            pts = np.column_stack((x, y, z)) @ LIDAR_FIX.T

            # Apply IMU
            times = [x[0] for x in imu_data]
            if len(times) > 0:
                idx = bisect.bisect_left(times, message.log_time)
                idx = min(max(idx, 0), len(times) - 1)
                raw_quat = imu_data[idx][1]
                rot_matrix = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()
                scan_pts = pts @ rot_matrix
            else:
                scan_pts = pts

            found_scan = True

        elif channel.topic == "/camera/image_raw" and found_scan and img is None:
            width = getattr(ros_msg, "width", 640)
            height = getattr(ros_msg, "height", 480)
            np_arr = np.frombuffer(ros_msg.data, dtype=np.uint8)
            img = np_arr.reshape((height, width, 3))
            break

    return scan_pts, img


def main():
    scan_pts, img = get_data(INPUT_FILE)
    if scan_pts is None or img is None:
        print("Could not load data.")
        return

    # 1. Create the Edge Distance Map
    print("Computing Visual Edge Map...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find hard edges in the image
    edges = cv2.Canny(gray, 50, 150)

    # Invert and calculate distance transform (0 at edges, increasing as you move away)
    edges_inv = cv2.bitwise_not(edges)
    dist_map = cv2.distanceTransform(edges_inv, cv2.DIST_L2, 5)

    h, w = dist_map.shape
    focal_length = (w / 2) / np.tan(np.deg2rad(CAM_FOV_DEG) / 2)
    K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])

    # 2. Sweep the Pitch
    print("Sweeping Pitch values from -180 to 180...")
    pitches = np.arange(-180, 180, 1.0)
    scores = []

    for pitch in pitches:
        # Build Camera Transform
        user_cam_fix = R.from_euler('xyz', [CAM_ROLL, pitch, CAM_YAW], degrees=True).as_matrix()
        final_robot_to_cam = user_cam_fix @ BASE_ROBOT_TO_CAM

        # Move points to camera frame
        pts_cam = scan_pts - CAM_OFFSET
        pts_optical = pts_cam @ final_robot_to_cam.T

        # Project to 2D Image
        z = pts_optical[:, 2]
        valid = z > 0.1

        if not np.any(valid):
            scores.append(9999)  # Severe penalty if camera is looking backward
            continue

        u = (pts_optical[valid, 0] * K[0, 0] / z[valid]) + K[0, 2]
        v = (pts_optical[valid, 1] * K[1, 1] / z[valid]) + K[1, 2]

        # Filter points that land inside the image bounds
        in_frame = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        u_in = u[in_frame].astype(int)
        v_in = v[in_frame].astype(int)

        if len(u_in) < 10:
            scores.append(9999)
            continue

        # 3. Calculate Penalty Score (Mean distance to nearest edge)
        penalty = np.mean(dist_map[v_in, u_in])
        scores.append(penalty)

    # 4. Find the Winner
    best_idx = np.argmin(scores)
    best_pitch = pitches[best_idx]

    print("\n" + "=" * 40)
    print(f"OPTIMAL CAM_PITCH FOUND: {best_pitch} degrees")
    print("=" * 40 + "\n")

    # 5. Show the Graph
    plt.figure(figsize=(10, 5))
    plt.plot(pitches, scores, label="Alignment Error")
    plt.axvline(best_pitch, color='r', linestyle='--', label=f"Best Pitch: {best_pitch}°")
    plt.title("Targetless Calibration: Pitch Auto-Tuner")
    plt.xlabel("Pitch Angle (Degrees)")
    plt.ylabel("Penalty Score (Lower is Better)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()