import os
import sys
import numpy as np
import cv2
import bisect
from scipy.spatial.transform import Rotation as R
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from mcap_zstd_helper import iter_decoded_messages_with_zstd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MIN_LIDAR_DIST = 1.0
RESOLUTION = 0.05  # 5cm per pixel for calibration

def generate_point_cloud(lidar_msgs, imu_data, pitch_offset):
    """
    Generates a 3D point cloud from LiDAR and IMU data for a given pitch offset.
    """
    global_points = []
    
    lidar_fix = R.from_euler('xyz', [90, pitch_offset, 0], degrees=True).as_matrix()
    imu_fix = R.from_euler('x', 90, degrees=True)

    for log_time, scan_msg in lidar_msgs:
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        count = min(len(angles), len(scan_msg.ranges))
        r = np.array(scan_msg.ranges[:count])
        a = angles[:count]

        valid = (r > MIN_LIDAR_DIST) & (r < scan_msg.range_max)

        x = r[valid] * np.cos(a[valid])
        y = r[valid] * np.sin(a[valid])
        z = np.zeros_like(x)
        pts = np.column_stack((x, y, z))

        pts = pts @ lidar_fix.T
        
        raw_quat = get_interpolated_pose(log_time, imu_data)
        rot_matrix = (R.from_quat(raw_quat) * imu_fix).as_matrix()
        pts = pts @ rot_matrix.T

        global_points.append(pts)

    return np.vstack(global_points)

def get_interpolated_pose(target_time, pose_data):
    """
    Finds the closest IMU pose for a given timestamp.
    """
    times = [x[0] for x in pose_data]
    idx = bisect.bisect_left(times, target_time)
    if idx == 0: return pose_data[0][1]
    if idx == len(times): return pose_data[-1][1]
    
    before = times[idx - 1]
    after = times[idx]
    
    if after - target_time < target_time - before:
        return pose_data[idx][1]
    return pose_data[idx - 1][1]

def calculate_sharpness_score(points):
    """
    Calculates a "sharpness" or "planarity" score for a point cloud.
    A higher score means the walls are flatter and more grid-like.
    """
    if len(points) == 0:
        return 0

    min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])
    mid_z = (max_z + min_z) / 2.0
    slice_thickness = 0.1  # A thin slice for calibration

    slice_mask = (points[:, 2] > (mid_z - slice_thickness)) & (points[:, 2] < (mid_z + slice_thickness))
    slice_pts = points[slice_mask]

    if len(slice_pts) < 100:
        return 0

    min_x, max_x = np.min(slice_pts[:, 0]), np.max(slice_pts[:, 0])
    min_y, max_y = np.min(slice_pts[:, 1]), np.max(slice_pts[:, 1])

    img_w = int((max_x - min_x) / RESOLUTION)
    img_h = int((max_y - min_y) / RESOLUTION)

    if img_w <= 0 or img_h <= 0:
        return 0

    floor_plan = np.zeros((img_h, img_w), dtype=np.uint8)
    px = ((slice_pts[:, 0] - min_x) / RESOLUTION).astype(int)
    py = ((slice_pts[:, 1] - min_y) / RESOLUTION).astype(int)
    
    px = np.clip(px, 0, img_w - 1)
    py = np.clip(py, 0, img_h - 1)

    floor_plan[py, px] = 255

    # Use Canny edge detector to find sharp outlines
    edges = cv2.Canny(floor_plan, 50, 150, apertureSize=3)
    
    # Use Hough Transform to find lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    if lines is None:
        return 0
        
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description="Automatically find the best LiDAR pitch offset.")
    parser.add_argument("input_file", help="Path to the input .mcap file.")
    parser.add_argument("--min_pitch", type=float, default=-25.0, help="Minimum pitch angle to test.")
    parser.add_argument("--max_pitch", type=float, default=-15.0, help="Maximum pitch angle to test.")
    parser.add_argument("--step", type=float, default=0.5, help="Step size for pitch angle search.")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Could not find {args.input_file}")
        return

    print("Step 1: Reading Data...")
    reader = make_reader(open(args.input_file, "rb"), decoder_factories=[DecoderFactory()])
    imu_data = []
    lidar_msgs = []

    for schema, channel, message, ros_msg in iter_decoded_messages_with_zstd(reader):
        if channel.topic == "/imu/data":
            q = [ros_msg.orientation.x, ros_msg.orientation.y, ros_msg.orientation.z, ros_msg.orientation.w]
            imu_data.append((message.log_time, q))
        elif channel.topic == "/scan":
            lidar_msgs.append((message.log_time, ros_msg))

    print(f"   Loaded: {len(lidar_msgs)} Scans, {len(imu_data)} IMU readings.")

    pitch_angles = np.arange(args.min_pitch, args.max_pitch, args.step)
    scores = []
    
    print("Step 2: Testing Pitch Angles...")
    for i, pitch in enumerate(pitch_angles):
        print(f"   Testing pitch = {pitch:.2f} degrees... ({i+1}/{len(pitch_angles)})")
        point_cloud = generate_point_cloud(lidar_msgs, imu_data, pitch)
        score = calculate_sharpness_score(point_cloud)
        scores.append(score)
        print(f"      -> Score = {score}")

    if not scores:
        print("Error: Could not calculate any scores.")
        return

    best_pitch_index = np.argmax(scores)
    best_pitch = pitch_angles[best_pitch_index]
    best_score = scores[best_pitch_index]

    print(f"Step 3: Results")
    print(f"   Best Pitch Angle: {best_pitch:.2f} degrees")
    print(f"   Best Score: {best_score}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(pitch_angles, scores, marker='o')
    plt.title('Pitch Angle vs. Sharpness Score')
    plt.xlabel('Pitch Angle (degrees)')
    plt.ylabel('Sharpness Score (Number of Detected Lines)')
    plt.grid(True)
    plt.axvline(x=best_pitch, color='r', linestyle='--', label=f'Best Pitch: {best_pitch:.2f}')
    plt.legend()
    
    output_filename = "pitch_calibration_results.png"
    plt.savefig(output_filename)
    print(f"Saved results plot to {output_filename}")


if __name__ == "__main__":
    main()