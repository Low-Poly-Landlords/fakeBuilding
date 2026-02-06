import numpy as np
import open3d as o3d
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as R
import bisect
import sys

# --- CONFIGURATION ---
MCAP_PATH = "/mnt/c/Users/Daniel/PycharmProjects/fakeBuilding/scan_imu_20260203_145509_0.mcap"
OUTPUT_FILENAME = "/mnt/c/Users/Daniel/PycharmProjects/fakeBuilding/final_3d_room.ply"

# --- CALIBRATION (Derived from your testing) ---
# 1. Lidar Rotation: Roll 90 degrees (1.57 rad) to make the ring vertical
LIDAR_FIX = R.from_euler('x', 90, degrees=True).as_matrix()

# 2. IMU Rotation: Roll 90 degrees (1.57 rad) to make Green Arrow point UP
# We represent this as a rotation object to combine with raw data later
IMU_FIX = R.from_euler('x', 90, degrees=True)


def get_interpolated_imu(target_time, imu_data):
    """Finds the IMU rotation closest to the requested timestamp."""
    times = [x[0] for x in imu_data]
    idx = bisect.bisect_left(times, target_time)

    if idx == 0: return imu_data[0][1]
    if idx == len(times): return imu_data[-1][1]

    # Pick the closest one
    before = times[idx - 1]
    after = times[idx]
    if after - target_time < target_time - before:
        return imu_data[idx][1]
    else:
        return imu_data[idx - 1][1]


def laserscan_to_points(msg):
    """Convert 2D scan to 3D points (flat on z=0)."""
    angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
    count = min(len(angles), len(msg.ranges))
    ranges = np.array(msg.ranges[:count])
    angles = angles[:count]

    # Filter valid range
    mask = (ranges > msg.range_min) & (ranges < msg.range_max)
    r = ranges[mask]
    a = angles[mask]

    x = r * np.cos(a)
    y = r * np.sin(a)
    z = np.zeros_like(x)
    return np.column_stack((x, y, z))


def main():
    print(f"Reading {MCAP_PATH}...")
    reader = make_reader(open(MCAP_PATH, "rb"), decoder_factories=[DecoderFactory()])

    imu_timeline = []
    lidar_msgs = []

    # 1. READ ALL DATA
    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic == "/imu/data":
            # Store orientation as (x,y,z,w)
            q = [ros_msg.orientation.x, ros_msg.orientation.y, ros_msg.orientation.z, ros_msg.orientation.w]
            imu_timeline.append((message.log_time, q))
        elif channel.topic == "/scan":
            lidar_msgs.append((message.log_time, ros_msg))

    print(f"loaded {len(imu_timeline)} IMU packets and {len(lidar_msgs)} Lidar scans.")

    global_pcd = o3d.geometry.PointCloud()

    # 2. FUSE DATA
    for i, (log_time, scan_msg) in enumerate(lidar_msgs):
        if i % 100 == 0: print(f"Processing scan {i}/{len(lidar_msgs)}...")

        # A. Convert Scan to Points
        local_points = laserscan_to_points(scan_msg)
        if len(local_points) == 0: continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(local_points)

        # B. Apply Lidar Calibration (Stand the ring up)
        pcd.rotate(LIDAR_FIX, center=(0, 0, 0))

        # C. Get Raw IMU Orientation
        raw_quat = get_interpolated_imu(log_time, imu_timeline)
        raw_rot = R.from_quat(raw_quat)

        # D. Apply IMU Calibration (Fix the Green Arrow)
        # We multiply the RAW rotation by our FIX rotation
        final_rot = raw_rot * IMU_FIX

        # E. Rotate the whole scan to match the IMU
        pcd.rotate(final_rot.as_matrix(), center=(0, 0, 0))

        global_pcd += pcd

    # 3. SAVE
    print("Downsampling and saving...")
    global_pcd = global_pcd.voxel_down_sample(voxel_size=0.03)  # 3cm resolution
    o3d.io.write_point_cloud(OUTPUT_FILENAME, global_pcd)
    print(f"SUCCESS! Model saved to: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()