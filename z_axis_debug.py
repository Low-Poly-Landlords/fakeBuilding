import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mcap.reader import make_reader
from rosbags.typesys import get_typestore, Stores
from scipy.spatial.transform import Rotation as R

INPUT_FILENAME = "C:/Robot_Scan_Project/processed_data/processed_data_0.mcap"


def laser_scan_to_xyz(msg):
    # Same conversion as before
    angles = np.arange(msg.angle_min, msg.angle_min + (len(msg.ranges) * msg.angle_increment), msg.angle_increment)
    if len(angles) > len(msg.ranges): angles = angles[:len(msg.ranges)]
    ranges = np.array(msg.ranges)
    valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max) & (ranges > 0.1)

    x = ranges[valid_mask] * np.cos(angles[valid_mask])
    y = ranges[valid_mask] * np.sin(angles[valid_mask])
    z = np.zeros_like(x)  # This assumes the sensor is FLAT!

    return np.stack([x, y, z], axis=1)


def main():
    print(f"Diagnosing Z-Axis in: {INPUT_FILENAME}...")
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    pose_map = {}

    # 1. Check TF Heights
    z_heights = []
    print("Reading TFs...")
    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=["/tf"]):
            msg = typestore.deserialize_cdr(message.data, "tf2_msgs/msg/TFMessage")
            for t in msg.transforms:
                if t.child_frame_id == "base_link" or "odom" in t.header.frame_id:
                    trans = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
                    quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
                            t.transform.rotation.w]

                    z_heights.append(trans[2])

                    # Store for stitching
                    rot_mat = R.from_quat(quat).as_matrix()
                    pose_mat = np.identity(4)
                    pose_mat[:3, :3] = rot_mat
                    pose_mat[:3, 3] = trans
                    pose_map[message.log_time] = pose_mat

    if z_heights:
        print(f"\n--- TRAJECTORY STATS ---")
        print(f"Min Height (Z): {min(z_heights):.4f} m")
        print(f"Max Height (Z): {max(z_heights):.4f} m")
        print(f"Total Vertical Movement: {max(z_heights) - min(z_heights):.4f} m")

    # 2. Stitch Cloud
    print("\nStitching points...")
    all_points = []
    with open(INPUT_FILENAME, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=["/scan"]):
            if not pose_map: break
            closest_ts = min(pose_map.keys(), key=lambda k: abs(k - message.log_time))
            if abs(closest_ts - message.log_time) > 1e8: continue

            pose = pose_map[closest_ts]
            msg = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/LaserScan")
            xyz = laser_scan_to_xyz(msg)
            if len(xyz) == 0: continue

            # Apply Pose
            xyz_T = xyz.T
            rotated = np.dot(pose[:3, :3], xyz_T)
            translated = rotated + pose[:3, 3:4]
            all_points.append(translated.T)

    if not all_points:
        print("No points found.")
        return

    full_cloud = np.vstack(all_points)
    z_vals = full_cloud[:, 2]

    print(f"\n--- POINT CLOUD STATS ---")
    print(f"Total Points: {len(full_cloud)}")
    print(f"Min Point Z: {np.min(z_vals):.4f} m")
    print(f"Max Point Z: {np.max(z_vals):.4f} m")
    print(f"Cloud Thickness: {np.max(z_vals) - np.min(z_vals):.4f} m")

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_cloud)
    # Color by Z to highlight any height
    colors = plt.get_cmap("jet")((z_vals - np.min(z_vals)) / (np.max(z_vals) - np.min(z_vals) + 1e-6))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("\nOpening Visualizer... (Look at the axes in the bottom left)")
    o3d.visualization.draw_geometries([pcd], window_name="Z-Axis Debugger")


if __name__ == "__main__":
    main()