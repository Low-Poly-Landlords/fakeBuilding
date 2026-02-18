import os
import sys
import numpy as np
import open3d as o3d
import cv2
import bisect
from scipy.spatial.transform import Rotation as R
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# --- CONFIGURATION ---
INPUT_FILE = "newestScan.mcap"
OUTPUT_OBJ = "newestScan_camera_fixed.obj"

# 1. GHOST REMOVAL
MIN_LIDAR_DIST = 1.0

# 2. LIDAR CALIBRATION (Your Tuned Values)
LIDAR_ROLL_OFFSET = 0.0
LIDAR_PITCH_OFFSET = -20.0
LIDAR_YAW_OFFSET = 0.0

# 3. CAMERA CALIBRATION (THE FIX)
CAM_OFFSET = [0.0, 0.0, 0.05]
CAM_FOV_DEG = 70.0

# If windows are on the ceiling, try Pitch = -90.0
# If windows are on the floor, try Pitch = 90.0
# If windows are on the left wall, try Yaw = 90.0
CAM_ROLL = 0.0
CAM_PITCH = -180.0  # <--- TRY THIS FIRST
CAM_YAW = 0.0

# 4. MESH SETTINGS
VOXEL_SIZE = 0.03
POISSON_DEPTH = 9
TRIM_AMOUNT = 0.01

# --- TRANSFORMS ---
LIDAR_FIX = R.from_euler('xyz', [
    90 + LIDAR_ROLL_OFFSET,
    0 + LIDAR_PITCH_OFFSET,
    0 + LIDAR_YAW_OFFSET
], degrees=True).as_matrix()

IMU_FIX = R.from_euler('x', 90, degrees=True)

# We construct the camera rotation matrix dynamically now
# Base: Robot Frame -> Camera Optical Frame (Standard Swap)
BASE_ROBOT_TO_CAM = np.array([
    [0, -1, 0],
    [0, 0, -1],
    [1, 0, 0]
])

# User Correction: Applied on top of the base
USER_CAM_FIX = R.from_euler('xyz', [CAM_ROLL, CAM_PITCH, CAM_YAW], degrees=True).as_matrix()
# Combined Transform
FINAL_ROBOT_TO_CAM = USER_CAM_FIX @ BASE_ROBOT_TO_CAM


def get_interpolated_pose(target_time, pose_data):
    times = [x[0] for x in pose_data]
    idx = bisect.bisect_left(times, target_time)
    if idx == 0: return pose_data[0][1]
    if idx == len(times): return pose_data[-1][1]
    before = times[idx - 1]
    after = times[idx]
    if after - target_time < target_time - before:
        return pose_data[idx][1]
    return pose_data[idx - 1][1]


def project_points_with_normals(points, normals, image, intrinsic_matrix):
    h, w, _ = image.shape
    z = points[:, 2]

    valid_mask = z > 0.1
    z_safe = z.copy()
    z_safe[~valid_mask] = 1.0

    u = (points[:, 0] * intrinsic_matrix[0, 0] / z_safe) + intrinsic_matrix[0, 2]
    v = (points[:, 1] * intrinsic_matrix[1, 1] / z_safe) + intrinsic_matrix[1, 2]

    in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # Normal Check (Backface Culling)
    # Normals facing camera have Z < -0.2 in Optical Frame
    nz = normals[:, 2]
    facing_camera = nz < -0.2

    final_mask = valid_mask & in_frame & facing_camera

    colors = np.zeros((len(points), 3), dtype=np.float64)
    u_valid = u[final_mask].astype(int)
    v_valid = v[final_mask].astype(int)

    img_colors = image[v_valid, u_valid, ::-1] / 255.0
    colors[final_mask] = img_colors
    return colors, final_mask


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print("Step 1: Reading Data...")
    reader = make_reader(open(INPUT_FILE, "rb"), decoder_factories=[DecoderFactory()])
    imu_data = []
    lidar_msgs = []
    images = []

    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic == "/imu/data":
            q = [ros_msg.orientation.x, ros_msg.orientation.y, ros_msg.orientation.z, ros_msg.orientation.w]
            imu_data.append((message.log_time, q))
        elif channel.topic == "/scan":
            lidar_msgs.append((message.log_time, ros_msg))
        elif channel.topic == "/camera/image_raw":
            try:
                width = getattr(ros_msg, "width", 640)
                height = getattr(ros_msg, "height", 480)
                np_arr = np.frombuffer(ros_msg.data, dtype=np.uint8)
                img = np_arr.reshape((height, width, 3))
                images.append((message.log_time, img))
            except:
                pass

    print(f"   Loaded: {len(lidar_msgs)} Scans, {len(images)} Images.")

    # --- BUILD GEOMETRY ---
    print("Step 2: Building Geometry...")
    global_points = []

    for i, (log_time, scan_msg) in enumerate(lidar_msgs):
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        count = min(len(angles), len(scan_msg.ranges))
        r = np.array(scan_msg.ranges[:count])
        a = angles[:count]

        valid = (r > MIN_LIDAR_DIST) & (r < scan_msg.range_max)

        x = r[valid] * np.cos(a[valid])
        y = r[valid] * np.sin(a[valid])
        z = np.zeros_like(x)
        pts = np.column_stack((x, y, z))

        pts = pts @ LIDAR_FIX.T
        raw_quat = get_interpolated_pose(log_time, imu_data)
        rot_matrix = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()
        pts = pts @ rot_matrix.T

        global_points.append(pts)

    all_points = np.vstack(global_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    colors = np.ones_like(all_points) * 0.7
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("   Estimating Normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # --- PAINTING ---
    print(f"Step 3: Painting (Pitch Adjustment: {CAM_PITCH} deg)...")
    if len(images) > 0:
        h, w, _ = images[0][1].shape
        focal_length = (w / 2) / np.tan(np.deg2rad(CAM_FOV_DEG) / 2)
        K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])

        paint_interval = max(1, len(images) // 40)

        points_np = np.asarray(pcd.points)
        normals_np = np.asarray(pcd.normals)

        for i in range(0, len(images), paint_interval):
            t, img = images[i]

            raw_quat = get_interpolated_pose(t, imu_data)
            robot_rot = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()

            # Transform Points
            pts_robot = points_np @ robot_rot.T
            pts_robot = pts_robot - CAM_OFFSET
            pts_optical = pts_robot @ FINAL_ROBOT_TO_CAM.T

            # Transform Normals
            norms_robot = normals_np @ robot_rot.T
            norms_optical = norms_robot @ FINAL_ROBOT_TO_CAM.T

            new_colors, mask = project_points_with_normals(pts_optical, norms_optical, img, K)

            current_colors = np.asarray(pcd.colors)
            current_colors[mask] = new_colors[mask]

    # --- MESHING ---
    print("Step 4: Meshing...")
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH, linear_fit=False
    )

    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, TRIM_AMOUNT)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print(f"Saving to {OUTPUT_OBJ}...")
    o3d.io.write_triangle_mesh(OUTPUT_OBJ, mesh)
    o3d.visualization.draw_geometries([mesh], window_name="Final Corrected Scan")


if __name__ == "__main__":
    main()