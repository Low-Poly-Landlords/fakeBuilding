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
INPUT_FILE = "new_scan_1.mcap"
OUTPUT_OBJ = "final_room_fixed.obj"

# Camera Settings
CAM_OFFSET = [0.0, 0.0, 0.05]
CAM_FOV_DEG = 70.0

# Meshing Settings (MATCHING YOUR PREFERRED CODE)
VOXEL_SIZE = 0.03
POISSON_DEPTH = 9
TRIM_AMOUNT = 0.01  # Keep 1% (Your setting)

# Calibration
LIDAR_FIX = R.from_euler('x', 90, degrees=True).as_matrix()
IMU_FIX = R.from_euler('x', 90, degrees=True)

# --- THE TEXTURE FIX (Only new part) ---
ROBOT_TO_CAM = np.array([
    [0, -1, 0],
    [0, 0, -1],
    [1, 0, 0]
])


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


def project_points_to_image(points, image, intrinsic_matrix):
    h, w, _ = image.shape
    # Points must be in Camera Optical Frame (Z=Forward)
    z = points[:, 2]
    valid_mask = z > 0.1
    z_safe = z.copy()
    z_safe[~valid_mask] = 1.0

    u = (points[:, 0] * intrinsic_matrix[0, 0] / z_safe) + intrinsic_matrix[0, 2]
    v = (points[:, 1] * intrinsic_matrix[1, 1] / z_safe) + intrinsic_matrix[1, 2]

    in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    final_mask = valid_mask & in_frame

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

    # --- STEP 1: LOAD ---
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

    # --- STEP 2: BUILD ---
    print("Step 2: Building Geometry...")
    global_points = []
    for i, (log_time, scan_msg) in enumerate(lidar_msgs):
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        count = min(len(angles), len(scan_msg.ranges))
        r = np.array(scan_msg.ranges[:count])
        a = angles[:count]
        valid = (r > scan_msg.range_min) & (r < scan_msg.range_max)
        x = r[valid] * np.cos(a[valid])
        y = r[valid] * np.sin(a[valid])
        z = np.zeros_like(x)
        pts = np.column_stack((x, y, z))

        # Apply Fixed Calibrations
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

    # --- STEP 3: PAINT WITH FIX ---
    print("Step 3: Painting with Axis Fix...")
    if len(images) > 0:
        h, w, _ = images[0][1].shape
        focal_length = (w / 2) / np.tan(np.deg2rad(CAM_FOV_DEG) / 2)
        K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])

        paint_interval = max(1, len(images) // 40)
        for i in range(0, len(images), paint_interval):
            t, img = images[i]

            raw_quat = get_interpolated_pose(t, imu_data)
            robot_rot = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()

            # Transform to Robot Frame
            pts_robot = all_points @ robot_rot.T

            # Apply Offset
            pts_robot = pts_robot - CAM_OFFSET

            # THE FIX: Swap axes so Z is forward for the camera
            pts_optical = pts_robot @ ROBOT_TO_CAM.T

            new_colors, mask = project_points_to_image(pts_optical, img, K)
            current_colors = np.asarray(pcd.colors)
            current_colors[mask] = new_colors[mask]

    # --- STEP 4: MESH (RESTORED TO YOUR SETTINGS) ---
    print("Step 4: Meshing (Restored Settings)...")

    # Using your original cleaning settings
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    # Standard Poisson (No cropping!)
    print("   Running Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH, linear_fit=False
    )

    # Using your original trim amount (0.01)
    print("   Trimming excess geometry...")
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, TRIM_AMOUNT)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # NO CROP HERE - This keeps the shape exactly as you liked it.

    print(f"Saving to {OUTPUT_OBJ}...")
    o3d.io.write_triangle_mesh(OUTPUT_OBJ, mesh)
    o3d.visualization.draw_geometries([mesh], window_name="Final Restored Mesh")


if __name__ == "__main__":
    main()