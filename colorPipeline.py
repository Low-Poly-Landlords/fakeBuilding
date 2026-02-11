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
INPUT_FILE = "scan_imu_20260203_145509_0.mcap"
OUTPUT_OBJ = "final_room_model.obj"

# Camera Settings
CAM_OFFSET = [0.0, 0.0, 0.05]  # Camera 5cm above Lidar
CAM_FOV_DEG = 70.0  # Lens width

# Wall Generation Settings
VOXEL_SIZE = 0.03  # 3cm resolution for point cloud
RANSAC_DIST = 0.05  # 5cm tolerance for wall flatness
PARALLEL_THRESH = 0.9  # Wall merge angle tolerance
MERGE_DIST = 0.20  # Merge walls within 20cm
MIN_POINTS = 100  # Ignore tiny walls
WALL_ITERATIONS = 50  # Max number of planes to find

# Calibration (Fixed)
LIDAR_FIX = R.from_euler('x', 90, degrees=True).as_matrix()
IMU_FIX = R.from_euler('x', 90, degrees=True)


def get_interpolated_pose(target_time, pose_data):
    """Finds the IMU rotation closest to the requested timestamp."""
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
    """Projects 3D points [N,3] into the Image [H,W,3] to get colors."""
    h, w, _ = image.shape

    # 1. Project 3D -> 2D
    z = points[:, 2]
    valid_mask = z > 0.1
    z_safe = z.copy()
    z_safe[~valid_mask] = 1.0

    u = (points[:, 0] * intrinsic_matrix[0, 0] / z_safe) + intrinsic_matrix[0, 2]
    v = (points[:, 1] * intrinsic_matrix[1, 1] / z_safe) + intrinsic_matrix[1, 2]

    # 2. Check bounds
    in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    final_mask = valid_mask & in_frame

    colors = np.zeros((len(points), 3), dtype=np.float64)

    # 3. Sample Colors
    u_valid = u[final_mask].astype(int)
    v_valid = v[final_mask].astype(int)

    # OpenCV (BGR) -> Open3D (RGB)
    img_colors = image[v_valid, u_valid, ::-1] / 255.0
    colors[final_mask] = img_colors

    return colors, final_mask


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # --- STEP 1: LOAD DATA ---
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

    # --- STEP 2: BUILD & COLOR POINT CLOUD ---
    print("Step 2: Building & Coloring Geometry...")
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

        # Calibration
        pts = pts @ LIDAR_FIX.T
        raw_quat = get_interpolated_pose(log_time, imu_data)
        rot_matrix = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()
        pts = pts @ rot_matrix.T

        global_points.append(pts)

    all_points = np.vstack(global_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    # Init Default Color (Light Grey)
    colors = np.ones_like(all_points) * 0.7
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # --- PAINTING LOOP ---
    if len(images) > 0:
        h, w, _ = images[0][1].shape
        focal_length = (w / 2) / np.tan(np.deg2rad(CAM_FOV_DEG) / 2)
        K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])

        paint_interval = max(1, len(images) // 40)  # Use 40 frames total for speed

        for i in range(0, len(images), paint_interval):
            t, img = images[i]
            raw_quat = get_interpolated_pose(t, imu_data)
            robot_rot = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()

            pts_local = all_points @ robot_rot.T
            pts_cam = pts_local - CAM_OFFSET
            new_colors, mask = project_points_to_image(pts_cam, img, K)

            current_colors = np.asarray(pcd.colors)
            current_colors[mask] = new_colors[mask]  # Overwrite with photo color

    # --- STEP 3: SEGMENT WALLS & FLOORS ---
    print("Step 3: Segmenting Solid Walls...")
    # Downsample for faster RANSAC
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    detected_planes = []  # List of {'equation': [], 'points': [], 'colors': []}

    for i in range(WALL_ITERATIONS):
        if len(pcd.points) < MIN_POINTS: break

        # RANSAC Plane Finding
        plane_model, inliers = pcd.segment_plane(distance_threshold=RANSAC_DIST,
                                                 ransac_n=3, num_iterations=1000)

        # Extract Candidate Data
        candidate_cloud = pcd.select_by_index(inliers)
        candidate_points = np.asarray(candidate_cloud.points)
        candidate_colors = np.asarray(candidate_cloud.colors)
        candidate_center = np.mean(candidate_points, axis=0)
        normal_candidate = np.array(plane_model[:3])

        # Merge Logic
        merged = False
        for existing in detected_planes:
            normal_existing = np.array(existing['equation'][:3])

            # Check Parallelism & Distance
            if abs(np.dot(normal_candidate, normal_existing)) > PARALLEL_THRESH:
                dist = abs(np.dot(normal_existing, candidate_center) + existing['equation'][3])
                if dist < MERGE_DIST:
                    existing['points'].extend(candidate_points)
                    existing['colors'].extend(candidate_colors)  # Merge colors too!
                    merged = True
                    break

        if not merged:
            detected_planes.append({
                'equation': plane_model,
                'points': list(candidate_points),
                'colors': list(candidate_colors)
            })

        pcd = pcd.select_by_index(inliers, invert=True)

    # --- STEP 4: BUILD FINAL COLORED MESH ---
    print(f"Step 4: Constructing {len(detected_planes)} colored blocks...")
    final_mesh = o3d.geometry.TriangleMesh()

    for plane in detected_planes:
        wall_pcd = o3d.geometry.PointCloud()
        wall_pcd.points = o3d.utility.Vector3dVector(plane['points'])
        wall_pcd.colors = o3d.utility.Vector3dVector(plane['colors'])

        # 1. Calculate Average Color of this wall
        # We take the median to ignore random noise/outliers
        avg_color = np.median(plane['colors'], axis=0)

        # 2. Create Solid Box (OBB)
        obb = wall_pcd.get_oriented_bounding_box()
        mesh_box = o3d.geometry.TriangleMesh.create_box(
            width=obb.extent[0], height=obb.extent[1], depth=obb.extent[2])

        # 3. Position Box
        mesh_box.translate(-mesh_box.get_center())
        mesh_box.rotate(obb.R, center=[0, 0, 0])
        mesh_box.translate(obb.center)
        mesh_box.compute_vertex_normals()

        # 4. Paint the Box with the Wall's Color
        mesh_box.paint_uniform_color(avg_color)

        final_mesh += mesh_box

    print(f"Saving to {OUTPUT_OBJ}...")
    o3d.io.write_triangle_mesh(OUTPUT_OBJ, final_mesh)

    print("Opening Viewer...")
    o3d.visualization.draw_geometries([final_mesh], window_name="Final Colored Room")


if __name__ == "__main__":
    main()