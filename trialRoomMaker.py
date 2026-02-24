import os
import numpy as np
import open3d as o3d
import cv2
import bisect
from scipy.spatial.transform import Rotation as R
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# --- CONFIGURATION ---
INPUT_FILE = "newestScan.mcap"
OUTPUT_OBJ = "trialRoom.obj"
RESOLUTION = 0.02

LIDAR_FIX = R.from_euler('xyz', [90, -20.0, 0.0], degrees=True).as_matrix()
IMU_FIX = R.from_euler('x', 90, degrees=True)


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


def main():
    print(f"Reading Lidar from {INPUT_FILE}...")
    reader = make_reader(open(INPUT_FILE, "rb"), decoder_factories=[DecoderFactory()])
    imu_data = []
    lidar_msgs = []

    for schema, channel, message, ros_msg in reader.iter_decoded_messages():
        if channel.topic == "/imu/data":
            q = [ros_msg.orientation.x, ros_msg.orientation.y, ros_msg.orientation.z, ros_msg.orientation.w]
            imu_data.append((message.log_time, q))
        elif channel.topic == "/scan":
            lidar_msgs.append((message.log_time, ros_msg))

    # --- 1. BUILD RAW GEOMETRY ---
    global_points = []
    for log_time, scan_msg in lidar_msgs:
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        count = min(len(angles), len(scan_msg.ranges))
        r = np.array(scan_msg.ranges[:count])
        a = angles[:count]

        valid = (r > 1.0) & (r < scan_msg.range_max)
        x = r[valid] * np.cos(a[valid])
        y = r[valid] * np.sin(a[valid])
        z = np.zeros_like(x)
        pts = np.column_stack((x, y, z))

        pts = pts @ LIDAR_FIX.T
        raw_quat = get_interpolated_pose(log_time, imu_data)
        rot_matrix = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()
        pts = pts @ rot_matrix.T
        global_points.append(pts)

    all_pts = np.vstack(global_points)

    # --- 2. THE 2D SLICE ---
    ROOM_MIN_Z = np.min(all_pts[:, 2])
    ROOM_MAX_Z = np.max(all_pts[:, 2])
    mid_z = (ROOM_MAX_Z + ROOM_MIN_Z) / 2.0

    slice_mask = (all_pts[:, 2] > (mid_z - 0.2)) & (all_pts[:, 2] < (mid_z + 0.2))
    slice_pts = all_pts[slice_mask]

    min_x, max_x = np.min(slice_pts[:, 0]) - 0.5, np.max(slice_pts[:, 0]) + 0.5
    min_y, max_y = np.min(slice_pts[:, 1]) - 0.5, np.max(slice_pts[:, 1]) + 0.5

    img_w, img_h = int((max_x - min_x) / RESOLUTION), int((max_y - min_y) / RESOLUTION)
    floor_plan = np.zeros((img_h, img_w), dtype=np.uint8)

    px = ((slice_pts[:, 0] - min_x) / RESOLUTION).astype(int)
    py = ((slice_pts[:, 1] - min_y) / RESOLUTION).astype(int)
    for x, y in zip(px, py):
        cv2.circle(floor_plan, (x, y), radius=2, color=255, thickness=-1)

    closed_plan = cv2.morphologyEx(floor_plan, cv2.MORPH_CLOSE, np.ones((40, 40), np.uint8))
    contours, _ = cv2.findContours(closed_plan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)

    # THE FIX 1: The Rubber Band (Prevents self-intersecting bow-ties)
    hull_contour = cv2.convexHull(main_contour)

    approx_polygon = cv2.approxPolyDP(hull_contour, 0.015 * cv2.arcLength(hull_contour, True), True)

    # --- 3. BUILD THE PAPER-THIN SHELL ---
    vertices = []
    triangles = []
    poly_pts = approx_polygon.reshape(-1, 2)
    num_pts = len(poly_pts)

    # Walls
    for i in range(num_pts):
        p1, p2 = poly_pts[i], poly_pts[(i + 1) % num_pts]
        x1, y1 = (p1[0] * RESOLUTION) + min_x, (p1[1] * RESOLUTION) + min_y
        x2, y2 = (p2[0] * RESOLUTION) + min_x, (p2[1] * RESOLUTION) + min_y

        v_idx = len(vertices)
        vertices.extend([[x1, y1, ROOM_MIN_Z], [x2, y2, ROOM_MIN_Z], [x2, y2, ROOM_MAX_Z], [x1, y1, ROOM_MAX_Z]])

        # THE FINAL FIX: Point the walls INWARD toward the cameras
        triangles.extend([[v_idx, v_idx + 2, v_idx + 1], [v_idx, v_idx + 3, v_idx + 2]])

    # Floor Cap (Normal points UP)
    floor_idx = len(vertices)
    vertices.extend([[min_x, min_y, ROOM_MIN_Z], [max_x, min_y, ROOM_MIN_Z], [max_x, max_y, ROOM_MIN_Z],
                     [min_x, max_y, ROOM_MIN_Z]])
    triangles.extend([[floor_idx, floor_idx + 1, floor_idx + 2], [floor_idx, floor_idx + 2, floor_idx + 3]])

    # Ceiling Cap (Normal points DOWN)
    ceil_idx = len(vertices)
    vertices.extend([[min_x, min_y, ROOM_MAX_Z], [max_x, min_y, ROOM_MAX_Z], [max_x, max_y, ROOM_MAX_Z],
                     [min_x, max_y, ROOM_MAX_Z]])
    triangles.extend([[ceil_idx, ceil_idx + 2, ceil_idx + 1], [ceil_idx, ceil_idx + 3, ceil_idx + 2]])

    # --- 4. EXPORT WITH COMPUTED NORMALS ---
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))

    # THE FIX 3: Force Open3D to bake the inward normals into the file
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    o3d.io.write_triangle_mesh(OUTPUT_OBJ, mesh)
    print(f"Saved clean, hollow shell to {OUTPUT_OBJ}")


if __name__ == "__main__":
    main()