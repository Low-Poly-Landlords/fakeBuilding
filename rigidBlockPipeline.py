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
OUTPUT_OBJ = "final_extruded_room.obj"

# 1. GHOST REMOVAL
MIN_LIDAR_DIST = 1.0

# 2. LIDAR CALIBRATION (Your Tuned Values)
LIDAR_ROLL_OFFSET = 0.0
LIDAR_PITCH_OFFSET = -20.0
LIDAR_YAW_OFFSET = 0.0

# 3. CAMERA CALIBRATION (THE FIX)
CAM_OFFSET = [0.0, 0.0, 0.05]
CAM_FOV_DEG = 70.0

CAM_ROLL = -17.0
CAM_PITCH = -1.0
CAM_YAW = 0.0

# 4. EXTRUSION SETTINGS
RESOLUTION = 0.02  # 2cm per pixel for the 2D floor plan
SUBDIVISION_LEVEL = 6  # Increase to 7 if floor textures look blurry

# --- TRANSFORMS ---
LIDAR_FIX = R.from_euler('xyz', [
    90 + LIDAR_ROLL_OFFSET,
    0 + LIDAR_PITCH_OFFSET,
    0 + LIDAR_YAW_OFFSET
], degrees=True).as_matrix()

IMU_FIX = R.from_euler('x', 90, degrees=True)

# Base: Robot Frame -> Camera Optical Frame
BASE_ROBOT_TO_CAM = np.array([
    [0, -1, 0],
    [0, 0, -1],
    [1, 0, 0]
])

# User Correction: Applied on top of the base
USER_CAM_FIX = R.from_euler('xyz', [CAM_ROLL, CAM_PITCH, CAM_YAW], degrees=True).as_matrix()
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


def project_points_with_smarter_normals(points, normals, image, intrinsic_matrix):
    h, w, _ = image.shape
    z = points[:, 2]

    valid_mask = z > 0.1
    z_safe = z.copy()
    z_safe[~valid_mask] = 1.0

    u = (points[:, 0] * intrinsic_matrix[0, 0] / z_safe) + intrinsic_matrix[0, 2]
    v = (points[:, 1] * intrinsic_matrix[1, 1] / z_safe) + intrinsic_matrix[1, 2]

    in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # Calculate Viewing Angle Score
    # In the optical frame, the camera looks down the +Z axis.
    # A wall perfectly facing the camera has a normal of [0, 0, -1].
    nz = normals[:, 2]
    view_scores = -nz

    # Strict cut-off: Only paint if the camera is facing the wall directly (Score > 0.3)
    facing_camera = view_scores > 0.3

    final_mask = valid_mask & in_frame & facing_camera

    colors = np.zeros((len(points), 3), dtype=np.float64)
    u_valid = u[final_mask].astype(int)
    v_valid = v[final_mask].astype(int)

    img_colors = image[v_valid, u_valid, ::-1] / 255.0
    colors[final_mask] = img_colors

    return colors, view_scores, final_mask


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

    # --- STEP 2: BUILD RAW GEOMETRY ---
    print("Step 2: Building Raw Geometry for Floor Plan...")
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

    all_pts = np.vstack(global_points)

    # --- STEP 3: 2D SLICE & EXTRUSION ---
    print("Step 3: Creating 2D Floor Plan and Extruding 3D Walls...")
    ROOM_MIN_Z = np.min(all_pts[:, 2])
    ROOM_MAX_Z = np.max(all_pts[:, 2])
    mid_z = (ROOM_MAX_Z + ROOM_MIN_Z) / 2.0
    slice_thickness = 0.2

    print(f"   Slicing room at Z={mid_z:.2f}m...")
    slice_mask = (all_pts[:, 2] > (mid_z - slice_thickness)) & (all_pts[:, 2] < (mid_z + slice_thickness))
    slice_pts = all_pts[slice_mask]

    min_x, max_x = np.min(slice_pts[:, 0]), np.max(slice_pts[:, 0])
    min_y, max_y = np.min(slice_pts[:, 1]), np.max(slice_pts[:, 1])

    pad = 0.5
    min_x, max_x = min_x - pad, max_x + pad
    min_y, max_y = min_y - pad, max_y + pad

    img_w = int((max_x - min_x) / RESOLUTION)
    img_h = int((max_y - min_y) / RESOLUTION)

    floor_plan = np.zeros((img_h, img_w), dtype=np.uint8)
    px = ((slice_pts[:, 0] - min_x) / RESOLUTION).astype(int)
    py = ((slice_pts[:, 1] - min_y) / RESOLUTION).astype(int)

    for x, y in zip(px, py):
        cv2.circle(floor_plan, (x, y), radius=2, color=255, thickness=-1)

    print("   Tracing structural perimeter...")

    # 1. Thicken the dots into solid lines
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(floor_plan, kernel, iterations=2)

    # 2. THE FIX: Seal the gaps (Virtual Caulk)
    # This huge kernel bridges gaps (like doorways) so the tracer can't leak inside
    close_kernel = np.ones((40, 40), np.uint8)
    closed_plan = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel)

    # 3. Find the single outermost boundary
    contours, _ = cv2.findContours(closed_plan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: Could not find walls in the 2D slice.")
        return

    main_contour = max(contours, key=cv2.contourArea)

    # 4. Optional: Force a purely convex shape (Rubber Band effect)
    # Uncomment this next line if your room is a simple rectangle and you want it perfectly solid.
    # Do not use this if your room is L-shaped.
    # main_contour = cv2.convexHull(main_contour)

    epsilon = 0.015 * cv2.arcLength(main_contour, True)
    approx_polygon = cv2.approxPolyDP(main_contour, epsilon, True)

    # Save 2D Floor Plan Image
    final_floor_plan = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    cv2.drawContours(final_floor_plan, [approx_polygon], -1, (0, 0, 0), 3)
    cv2.imwrite("floor_plan.jpg", final_floor_plan)
    print("   Saved 2D slice as 'floor_plan.jpg'")

    print("   Extruding walls to floor and ceiling...")
    vertices = []
    triangles = []
    vertex_colors = []
    poly_pts = approx_polygon.reshape(-1, 2)
    num_pts = len(poly_pts)

    base_color = [0.5, 0.5, 0.5]

    # 1. Extrude Walls
    for i in range(num_pts):
        p1 = poly_pts[i]
        p2 = poly_pts[(i + 1) % num_pts]

        x1 = (p1[0] * RESOLUTION) + min_x
        y1 = (p1[1] * RESOLUTION) + min_y
        x2 = (p2[0] * RESOLUTION) + min_x
        y2 = (p2[1] * RESOLUTION) + min_y

        v_idx = len(vertices)
        vertices.extend([
            [x1, y1, ROOM_MIN_Z],
            [x2, y2, ROOM_MIN_Z],
            [x2, y2, ROOM_MAX_Z],
            [x1, y1, ROOM_MAX_Z]
        ])
        triangles.extend([[v_idx, v_idx + 1, v_idx + 2], [v_idx, v_idx + 2, v_idx + 3]])
        vertex_colors.extend([base_color] * 4)

    # 2. Floor Cap (Counter-Clockwise to point UP)
    floor_idx = len(vertices)
    vertices.extend([
        [min_x, min_y, ROOM_MIN_Z],
        [max_x, min_y, ROOM_MIN_Z],
        [max_x, max_y, ROOM_MIN_Z],
        [min_x, max_y, ROOM_MIN_Z]
    ])
    triangles.extend([[floor_idx, floor_idx + 1, floor_idx + 2], [floor_idx, floor_idx + 2, floor_idx + 3]])
    vertex_colors.extend([[0.3, 0.3, 0.3]] * 4)

    # 3. Ceiling Cap (Clockwise to point DOWN)
    ceil_idx = len(vertices)
    vertices.extend([
        [min_x, min_y, ROOM_MAX_Z],
        [max_x, min_y, ROOM_MAX_Z],
        [max_x, max_y, ROOM_MAX_Z],
        [min_x, max_y, ROOM_MAX_Z]
    ])
    triangles.extend([[ceil_idx, ceil_idx + 2, ceil_idx + 1], [ceil_idx, ceil_idx + 3, ceil_idx + 2]])
    vertex_colors.extend([[0.9, 0.9, 0.9]] * 4)

    final_rigid_mesh = o3d.geometry.TriangleMesh()
    final_rigid_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    final_rigid_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    final_rigid_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # --- STEP 4: PREPARE MESH CANVAS ---
    print("Step 4: Subdividing Mesh for High-Res Painting...")
    # This slices the flat walls/floors into thousands of tiny triangles to hold pixel data
    final_rigid_mesh = final_rigid_mesh.subdivide_midpoint(number_of_iterations=SUBDIVISION_LEVEL)
    # After final_rigid_mesh = final_rigid_mesh.subdivide_midpoint(...)
    num_vertices = len(final_rigid_mesh.vertices)
    vertex_scores = np.zeros(num_vertices, dtype=np.float64) - 1.0  # Start with terrible scores
    final_rigid_mesh.compute_vertex_normals()

    # --- STEP 5: PAINT THE RIGID MESH ---
    print(f"Step 5: Painting Extruded Architecture (Pitch: {CAM_PITCH} deg)...")
    if len(images) > 0:
        h, w, _ = images[0][1].shape
        focal_length = (w / 2) / np.tan(np.deg2rad(CAM_FOV_DEG) / 2)
        K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])

        paint_interval = max(1, len(images) // 40)

        points_np = np.asarray(final_rigid_mesh.vertices)
        normals_np = np.asarray(final_rigid_mesh.vertex_normals)

        # Array to track the "best" viewing angle for every single vertex
        num_vertices = len(points_np)
        vertex_scores = np.zeros(num_vertices, dtype=np.float64) - 1.0

        for i in range(0, len(images), paint_interval):
            t, img = images[i]

            raw_quat = get_interpolated_pose(t, imu_data)
            robot_rot = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()

            # --- THE BUG FIX: Removed .T so we correctly reverse the rotation ---
            pts_robot = points_np @ robot_rot
            pts_robot = pts_robot - CAM_OFFSET
            pts_optical = pts_robot @ FINAL_ROBOT_TO_CAM.T

            norms_robot = normals_np @ robot_rot
            norms_optical = norms_robot @ FINAL_ROBOT_TO_CAM.T
            # --------------------------------------------------------------------

            # Project using the new smarter normals
            new_colors, new_scores, mask = project_points_with_smarter_normals(pts_optical, norms_optical, img, K)

            # --- THE TEXTURE UPGRADE: Only paint if it's a better angle ---
            current_colors = np.copy(np.asarray(final_rigid_mesh.vertex_colors))

            # Find vertices where this image is BOTH in-frame AND has a better view score
            better_view_mask = mask & (new_scores > vertex_scores)

            current_colors[better_view_mask] = new_colors[better_view_mask]
            vertex_scores[better_view_mask] = new_scores[better_view_mask]

            final_rigid_mesh.vertex_colors = o3d.utility.Vector3dVector(current_colors)

    # --- STEP 6: SAVE ---
    print(f"Saving to {OUTPUT_OBJ}...")
    o3d.io.write_triangle_mesh(OUTPUT_OBJ, final_rigid_mesh)

    print("Opening Viewer...")
    o3d.visualization.draw_geometries([final_rigid_mesh], window_name="Textured 2.5D Architecture")


if __name__ == "__main__":
    main()