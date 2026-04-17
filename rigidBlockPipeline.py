import os
import sys
import numpy as np
import cv2
import bisect
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.stats import binned_statistic_2d
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN
import ezdxf
import argparse
from pathlib import Path
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from mcap_zstd_helper import iter_decoded_messages_with_zstd
import plotly.graph_objects as go

# --- CONFIGURATION ---
MIN_LIDAR_DIST = 1.0
MAX_JUMP_DIST = 0.2

# LiDAR Calibration
LIDAR_ROLL_OFFSET = 0.0
LIDAR_PITCH_OFFSET = 0.0
LIDAR_YAW_OFFSET = 0.0

# 2D Floor Plan Settings
RESOLUTION = 0.02  # 2cm per pixel for the 2D floor plan grid
ORTHOGONAL_TOLERANCE = 12.0  # Degrees to snap walls to 90/180/270

# Vertical Filter Settings
GRID_RESOLUTION = 0.15  # 15cm grid squares to analyze height
MIN_WALL_HEIGHT = 0.40  # Objects must be at least 40cm tall to be considered a wall

# Mesh Texturing Settings
WALL_SUBDIVISION_RES = 0.05  # 5cm grid resolution for walls to catch texture detail
MAX_TEXTURE_FRAMES = 100  # Maximum number of camera frames to project to save RAM

# Camera Calibration (From Poisson Script)
CAM_OFFSET = [0.0, 0.0, 0.05]
CAM_FOV_DEG = 70.0
CAM_ROLL = -17
CAM_PITCH = 0
CAM_YAW = 0.0

# Options: cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, or None
IMAGE_ROTATION = cv2.ROTATE_180

# Permanent Calibration Modifier
CALIBRATION_SCALE_FACTOR = 1

# --- TRANSFORMS ---
LIDAR_FIX = R.from_euler('xyz', [
    90 + LIDAR_ROLL_OFFSET,
    0 + LIDAR_PITCH_OFFSET,
    0 + LIDAR_YAW_OFFSET
], degrees=True).as_matrix()

IMU_FIX = R.from_euler('x', 90, degrees=True)

BASE_ROBOT_TO_CAM = np.array([
    [0, -1, 0],
    [0, 0, -1],
    [1, 0, 0]
])

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


def align_and_snap_polygon(polygon, angle_tolerance=15.0):
    points = polygon.reshape(-1, 2).astype(np.float32)
    if len(points) < 3:
        return polygon

    max_len = 0
    dominant_angle = 0
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        length = np.linalg.norm(p2 - p1)
        if length > max_len:
            max_len = length
            dominant_angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    c, s = np.cos(-dominant_angle), np.sin(-dominant_angle)
    R_mat = np.array(((c, -s), (s, c)))
    rotated_points = np.dot(points, R_mat.T)

    snapped_rotated = []
    for i in range(len(rotated_points)):
        p1 = rotated_points[i]
        p2 = rotated_points[(i + 1) % len(rotated_points)]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))

        if angle < angle_tolerance or angle > (180 - angle_tolerance):
            p2[1] = p1[1]
        elif abs(angle - 90) < angle_tolerance:
            p2[0] = p1[0]

        snapped_rotated.append(p1)

    snapped_rotated = np.array(snapped_rotated)
    c_back, s_back = np.cos(dominant_angle), np.sin(dominant_angle)
    R_back = np.array(((c_back, -s_back), (s_back, c_back)))
    final_points = np.dot(snapped_rotated, R_back.T)

    return final_points.reshape(-1, 1, 2).astype(int)


def subdivide_wall(p1, p2, z_min, z_max, res=0.05):
    """Subdivides a wall into a grid of triangles for vertex coloring."""
    length = np.linalg.norm(np.array(p2) - np.array(p1))
    height = z_max - z_min
    cols = max(2, int(length / res) + 1)
    rows = max(2, int(height / res) + 1)

    vs = []
    ts = []
    for r in range(rows):
        z = z_min + (r / (rows - 1)) * height
        for c in range(cols):
            f = c / (cols - 1)
            x = p1[0] + f * (p2[0] - p1[0])
            y = p1[1] + f * (p2[1] - p1[1])
            vs.append([x, y, z])

    for r in range(rows - 1):
        for c in range(cols - 1):
            idx = r * cols + c
            ts.append([idx, idx + 1, idx + cols])
            ts.append([idx + 1, idx + cols + 1, idx + cols])

    return np.array(vs), np.array(ts)


def project_points_with_normals(points, normals, image, intrinsic_matrix):
    h, w, _ = image.shape
    z = points[:, 2]

    valid_mask = z > 0.1
    z_safe = z.copy()
    z_safe[~valid_mask] = 1.0

    u = (points[:, 0] * intrinsic_matrix[0, 0] / z_safe) + intrinsic_matrix[0, 2]
    v = (points[:, 1] * intrinsic_matrix[1, 1] / z_safe) + intrinsic_matrix[1, 2]

    in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    nz = normals[:, 2]
    facing_camera = nz < -0.1

    final_mask = valid_mask & in_frame & facing_camera

    colors = np.zeros((len(points), 3), dtype=np.float64)
    u_valid = u[final_mask].astype(int)
    v_valid = v[final_mask].astype(int)

    # Image is already RGB from the reading step
    colors[final_mask] = image[v_valid, u_valid] / 255.0
    return colors, final_mask


def export_to_plotly(mesh, output_html_path):
    print("   Preparing Plotly visualization...")
    target_triangles = 100000
    if len(mesh.triangles) > target_triangles:
        print("   Decimating mesh for web browser performance...")
        mesh = mesh.simplify_quadric_decimation(target_triangles)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        color_strings = [f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})' for r, g, b in colors]
    else:
        color_strings = None

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
            vertexcolor=color_strings, opacity=1.0, name="Textured Room"
        )
    ])

    fig.update_layout(scene=dict(aspectmode='data'), title="Interactive 3D Textured Room")
    fig.write_html(str(output_html_path))
    print(f"   Saved interactive web viewer to: {output_html_path}")


def process_mcap(input_path, show_viewer=True, make_html=False):
    print(f"\n{'=' * 50}\nGenerating DXF and OBJ: {input_path.name}\n{'=' * 50}")

    output_stem = input_path.stem
    dxf_filename = input_path.with_name(f"{output_stem}_floorplan.dxf")
    obj_filename = input_path.with_name(f"{output_stem}_extruded.obj")
    html_filename = input_path.with_name(f"{output_stem}_viewer.html")

    print("Step 1: Reading LiDAR, IMU, and Camera Data...")
    reader = make_reader(open(input_path, "rb"), decoder_factories=[DecoderFactory()])
    imu_data = []
    lidar_msgs = []
    camera_imgs = []

    for schema, channel, message, ros_msg in iter_decoded_messages_with_zstd(reader):
        if channel.topic == "/imu/data":
            q = [ros_msg.orientation.x, ros_msg.orientation.y, ros_msg.orientation.z, ros_msg.orientation.w]
            imu_data.append((message.log_time, q))
        elif channel.topic == "/scan":
            lidar_msgs.append((message.log_time, ros_msg))
        elif channel.topic == "/camera/image_raw":
            if hasattr(ros_msg, 'data'):
                encoding = ros_msg.encoding.lower()
                if encoding in ['rgb8', 'bgr8']:
                    channels = 3
                elif encoding in ['xrgb8888', 'rgba8', 'bgra8']:
                    channels = 4
                else:
                    continue

                img_data = np.frombuffer(ros_msg.data, dtype=np.uint8)
                try:
                    img = img_data.reshape((ros_msg.height, ros_msg.width, channels))
                except ValueError:
                    continue

                if channels == 4:
                    img = img[:, :, :3]
                if 'bgr' in encoding or 'xrgb' in encoding:
                    img = img[:, :, ::-1]

                if IMAGE_ROTATION is not None:
                    img = cv2.rotate(img, IMAGE_ROTATION)

                camera_imgs.append((message.log_time, img))

    print(f"   Loaded: {len(lidar_msgs)} Scans, {len(camera_imgs)} Images.")

    # --- STEP 2: BUILD RAW POINT CLOUD ---
    print("Step 2: Correcting Point Cloud Geometry...")
    global_points = []

    for i, (log_time, scan_msg) in enumerate(lidar_msgs):
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        count = min(len(angles), len(scan_msg.ranges))
        r = np.array(scan_msg.ranges[:count])
        a = angles[:count]

        r_safe = np.nan_to_num(r, posinf=0.0, neginf=0.0, nan=0.0)

        valid = (r_safe > MIN_LIDAR_DIST) & (r_safe < scan_msg.range_max)
        r_diff = np.abs(np.diff(r_safe, prepend=r_safe[0]))
        valid = valid & (r_diff < MAX_JUMP_DIST)

        r_clean = r_safe[valid]
        a_clean = a[valid]

        x = r_clean * np.cos(a_clean)
        y = r_clean * np.sin(a_clean)
        z = np.zeros_like(x)
        pts = np.column_stack((x, y, z))

        pts = pts @ LIDAR_FIX.T
        raw_quat = get_interpolated_pose(log_time, imu_data)
        rot_matrix = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()
        pts = pts @ rot_matrix.T

        global_points.append(pts)

    if not global_points:
        print(f"Error: No lidar points processed for {input_path.name}.")
        return

    all_pts = np.vstack(global_points)

    z_vals = all_pts[:, 2]
    hist, bin_edges = np.histogram(z_vals, bins=100)
    floor_peak_idx = np.argmax(hist[:30])

    ROOM_MIN_Z = bin_edges[floor_peak_idx]
    ROOM_MAX_Z = np.max(z_vals)

    # --- STEP 3: VERTICAL COLUMN FILTERING & ISLAND DELETION ---
    print("Step 3: Filtering flat noise and extracting tall walls...")

    min_z = ROOM_MIN_Z + 0.3
    max_z = ROOM_MIN_Z + 2.0

    slice_mask = (all_pts[:, 2] > min_z) & (all_pts[:, 2] < max_z)
    slice_pts = all_pts[slice_mask]

    min_x, max_x = np.min(slice_pts[:, 0]), np.max(slice_pts[:, 0])
    min_y, max_y = np.min(slice_pts[:, 1]), np.max(slice_pts[:, 1])

    bins_x = int((max_x - min_x) / GRID_RESOLUTION)
    bins_y = int((max_y - min_y) / GRID_RESOLUTION)

    z_range_grid, _, _, _ = binned_statistic_2d(
        slice_pts[:, 0], slice_pts[:, 1], slice_pts[:, 2],
        statistic=np.ptp, bins=[bins_x, bins_y]
    )
    z_range_grid = np.nan_to_num(z_range_grid, nan=0.0)

    idx_x = np.clip(((slice_pts[:, 0] - min_x) / GRID_RESOLUTION).astype(int), 0, bins_x - 1)
    idx_y = np.clip(((slice_pts[:, 1] - min_y) / GRID_RESOLUTION).astype(int), 0, bins_y - 1)

    valid_cells = z_range_grid >= MIN_WALL_HEIGHT
    tall_point_mask = valid_cells[idx_x, idx_y]
    tall_pts = slice_pts[tall_point_mask]

    if len(tall_pts) == 0:
        print("Error: No tall walls found. Adjust MIN_WALL_HEIGHT.")
        return

    clustering = DBSCAN(eps=0.8, min_samples=15).fit(tall_pts[:, :2])
    labels = clustering.labels_

    if len(labels) > 0 and np.any(labels != -1):
        valid_mask = labels != -1
        final_wall_pts = tall_pts[valid_mask]
    else:
        final_wall_pts = tall_pts

    # --- STEP 4: 2D MORPHOLOGY & TRACING ---
    print("Step 4: Bridging gaps and tracing perimeter...")

    pad = 2.0
    min_x, max_x = min_x - pad, max_x + pad
    min_y, max_y = min_y - pad, max_y + pad

    img_w = int((max_x - min_x) / RESOLUTION)
    img_h = int((max_y - min_y) / RESOLUTION)

    floor_plan = np.zeros((img_h, img_w), dtype=np.uint8)

    px = ((final_wall_pts[:, 0] - min_x) / RESOLUTION).astype(int)
    py = ((final_wall_pts[:, 1] - min_y) / RESOLUTION).astype(int)

    for x, y in zip(px, py):
        cv2.circle(floor_plan, (x, y), radius=3, color=255, thickness=-1)

    k_size = 31
    closed_plan = np.zeros_like(floor_plan)

    for angle in range(0, 180, 15):
        base_kernel = np.zeros((k_size, k_size), dtype=np.uint8)
        cv2.line(base_kernel, (0, k_size // 2), (k_size - 1, k_size // 2), 1, 1)

        M = cv2.getRotationMatrix2D((k_size // 2, k_size // 2), angle, 1.0)
        rotated_kernel = cv2.warpAffine(base_kernel, M, (k_size, k_size))
        rotated_kernel = (rotated_kernel > 0).astype(np.uint8)

        directional_close = cv2.morphologyEx(floor_plan, cv2.MORPH_CLOSE, rotated_kernel, iterations=2)
        closed_plan = cv2.bitwise_or(closed_plan, directional_close)

    stitch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_plan = cv2.morphologyEx(closed_plan, cv2.MORPH_CLOSE, stitch_kernel)

    contours, _ = cv2.findContours(closed_plan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Error: Could not trace walls.")
        return

    main_contour = max(contours, key=lambda c: cv2.arcLength(c, closed=True))
    epsilon = 0.02 * cv2.arcLength(main_contour, True)
    approx_polygon = cv2.approxPolyDP(main_contour, epsilon, True)
    approx_polygon = align_and_snap_polygon(approx_polygon, angle_tolerance=ORTHOGONAL_TOLERANCE)

    poly_pts_pixels = approx_polygon.reshape(-1, 2)
    poly_pts_meters = []
    for p in poly_pts_pixels:
        xm = (p[0] * RESOLUTION) + min_x
        ym = (p[1] * RESOLUTION) + min_y
        poly_pts_meters.append([xm, ym])

    poly_pts_meters = np.array(poly_pts_meters)

    if CALIBRATION_SCALE_FACTOR != 1.0:
        centroid = np.mean(poly_pts_meters, axis=0)
        poly_pts_meters = centroid + (poly_pts_meters - centroid) * CALIBRATION_SCALE_FACTOR

    # --- STEP 5: 3D EXTRUSION & PROJECTION ---
    print("Step 5: Extruding Subdivided Mesh...")
    vertices = []
    triangles = []
    vertex_colors = []
    num_pts = len(poly_pts_meters)

    COLOR_WALL = [0.6, 0.7, 0.8]
    COLOR_FLOOR = [0.4, 0.4, 0.4]
    COLOR_CEIL = [0.9, 0.9, 0.9]

    # 1. Extrude Subdivided Vertical Walls
    for i in range(num_pts):
        p1 = poly_pts_meters[i]
        p2 = poly_pts_meters[(i + 1) % num_pts]

        vs, ts = subdivide_wall(p1, p2, ROOM_MIN_Z, ROOM_MAX_Z, WALL_SUBDIVISION_RES)
        v_idx = len(vertices)

        vertices.extend(vs.tolist())
        vertex_colors.extend([COLOR_WALL] * len(vs))
        triangles.extend((ts + v_idx).tolist())

    # 2. Triangulate Floor and Ceiling Caps
    tri = Delaunay(poly_pts_pixels)
    valid_simplices = []

    for simplex in tri.simplices:
        pts = poly_pts_pixels[simplex]
        centroid = np.mean(pts, axis=0)
        if cv2.pointPolygonTest(approx_polygon, (float(centroid[0]), float(centroid[1])), False) >= 0:
            valid_simplices.append(simplex)

    floor_start_idx = len(vertices)
    for p in poly_pts_meters:
        vertices.append([p[0], p[1], ROOM_MIN_Z])
        vertex_colors.append(COLOR_FLOOR)
    for simplex in valid_simplices:
        triangles.append([floor_start_idx + simplex[0], floor_start_idx + simplex[1], floor_start_idx + simplex[2]])

    ceil_start_idx = len(vertices)
    for p in poly_pts_meters:
        vertices.append([p[0], p[1], ROOM_MAX_Z])
        vertex_colors.append(COLOR_CEIL)
    for simplex in valid_simplices:
        triangles.append([ceil_start_idx + simplex[0], ceil_start_idx + simplex[2], ceil_start_idx + simplex[1]])

    # Build the mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Compute Normals and orient them towards the center of the room to prevent back-face painting
    # Compute base normals
    mesh.compute_vertex_normals()

    points_np = np.asarray(mesh.vertices)
    normals_np = np.asarray(mesh.vertex_normals)
    colors_np = np.array(vertex_colors)

    # Manually orient normals towards the center of the room to prevent back-face painting
    room_center = np.array(
        [np.mean(poly_pts_meters[:, 0]), np.mean(poly_pts_meters[:, 1]), (ROOM_MAX_Z + ROOM_MIN_Z) / 2.0])

    # Get vectors pointing from each vertex to the room center
    directions_to_center = room_center - points_np

    # Dot product: If negative, the normal is facing away from the center
    dot_products = np.sum(normals_np * directions_to_center, axis=1)

    # Flip the normals that are facing outward
    flip_mask = dot_products < 0
    normals_np[flip_mask] *= -1

    # Apply the fixed normals back to the mesh
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals_np)

    # 3. Project Camera Textures using Optical Frame Logic
    if camera_imgs:
        print(f"   Projecting Camera Textures onto Walls...")
        step = max(1, len(camera_imgs) // MAX_TEXTURE_FRAMES)
        sample_imgs = camera_imgs[::step]

        color_sums = np.zeros_like(colors_np, dtype=np.float64)
        color_counts = np.zeros(len(points_np), dtype=np.float64)

        h, w, _ = sample_imgs[0][1].shape
        focal_length = (w / 2) / np.tan(np.deg2rad(CAM_FOV_DEG) / 2)
        K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])

        for log_time, img in sample_imgs:
            raw_quat = get_interpolated_pose(log_time, imu_data)
            robot_rot = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()

            pts_robot = points_np @ robot_rot.T
            pts_robot = pts_robot - CAM_OFFSET
            pts_optical = pts_robot @ FINAL_ROBOT_TO_CAM.T

            norms_robot = normals_np @ robot_rot.T
            norms_optical = norms_robot @ FINAL_ROBOT_TO_CAM.T

            new_colors, mask = project_points_with_normals(pts_optical, norms_optical, img, K)

            color_sums[mask] += new_colors[mask]
            color_counts[mask] += 1.0

        # Apply Averaged Colors
        painted_mask = color_counts > 0
        colors_np[painted_mask] = color_sums[painted_mask] / color_counts[painted_mask][:, np.newaxis]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)

    o3d.io.write_triangle_mesh(str(obj_filename), mesh)
    print(f"   Success! Saved textured 3D mesh to {obj_filename}")

    # --- STEP 6: DXF EXPORT ---
    print("Step 6: Writing DXF File...")
    doc = ezdxf.new('R2010')
    doc.header['$INSUNITS'] = 6
    msp = doc.modelspace()

    for i in range(num_pts):
        p1 = poly_pts_meters[i]
        p2 = poly_pts_meters[(i + 1) % num_pts]
        msp.add_line((p1[0], p1[1]), (p2[0], p2[1]))

    doc.saveas(dxf_filename)
    print(f"   Success! Saved 2D floor plan to {dxf_filename}")

    # --- STEP 7: PREVIEW ---
    if make_html:
        export_to_plotly(mesh, html_filename)

    if show_viewer:
        print("   Opening Floor Plan Preview...")
        plt.figure(figsize=(8, 8))

        plot_x = np.append(poly_pts_meters[:, 0], poly_pts_meters[0, 0])
        plot_y = np.append(poly_pts_meters[:, 1], poly_pts_meters[0, 1])

        plt.plot(plot_x, plot_y, 'b-', linewidth=2, label="Detected Walls (Calibrated)")
        plt.scatter(poly_pts_meters[:, 0], poly_pts_meters[:, 1], c='red', s=40, zorder=5, label="Corners")
        plt.scatter(final_wall_pts[:, 0], final_wall_pts[:, 1], c='gray', s=1, alpha=0.3, zorder=1,
                    label="Filtered LiDAR Walls")

        plt.axis('equal')
        plt.title(f"DXF Preview - {input_path.name}")
        plt.xlabel("Meters (X)")
        plt.ylabel("Meters (Y)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate rigid block walls and camera-textured 3D .obj meshes from .mcap LiDAR scans.")
    parser.add_argument("input_paths", nargs="+", help="Path(s) to the input .mcap file(s) or directory(ies).")
    parser.add_argument("-s", "--show-scan", action="store_true", help="Pop up a blueprint preview of the floor plan.")
    parser.add_argument("-w", "--web-viewer", action="store_true",
                        help="Generate an interactive HTML Plotly file for web/remote viewing.")
    args = parser.parse_args()

    mcap_files = []
    for path_str in args.input_paths:
        p = Path(path_str)
        if p.is_dir():
            mcap_files.extend(list(p.rglob("*.mcap")))
        elif p.is_file() and p.suffix.lower() == '.mcap':
            mcap_files.append(p)

    mcap_files = list(dict.fromkeys(mcap_files))

    if not mcap_files:
        print("Error: No valid .mcap files found.")
        sys.exit(1)

    for f in mcap_files:
        try:
            process_mcap(f, show_viewer=args.show_scan, make_html=args.web_viewer)
        except Exception as e:
            print(f"An error occurred while processing {f.name}: {e}")


if __name__ == "__main__":
    main()