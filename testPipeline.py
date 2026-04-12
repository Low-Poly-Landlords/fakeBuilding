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

# --- CONFIGURATION ---
MIN_LIDAR_DIST = 1.0
MAX_JUMP_DIST = 0.2

LIDAR_ROLL_OFFSET = 0.0
LIDAR_PITCH_OFFSET = 0.0
LIDAR_YAW_OFFSET = 0.0

# 2D Floor Plan Settings
RESOLUTION = 0.02  # 2cm per pixel for the 2D floor plan grid
ORTHOGONAL_TOLERANCE = 12.0  # Degrees to snap walls to 90/180/270

# Vertical Filter Settings
GRID_RESOLUTION = 0.15  # 15cm grid squares to analyze height
MIN_WALL_HEIGHT = 0.40  # Objects must be at least 40cm tall to be considered a wall

# --- TRANSFORMS ---
LIDAR_FIX = R.from_euler('xyz', [
    90 + LIDAR_ROLL_OFFSET,
    0 + LIDAR_PITCH_OFFSET,
    0 + LIDAR_YAW_OFFSET
], degrees=True).as_matrix()

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


def snap_polygon_to_orthogonal(polygon, angle_tolerance=15.0):
    """Forces polygon edges to be horizontal or vertical."""
    points = polygon.reshape(-1, 2).astype(np.float32)
    snapped_points = []

    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))

        if angle < angle_tolerance:
            p2[1] = p1[1]
        elif angle > (90 - angle_tolerance):
            p2[0] = p1[0]

        snapped_points.append(p1)

    return np.array(snapped_points).reshape(-1, 1, 2)


def process_mcap(input_path, show_viewer=True):
    print(f"\n{'=' * 50}\nGenerating DXF and OBJ: {input_path.name}\n{'=' * 50}")

    output_stem = input_path.stem
    dxf_filename = input_path.with_name(f"{output_stem}_floorplan.dxf")
    obj_filename = input_path.with_name(f"{output_stem}_extruded.obj")

    print("Step 1: Reading LiDAR and IMU Data...")
    reader = make_reader(open(input_path, "rb"), decoder_factories=[DecoderFactory()])
    imu_data = []
    lidar_msgs = []

    for schema, channel, message, ros_msg in iter_decoded_messages_with_zstd(reader):
        if channel.topic == "/imu/data":
            q = [ros_msg.orientation.x, ros_msg.orientation.y, ros_msg.orientation.z, ros_msg.orientation.w]
            imu_data.append((message.log_time, q))
        elif channel.topic == "/scan":
            lidar_msgs.append((message.log_time, ros_msg))

    print(f"   Loaded: {len(lidar_msgs)} Scans.")

    # --- STEP 2: BUILD RAW POINT CLOUD ---
    print("Step 2: Correcting Point Cloud Geometry...")
    global_points = []

    for i, (log_time, scan_msg) in enumerate(lidar_msgs):
        angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
        count = min(len(angles), len(scan_msg.ranges))
        r = np.array(scan_msg.ranges[:count])
        a = angles[:count]

        r_safe = np.nan_to_num(r, posinf=0.0, neginf=0.0, nan=0.0)

        # Basic Mask & Edge Jump Filter
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

    # Extract overall room heights for the 3D extrusion later
    z_vals = all_pts[:, 2]
    hist, bin_edges = np.histogram(z_vals, bins=100)
    floor_peak_idx = np.argmax(hist[:30])

    ROOM_MIN_Z = bin_edges[floor_peak_idx]  # True Floor
    ROOM_MAX_Z = np.max(z_vals)  # True Ceiling

    # --- STEP 3: VERTICAL COLUMN FILTERING & ISLAND DELETION ---
    print("Step 3: Filtering flat noise and extracting tall walls...")

    # 30cm to 2.0m vertical slice
    min_z = ROOM_MIN_Z + 0.3
    max_z = ROOM_MIN_Z + 2.0

    slice_mask = (all_pts[:, 2] > min_z) & (all_pts[:, 2] < max_z)
    slice_pts = all_pts[slice_mask]

    # Grid boundaries
    min_x, max_x = np.min(slice_pts[:, 0]), np.max(slice_pts[:, 0])
    min_y, max_y = np.min(slice_pts[:, 1]), np.max(slice_pts[:, 1])

    bins_x = int((max_x - min_x) / GRID_RESOLUTION)
    bins_y = int((max_y - min_y) / GRID_RESOLUTION)

    # Calculate vertical height (ptp) for every grid cell
    z_range_grid, _, _, _ = binned_statistic_2d(
        slice_pts[:, 0], slice_pts[:, 1], slice_pts[:, 2],
        statistic=np.ptp, bins=[bins_x, bins_y]
    )
    z_range_grid = np.nan_to_num(z_range_grid, nan=0.0)

    # Map each point to its corresponding grid cell to check if it belongs to a "tall" object
    idx_x = np.clip(((slice_pts[:, 0] - min_x) / GRID_RESOLUTION).astype(int), 0, bins_x - 1)
    idx_y = np.clip(((slice_pts[:, 1] - min_y) / GRID_RESOLUTION).astype(int), 0, bins_y - 1)

    # Keep only the points that land in a grid cell with a height >= MIN_WALL_HEIGHT
    valid_cells = z_range_grid >= MIN_WALL_HEIGHT
    tall_point_mask = valid_cells[idx_x, idx_y]
    tall_pts = slice_pts[tall_point_mask]

    if len(tall_pts) == 0:
        print("Error: No tall walls found. Adjust MIN_WALL_HEIGHT.")
        return

    # Island Deletion - Keep ALL walls, drop ONLY isolated noise
    clustering = DBSCAN(eps=0.8, min_samples=15).fit(tall_pts[:, :2])
    labels = clustering.labels_

    if len(labels) > 0 and np.any(labels != -1):
        valid_mask = labels != -1
        final_wall_pts = tall_pts[valid_mask]
    else:
        final_wall_pts = tall_pts  # Fallback

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

    # Draw FATTER points to bridge natural LiDAR sparsity (Radius = 5)
    for x, y in zip(px, py):
        cv2.circle(floor_plan, (x, y), radius=5, color=255, thickness=-1)

    # Closing kernel to jump across doorways and massive windows
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    closed_plan = cv2.morphologyEx(floor_plan, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed_plan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Error: Could not trace walls.")
        return

    # Sort by Perimeter length
    main_contour = max(contours, key=lambda c: cv2.arcLength(c, closed=True))

    # Simplify and snap
    epsilon = 0.015 * cv2.arcLength(main_contour, True)
    approx_polygon = cv2.approxPolyDP(main_contour, epsilon, True)
    approx_polygon = snap_polygon_to_orthogonal(approx_polygon, angle_tolerance=ORTHOGONAL_TOLERANCE)

    poly_pts_pixels = approx_polygon.reshape(-1, 2)
    poly_pts_meters = []
    for p in poly_pts_pixels:
        xm = (p[0] * RESOLUTION) + min_x
        ym = (p[1] * RESOLUTION) + min_y
        poly_pts_meters.append([xm, ym])
    poly_pts_meters = np.array(poly_pts_meters)

    # --- STEP 5: 3D EXTRUSION TO OBJ ---
    print("Step 5: Extruding 3D Mesh...")
    vertices = []
    triangles = []
    num_pts = len(poly_pts_meters)

    # 1. Extrude Vertical Walls
    for i in range(num_pts):
        p1 = poly_pts_meters[i]
        p2 = poly_pts_meters[(i + 1) % num_pts]

        v_idx = len(vertices)
        vertices.extend([
            [p1[0], p1[1], ROOM_MIN_Z],
            [p2[0], p2[1], ROOM_MIN_Z],
            [p2[0], p2[1], ROOM_MAX_Z],
            [p1[0], p1[1], ROOM_MAX_Z]
        ])
        triangles.append([v_idx, v_idx + 1, v_idx + 2])
        triangles.append([v_idx, v_idx + 2, v_idx + 3])

    # 2. Triangulate Floor and Ceiling Caps
    tri = Delaunay(poly_pts_pixels)
    valid_simplices = []

    for simplex in tri.simplices:
        pts = poly_pts_pixels[simplex]
        centroid = np.mean(pts, axis=0)
        # Verify the triangle center is inside the room perimeter
        if cv2.pointPolygonTest(approx_polygon, (float(centroid[0]), float(centroid[1])), False) >= 0:
            valid_simplices.append(simplex)

    # Floor Cap
    floor_start_idx = len(vertices)
    for p in poly_pts_meters:
        vertices.append([p[0], p[1], ROOM_MIN_Z])
    for simplex in valid_simplices:
        triangles.append([floor_start_idx + simplex[0], floor_start_idx + simplex[1], floor_start_idx + simplex[2]])

    # Ceiling Cap
    ceil_start_idx = len(vertices)
    for p in poly_pts_meters:
        vertices.append([p[0], p[1], ROOM_MAX_Z])
    for simplex in valid_simplices:
        triangles.append([ceil_start_idx + simplex[0], ceil_start_idx + simplex[2], ceil_start_idx + simplex[1]])

    # Generate and Save Mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(obj_filename), mesh)
    print(f"   Success! Saved 3D mesh to {obj_filename}")

    # --- STEP 6: DXF EXPORT ---
    print("Step 6: Writing DXF File...")
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    for i in range(num_pts):
        p1 = poly_pts_meters[i]
        p2 = poly_pts_meters[(i + 1) % num_pts]
        msp.add_line((p1[0], p1[1]), (p2[0], p2[1]))

    doc.saveas(dxf_filename)
    print(f"   Success! Saved 2D floor plan to {dxf_filename}")

    # --- STEP 7: PREVIEW ---
    if show_viewer:
        print("   Opening Floor Plan Preview...")
        plt.figure(figsize=(8, 8))

        plot_x = np.append(poly_pts_meters[:, 0], poly_pts_meters[0, 0])
        plot_y = np.append(poly_pts_meters[:, 1], poly_pts_meters[0, 1])

        plt.plot(plot_x, plot_y, 'b-', linewidth=2, label="Detected Walls")
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
        description="Generate 2D .dxf floor plans and 3D .obj meshes from .mcap LiDAR scans.")
    parser.add_argument("input_paths", nargs="+", help="Path(s) to the input .mcap file(s) or directory(ies).")
    parser.add_argument("-s", "--show-scan", action="store_true", help="Pop up a blueprint preview of the floor plan.")
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
            process_mcap(f, show_viewer=args.show_scan)
        except Exception as e:
            print(f"An error occurred while processing {f.name}: {e}")


if __name__ == "__main__":
    main()