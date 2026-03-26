import os
import sys
import numpy as np
import open3d as o3d
import cv2
import bisect
from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from mcap_zstd_helper import iter_decoded_messages_with_zstd
import json
import ezdxf
import argparse
from pathlib import Path

# --- CONFIGURATION ---
# 1. GHOST REMOVAL
MIN_LIDAR_DIST = 1.0

# 2. LIDAR CALIBRATION
LIDAR_ROLL_OFFSET = 0.0
LIDAR_PITCH_OFFSET = -20.0
LIDAR_YAW_OFFSET = 0.0

# 3. CAMERA CALIBRATION (Restored)
CAM_OFFSET = [0.0, 0.0, 0.05]
CAM_FOV_DEG = 70.0
CAM_ROLL = -17.0
CAM_PITCH = 174
CAM_YAW = 0.0

# 4. EXTRUSION SETTINGS
RESOLUTION = 0.02  # 2cm per pixel for the 2D floor plan

# --- TRANSFORMS ---
LIDAR_FIX = R.from_euler('xyz', [
    90 + LIDAR_ROLL_OFFSET,
    0 + LIDAR_PITCH_OFFSET,
    0 + LIDAR_YAW_OFFSET
], degrees=True).as_matrix()

IMU_FIX = R.from_euler('x', 90, degrees=True)

# Base: Robot Frame -> Camera Optical Frame (Restored)
BASE_ROBOT_TO_CAM = np.array([
    [0, -1, 0],
    [0, 0, -1],
    [1, 0, 0]
])

# User Correction: Applied on top of the base (Restored)
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


def bake_walls_to_atlas(images, imu_data, poly_pts_meters, segment_lengths, total_perimeter,
                        ROOM_MIN_Z, ROOM_MAX_Z, min_x, max_x, min_y, max_y, valid_simplices,
                        K, atlas_img, cam_offset, base_to_cam):
    atlas_h, atlas_w, _ = atlas_img.shape
    cam_h, cam_w = images[0][1].shape[:2]

    # Winner-Takes-All Score Map (replaces the blurry weight_map)
    score_map = np.zeros((atlas_h, atlas_w), dtype=np.float32) - 1.0

    print("   Baking camera images into UV Atlas (Clean Corner Method)...")
    paint_interval = max(1, len(images) // 40)

    for i in range(0, len(images), paint_interval):
        t, img = images[i]
        raw_quat = get_interpolated_pose(t, imu_data)
        robot_rot = (R.from_quat(raw_quat) * IMU_FIX).as_matrix()

        # ==========================================
        # 1. PROCESS VERTICAL WALLS
        # ==========================================
        cumulative_dist = 0
        for j in range(len(poly_pts_meters)):
            p1 = poly_pts_meters[j]
            p2 = poly_pts_meters[(j + 1) % len(poly_pts_meters)]

            wall_3d = np.array([
                [p1[0], p1[1], ROOM_MIN_Z],
                [p2[0], p2[1], ROOM_MIN_Z],
                [p2[0], p2[1], ROOM_MAX_Z],
                [p1[0], p1[1], ROOM_MAX_Z]
            ])

            pts_robot = wall_3d @ robot_rot - cam_offset
            pts_optical = pts_robot @ base_to_cam.T
            z = pts_optical[:, 2]

            # If any corner is behind the camera, skip to prevent math explosions
            if np.any(z < 0.1):
                cumulative_dist += segment_lengths[j]
                continue

            u_cam = (pts_optical[:, 0] * K[0, 0] / z) + K[0, 2]
            v_cam = (pts_optical[:, 1] * K[1, 1] / z) + K[1, 2]

            if np.all(u_cam < 0) or np.all(u_cam > cam_w) or np.all(v_cam < 0) or np.all(v_cam > cam_h):
                cumulative_dist += segment_lengths[j]
                continue

            cam_corners = np.column_stack((u_cam, v_cam)).astype(np.float32)

            u_start = cumulative_dist / total_perimeter
            u_end = (cumulative_dist + segment_lengths[j]) / total_perimeter
            v_bottom, v_top = 0.5, 0.0

            atlas_corners = np.array([
                [u_start * atlas_w, v_bottom * atlas_h],
                [u_end * atlas_w, v_bottom * atlas_h],
                [u_end * atlas_w, v_top * atlas_h],
                [u_start * atlas_w, v_top * atlas_h]
            ], dtype=np.float32)

            H, _ = cv2.findHomography(cam_corners, atlas_corners)
            if H is not None:
                warped_img = cv2.warpPerspective(img, H, (atlas_w, atlas_h))
                mask = np.zeros((atlas_h, atlas_w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, atlas_corners.astype(np.int32), 255)

                # Score based on how centered the wall is in the photo
                cx, cy = np.mean(u_cam), np.mean(v_cam)
                dist_from_center = np.sqrt((cx - cam_w / 2) ** 2 + (cy - cam_h / 2) ** 2)
                view_score = 10000.0 - dist_from_center

                better_mask = (mask > 0) & (view_score > score_map)
                atlas_img[better_mask] = warped_img[better_mask]
                score_map[better_mask] = view_score

            cumulative_dist += segment_lengths[j]

        # ==========================================
        # 2. PROCESS FLOOR & CEILING TRIANGLES
        # ==========================================
        for simplex in valid_simplices:
            p1 = poly_pts_meters[simplex[0]]
            p2 = poly_pts_meters[simplex[1]]
            p3 = poly_pts_meters[simplex[2]]

            # The Parallelogram Trick: Compute a 4th dummy point to satisfy findHomography
            p4 = p2 + p3 - p1

            for is_floor in [True, False]:
                Z_LVL = ROOM_MIN_Z if is_floor else ROOM_MAX_Z

                plane_3d = np.array([
                    [p1[0], p1[1], Z_LVL],
                    [p2[0], p2[1], Z_LVL],
                    [p3[0], p3[1], Z_LVL],
                    [p4[0], p4[1], Z_LVL]
                ])

                pts_robot = plane_3d @ robot_rot - cam_offset
                pts_optical = pts_robot @ base_to_cam.T
                z = pts_optical[:, 2]

                if np.any(z < 0.1):
                    continue

                u_cam = (pts_optical[:, 0] * K[0, 0] / z) + K[0, 2]
                v_cam = (pts_optical[:, 1] * K[1, 1] / z) + K[1, 2]

                if np.all(u_cam[:3] < 0) or np.all(u_cam[:3] > cam_w) or np.all(v_cam[:3] < 0) or np.all(
                        v_cam[:3] > cam_h):
                    continue

                cam_corners = np.column_stack((u_cam, v_cam)).astype(np.float32)

                # Map to Atlas
                atlas_corners = []
                for pt in [p1, p2, p3, p4]:
                    u = ((pt[0] - min_x) / (max_x - min_x)) * 0.5
                    v = ((pt[1] - min_y) / (max_y - min_y)) * 0.5 + 0.5
                    if not is_floor:
                        u += 0.5  # Shift ceiling to the right half of the atlas
                    atlas_corners.append([u * atlas_w, v * atlas_h])

                atlas_corners = np.array(atlas_corners, dtype=np.float32)

                H, _ = cv2.findHomography(cam_corners, atlas_corners)
                if H is not None:
                    warped_img = cv2.warpPerspective(img, H, (atlas_w, atlas_h))

                    # Create mask using ONLY the original 3 points (ignoring the 4th dummy point)
                    mask = np.zeros((atlas_h, atlas_w), dtype=np.uint8)
                    tri_pts = atlas_corners[:3].astype(np.int32)
                    cv2.fillConvexPoly(mask, tri_pts, 255)

                    cx, cy = np.mean(u_cam[:3]), np.mean(v_cam[:3])
                    dist_from_center = np.sqrt((cx - cam_w / 2) ** 2 + (cy - cam_h / 2) ** 2)
                    view_score = 10000.0 - dist_from_center

                    better_mask = (mask > 0) & (view_score > score_map)
                    atlas_img[better_mask] = warped_img[better_mask]
                    score_map[better_mask] = view_score

    return atlas_img


def filter_sharp_angles(polygon, length_scale_for_filter):
    """
    Filters a polygon to remove sharp concave "juts" AND sharp convex "spikes"
    that are unlikely in a building plan.
    It iteratively removes the worst offending point until no more points meet the criteria.
    """
    pixel_threshold = length_scale_for_filter * 4.0
    MIN_CONVEX_ANGLE = 75  # Minimum angle for a "real" corner.

    while True:
        points = polygon.reshape(-1, 2)
        num_points = len(points)
        if num_points < 4:
            break

        signed_area = cv2.contourArea(points.astype(np.float32), oriented=True)
        is_ccw = signed_area > 0

        worst_candidate_idx = -1
        highest_score = 0  # Higher score = better candidate for removal

        for i in range(num_points):
            p_prev = points[(i - 1 + num_points) % num_points]
            p_curr = points[i]
            p_next = points[(i + 1) % num_points]

            v_prev = p_prev - p_curr
            v_next = p_next - p_curr

            len_prev = np.linalg.norm(v_prev)
            len_next = np.linalg.norm(v_next)

            if len_prev == 0 or len_next == 0:
                continue

            cross_product_z = v_prev[0] * v_next[1] - v_prev[1] * v_next[0]

            is_concave = (is_ccw and cross_product_z < 0) or \
                         (not is_ccw and cross_product_z > 0)
            is_convex = (is_ccw and cross_product_z > 0) or \
                        (not is_ccw and cross_product_z < 0)

            angle_at_vertex = np.degrees(np.arccos(np.clip(np.dot(v_prev, v_next) / (len_prev * len_next), -1, 1)))

            should_score = False
            # Condition for removing a concave "dent"
            if is_concave and len_prev < pixel_threshold and len_next < pixel_threshold:
                should_score = True

            # Condition for removing a convex "spike"
            if is_convex and angle_at_vertex < MIN_CONVEX_ANGLE and len_prev < pixel_threshold and len_next < pixel_threshold:
                should_score = True

            if should_score:
                # The score is the distance of the point from the line connecting its neighbors.
                # This works for both concave and convex artifacts.
                line_vec = p_next - p_prev
                if np.linalg.norm(line_vec) == 0: continue

                area = np.abs(np.cross(line_vec, p_curr - p_prev))
                dist_score = area / np.linalg.norm(line_vec)

                if dist_score > highest_score:
                    highest_score = dist_score
                    worst_candidate_idx = i

        if worst_candidate_idx != -1:
            # Remove the point that creates the worst artifact and restart the process
            points = np.delete(points, worst_candidate_idx, axis=0)
            polygon = points.reshape(-1, 1, 2)
        else:
            # No more artifacts to remove
            break

    return polygon


def main():
    parser = argparse.ArgumentParser(description="Process a .mcap file to generate a 3D model and floor plan.")
    parser.add_argument("input_file", help="Path to the input .mcap file.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_stem = input_path.stem
    output_obj = input_path.with_name(f"{output_stem}.obj")
    texture_file = input_path.with_name(f"{output_stem}.png")
    dxf_filename = input_path.with_name(f"{output_stem}.dxf")
    json_filename = input_path.with_name(f"{output_stem}_wall_materials.json")

    if not os.path.exists(args.input_file):
        print(f"Error: Could not find {args.input_file}")
        return

    print("Step 1: Reading Data...")
    reader = make_reader(open(args.input_file, "rb"), decoder_factories=[DecoderFactory()])
    imu_data = []
    lidar_msgs = []
    images = []  # Restored!

    for schema, channel, message, ros_msg in iter_decoded_messages_with_zstd(reader):
        if channel.topic == "/imu/data":
            q = [ros_msg.orientation.x, ros_msg.orientation.y, ros_msg.orientation.z, ros_msg.orientation.w]
            imu_data.append((message.log_time, q))
        elif channel.topic == "/scan":
            lidar_msgs.append((message.log_time, ros_msg))
        elif channel.topic == "/camera/image_raw": # Restored!
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

    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(floor_plan, kernel, iterations=2)
    close_kernel = np.ones((40, 40), np.uint8)
    closed_plan = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed_plan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Error: Could not find walls in the 2D slice.")
        return

    main_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.015 * cv2.arcLength(main_contour, True)
    approx_polygon = cv2.approxPolyDP(main_contour, epsilon, True)

    # Filter out small, sharp concave angles that are not typical of room geometry
    approx_polygon = filter_sharp_angles(approx_polygon, epsilon)

    vertices = []
    triangles = []
    uvs = []

    poly_pts_pixels = approx_polygon.reshape(-1, 2)
    num_pts = len(poly_pts_pixels)

    poly_pts_meters = []
    for p in poly_pts_pixels:
        xm = (p[0] * RESOLUTION) + min_x
        ym = (p[1] * RESOLUTION) + min_y
        poly_pts_meters.append([xm, ym])
    poly_pts_meters = np.array(poly_pts_meters)

    total_perimeter = 0
    segment_lengths = []
    for i in range(num_pts):
        p1 = poly_pts_meters[i]
        p2 = poly_pts_meters[(i + 1) % num_pts]
        dist = np.linalg.norm(p2 - p1)
        segment_lengths.append(dist)
        total_perimeter += dist

    print("Step 4: Unwrapping UVs...")

    # 1. Extrude Walls
    cumulative_dist = 0
    for i in range(num_pts):
        p1 = poly_pts_meters[i]
        p2 = poly_pts_meters[(i + 1) % num_pts]
        x1, y1 = p1
        x2, y2 = p2

        u_start = cumulative_dist / total_perimeter
        cumulative_dist += segment_lengths[i]
        u_end = cumulative_dist / total_perimeter

        v_bottom = 0.5
        v_top = 0.0

        v_idx = len(vertices)
        vertices.extend([
            [x1, y1, ROOM_MIN_Z],
            [x2, y2, ROOM_MIN_Z],
            [x2, y2, ROOM_MAX_Z],
            [x1, y1, ROOM_MAX_Z]
        ])

        triangles.append([v_idx, v_idx + 1, v_idx + 2])
        uvs.extend([[u_start, v_bottom], [u_end, v_bottom], [u_end, v_top]])

        triangles.append([v_idx, v_idx + 2, v_idx + 3])
        uvs.extend([[u_start, v_bottom], [u_end, v_top], [u_start, v_top]])

    # 2. Triangulate Floor and Ceiling
    tri = Delaunay(poly_pts_pixels)
    valid_simplices = []
    for simplex in tri.simplices:
        pts = poly_pts_pixels[simplex]
        centroid = np.mean(pts, axis=0)
        if cv2.pointPolygonTest(approx_polygon, (centroid[0], centroid[1]), False) >= 0:
            valid_simplices.append(simplex)

    # Floor Cap
    floor_start_idx = len(vertices)
    for p in poly_pts_meters:
        vertices.append([p[0], p[1], ROOM_MIN_Z])

    for simplex in valid_simplices:
        triangles.append([
            floor_start_idx + simplex[0],
            floor_start_idx + simplex[1],
            floor_start_idx + simplex[2]
        ])
        for idx in simplex:
            px, py = poly_pts_meters[idx]
            u = ((px - min_x) / (max_x - min_x)) * 0.5
            v = ((py - min_y) / (max_y - min_y)) * 0.5 + 0.5
            uvs.append([u, v])

    # Ceiling Cap
    ceil_start_idx = len(vertices)
    for p in poly_pts_meters:
        vertices.append([p[0], p[1], ROOM_MAX_Z])

    for simplex in valid_simplices:
        triangles.append([
            ceil_start_idx + simplex[0],
            ceil_start_idx + simplex[2],
            ceil_start_idx + simplex[1]
        ])
        for idx in [0, 2, 1]:
            px, py = poly_pts_meters[simplex[idx]]
            u = ((px - min_x) / (max_x - min_x)) * 0.5 + 0.5
            v = ((py - min_y) / (max_y - min_y)) * 0.5 + 0.5
            uvs.append([u, v])

    # --- STEP 5: ASSEMBLE MESH AND BAKE TEXTURE ---
    # FIXED: This section is now properly un-indented out of the ceiling loop!
    print("Step 5: Applying Texture Atlas...")
    final_rigid_mesh = o3d.geometry.TriangleMesh()
    final_rigid_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    final_rigid_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    final_rigid_mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

    # 1. Generate the blank canvas
    atlas_img = np.zeros((4096, 4096, 3), dtype=np.uint8)

    # 2. Setup Camera Intrinsics
    if len(images) > 0:
        h, w, _ = images[0][1].shape
        focal_length = (w / 2) / np.tan(np.deg2rad(CAM_FOV_DEG) / 2)
        K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])

        # 3. Bake the images onto the atlas!
        atlas_img = bake_walls_to_atlas(
            images, imu_data, poly_pts_meters, segment_lengths, total_perimeter,
            ROOM_MIN_Z, ROOM_MAX_Z, min_x, max_x, min_y, max_y, valid_simplices,
            K, atlas_img, CAM_OFFSET, FINAL_ROBOT_TO_CAM
        )

    # Convert BGR (OpenCV) to RGB (Open3D)
    atlas_img_rgb = cv2.cvtColor(atlas_img, cv2.COLOR_BGR2RGB)

    texture = o3d.geometry.Image(atlas_img_rgb)
    final_rigid_mesh.textures = [texture]
    final_rigid_mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(triangles), dtype=np.int32))

    cv2.imwrite(str(texture_file), atlas_img)
    print(f"   Saved baked atlas image to {texture_file}")

    # --- STEP 6: SAVE ---
    print(f"Saving to {output_obj}...")
    o3d.io.write_triangle_mesh(str(output_obj), final_rigid_mesh)

    # ==========================================
    # --- STEP 7: CAD EXPORT & ML MATERIAL DATA ---
    # ==========================================
    print("Step 7: Exporting CAD Floor Plan and Material Data...")

    # 1. Analyze Wall Colors from Texture Atlas
    wall_materials = []
    cumulative_dist = 0
    atlas_h, atlas_w = atlas_img.shape[:2]

    for i in range(num_pts):
        p1 = poly_pts_meters[i]
        p2 = poly_pts_meters[(i + 1) % num_pts]

        # Find where this wall lives on the U-axis of the atlas
        u_start = cumulative_dist / total_perimeter
        u_end = (cumulative_dist + segment_lengths[i]) / total_perimeter
        u_mid = (u_start + u_end) / 2.0
        px_x = int(u_mid * atlas_w)

        # Extract the vertical column of pixels at the wall's midpoint
        # Walls are in the top half of the atlas (V from 0.0 to 0.5)
        color_column_bgr = atlas_img[0:int(atlas_h * 0.5), px_x]

        # Find the most frequent color in the column
        colors, counts = np.unique(color_column_bgr.reshape(-1, 3), axis=0, return_counts=True)
        
        # Handle case where column is empty or has no colors
        if len(colors) > 0:
            most_frequent_bgr = colors[counts.argmax()]
            b, g, r = most_frequent_bgr
        else:
            r, g, b = 0, 0, 0 # Default to black if no color found

        wall_materials.append({
            "wall_id": i,
            "start_pt_meters": [float(p1[0]), float(p1[1])],
            "end_pt_meters": [float(p2[0]), float(p2[1])],
            "length_meters": float(segment_lengths[i]),
            "sampled_midpoint_rgb": [int(r), int(g), int(b)]
        })

        cumulative_dist += segment_lengths[i]

    # 2. Export 2D Floor Plan as DXF with Wall Colors
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Define an offset for the dimension lines to avoid overlapping the walls
    DIM_OFFSET = 0.2  # 20cm offset from the wall

    for i in range(num_pts):
        p1 = poly_pts_meters[i]
        p2 = poly_pts_meters[(i + 1) % num_pts]
        material_info = wall_materials[i]
        r, g, b = material_info["sampled_midpoint_rgb"]

        # Add line to DXF and set its color
        line = msp.add_line((p1[0], p1[1]), (p2[0], p2[1]))
        line.dxf.true_color = ezdxf.rgb2int((r, g, b))

        # Add aligned dimension for each wall segment
        # This places a measurement label parallel to the wall
        msp.add_aligned_dim(p1, p2, distance=DIM_OFFSET).render()

    doc.saveas(dxf_filename)
    print(f"   Saved CAD floor plan to {dxf_filename}")

    # 3. Save material data to a JSON file
    with open(json_filename, "w") as f:
        json.dump(wall_materials, f, indent=4)
    print(f"   Saved material data to {json_filename}")

    print("Opening Viewer...")
    o3d.visualization.draw_geometries([final_rigid_mesh], window_name="UV Mapped Architecture")


if __name__ == "__main__":
    main()