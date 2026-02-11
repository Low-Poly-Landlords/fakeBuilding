import open3d as o3d
import numpy as np
import copy

# --- CONFIGURATION ---
INPUT_FILE = "final_3d_room.ply"
OUTPUT_FILE = "merged_room.obj"

# Tuning Parameters
DISTANCE_THRESHOLD = 0.05  # 5cm tolerance for points to belong to a plane
PARALLEL_THRESHOLD = 0.9  # 0.9 = Roughly parallel, 0.99 = Perfectly parallel
PLANE_DISTANCE_TOLERANCE = 0.20  # Merge planes if they are within 20cm
MIN_POINTS = 100  # Minimum points to count as a wall
ITERATIONS = 20  # How many planes to look for


def get_plane_center(points):
    """Calculates the center of a cloud of points."""
    return np.mean(points, axis=0)


def main():
    print(f"Loading {INPUT_FILE}...")
    pcd = o3d.io.read_point_cloud(INPUT_FILE)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # Store found walls here
    detected_planes = []

    print("Iteratively finding and merging planes...")

    for i in range(ITERATIONS):
        if len(pcd.points) < MIN_POINTS: break

        # 1. Detect a candidate plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=DISTANCE_THRESHOLD,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        [a, b, c, d] = plane_model
        normal_candidate = np.array([a, b, c])

        # Extract candidate points
        candidate_cloud = pcd.select_by_index(inliers)
        candidate_points = np.asarray(candidate_cloud.points)
        candidate_center = get_plane_center(candidate_points)

        # 2. Check against EXISTING planes
        merged = False

        # FIX: Use enumerate so we have the index (idx) without asking Python to search for it
        for idx, existing in enumerate(detected_planes):
            [ea, eb, ec, ed] = existing['equation']
            normal_existing = np.array([ea, eb, ec])

            # Check Parallelism (Dot Product)
            dot = abs(np.dot(normal_candidate, normal_existing))

            if dot > PARALLEL_THRESHOLD:
                # Check Distance
                dist = abs(np.dot(normal_existing, candidate_center) + ed)

                if dist < PLANE_DISTANCE_TOLERANCE:
                    print(f"   Merge: Plane match found for Wall #{idx + 1}")

                    # Merge the new points into the existing wall's list
                    existing['points'].extend(candidate_points)
                    merged = True
                    break

        if not merged:
            print(f"   New: Detected Wall #{len(detected_planes) + 1}")
            detected_planes.append({
                'equation': plane_model,
                'points': list(candidate_points)  # Convert to list for easier extending
            })

        # 3. Remove points from main cloud
        pcd = pcd.select_by_index(inliers, invert=True)

    # --- BUILD FINAL MESH ---
    print(f"Result: Consolidated into {len(detected_planes)} unique walls.")
    final_mesh = o3d.geometry.TriangleMesh()
    vis_geometries = []

    for idx, plane in enumerate(detected_planes):
        # Convert list of points back to Open3D PointCloud
        wall_pcd = o3d.geometry.PointCloud()
        wall_pcd.points = o3d.utility.Vector3dVector(plane['points'])

        # Generate a Solid Box (OBB) for this wall
        color = [np.random.random(), np.random.random(), np.random.random()]
        wall_pcd.paint_uniform_color(color)

        obb = wall_pcd.get_oriented_bounding_box()
        obb.color = color

        # Create the solid mesh block
        mesh_box = o3d.geometry.TriangleMesh.create_box(
            width=obb.extent[0],
            height=obb.extent[1],
            depth=obb.extent[2]
        )
        # Move box to correct location
        mesh_box.translate(-mesh_box.get_center())
        mesh_box.rotate(obb.R, center=[0, 0, 0])
        mesh_box.translate(obb.center)
        mesh_box.paint_uniform_color(color)
        mesh_box.compute_vertex_normals()

        # Add to final output
        final_mesh += mesh_box
        vis_geometries.append(mesh_box)
        vis_geometries.append(obb)  # Show wireframe too

    # Save to file
    print(f"Saving merged mesh to {OUTPUT_FILE}...")
    o3d.io.write_triangle_mesh(OUTPUT_FILE, final_mesh)

    o3d.visualization.draw_geometries(vis_geometries, window_name="Merged Walls Fixed")


if __name__ == "__main__":
    main()