import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    print("Loading Mesh...")
    try:
        mesh = o3d.io.read_triangle_mesh("trialRoom.obj")
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
    except Exception as e:
        print("Could not load trialRoom.obj")
        return

    print("Loading Cameras...")
    cameras = []
    cam_centers = []
    try:
        with open("colmap_workspace/sparse/images.txt", "r") as f:
            lines = f.readlines()
    except Exception as e:
        print("Could not find images.txt")
        return

    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.split()
        if len(parts) >= 9 and parts[0].isdigit():
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])

            R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t_wc = np.array([tx, ty, tz])

            # Convert to Camera-to-World
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            cam_centers.append(t_cw)

            # Draw red camera pyramid
            points = [
                t_cw,
                t_cw + R_cw @ np.array([-0.2, -0.2, 0.5]),
                t_cw + R_cw @ np.array([0.2, -0.2, 0.5]),
                t_cw + R_cw @ np.array([0.2, 0.2, 0.5]),
                t_cw + R_cw @ np.array([-0.2, 0.2, 0.5])
            ]
            lines_idx = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
            colors = [[1, 0, 0] for _ in range(len(lines_idx))]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines_idx)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            cameras.append(line_set)

    if not cam_centers:
        print("No cameras found in images.txt!")
        return

    cam_centers = np.array(cam_centers)
    bbox = mesh.get_axis_aligned_bounding_box()

    # --- THE DIAGNOSTICS ---
    print("\n--- DIAGNOSTICS ---")
    print(f"Room Bounds: Z({bbox.get_min_bound()[2]:.2f} to {bbox.get_max_bound()[2]:.2f})")
    print(
        f"Cameras Center: X={np.mean(cam_centers[:, 0]):.2f}, Y={np.mean(cam_centers[:, 1]):.2f}, Z={np.mean(cam_centers[:, 2]):.2f}")
    print("-------------------\n")

    print("Opening X-Ray Visualizer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="X-Ray Camera Debugger")

    vis.add_geometry(mesh)
    for c in cameras:
        vis.add_geometry(c)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    # Enable X-Ray mode
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.mesh_show_wireframe = True

    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()