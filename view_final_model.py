import open3d as o3d
import os

# --- CONFIGURATION ---
# Ensure this matches the name of the file you just generated
FILE_PATH = "final_3d_room.ply"


def main():
    print(f"Attempting to load: {FILE_PATH}")

    # 1. Load the Point Cloud
    if not os.path.exists(FILE_PATH):
        print(f"ERROR: Could not find file '{FILE_PATH}'")
        print(f"Make sure you are running this script from the same folder as the .ply file.")
        return

    pcd = o3d.io.read_point_cloud(FILE_PATH)

    if pcd.is_empty():
        print("ERROR: File loaded but contains no points.")
        return

    print(f"Successfully loaded {len(pcd.points)} points.")
    print("\nControls:")
    print("  [Left Click + Drag]  Rotate")
    print("  [Ctrl + Left Click]  Pan")
    print("  [Scroll Wheel]       Zoom")
    print("  [+] or [-]           Change Point Size")
    print("  [N]                  Turn on/off Surface Normals")
    print("  [Q]                  Close Window")

    # 2. Add Axes for Reference (Red=X, Green=Y, Blue=Z)
    # This helps you see which way is 'Up'
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # 3. Visualize
    o3d.visualization.draw_geometries([pcd, axes],
                                      window_name="Room Scan Viewer",
                                      width=1280,
                                      height=720)


if __name__ == "__main__":
    main()