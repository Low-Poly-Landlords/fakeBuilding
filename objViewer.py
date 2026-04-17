import open3d as o3d
import argparse
import sys
from pathlib import Path


def view_obj(file_path):
    path = Path(file_path)

    if not path.is_file() or path.suffix.lower() != '.obj':
        print(f"Error: '{file_path}' is not a valid .obj file.")
        sys.exit(1)

    print(f"Loading mesh: {path.name}...")

    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(str(path))

    if not mesh.has_vertices():
        print("Error: The mesh has no vertices. It might be empty or corrupted.")
        sys.exit(1)

    # Recompute normals so the lighting reflects correctly in the viewer
    mesh.compute_vertex_normals()

    print("Opening viewer... (Press 'Q' or 'Esc' to close)")

    # Launch the interactive viewer
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=f"3D Preview - {path.name}",
        width=1024,
        height=768,
        mesh_show_wireframe=True,  # Optional: Adds a wireframe overlay so you can see the polygons
        mesh_show_back_face=True  # Ensures you can see the inside walls of the room
    )


def main():
    parser = argparse.ArgumentParser(description="Quickly view 3D .obj files.")
    parser.add_argument("input_path", help="Path to the .obj file you want to view.")
    args = parser.parse_args()

    view_obj(args.input_path)


if __name__ == "__main__":
    main()