import open3d as o3d
import json
import numpy as np

INPUT_FILENAME = "hotel_layout.json"
SHOW_CEILINGS = False  # Set to True if you want to see the roof


def get_color(color_name):
    """ Maps text colors to RGB values [0-1] """
    colors = {
        "grey": [0.7, 0.7, 0.7],  # Walls
        "blue": [0.3, 0.3, 0.8],  # Floors
        "red": [0.8, 0.2, 0.2],  # Beds
        "green": [0.2, 0.7, 0.2],  # Couches
        "brown": [0.6, 0.4, 0.2],  # Desks
        "black": [0.1, 0.1, 0.1],  # TVs
        "white": [0.9, 0.9, 0.9]
    }
    return colors.get(color_name, [0.5, 0.5, 0.5])


def main():
    print(f"Loading {INPUT_FILENAME}...")

    try:
        with open(INPUT_FILENAME, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: File not found. Run generate_building_layout.py first!")
        return

    geometries = []

    print(f"Building 3D model from {len(data)} objects...")

    for box_data in data:
        # 1. Skip Ceilings if requested (for better view)
        if not SHOW_CEILINGS and "Ceil" in box_data['name']:
            continue

        # 2. Calculate Dimensions
        min_x, max_x = box_data['min_x'], box_data['max_x']
        min_y, max_y = box_data['min_y'], box_data['max_y']
        min_z, max_z = box_data['min_z'], box_data['max_z']

        width = max_x - min_x
        depth = max_y - min_y
        height = max_z - min_z

        # 3. Create Box Mesh
        # Open3D creates boxes starting at (0,0,0) with (width, height, depth)
        # Note: Open3D uses (width, height, depth) for (x, y, z) usually
        mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=depth, depth=height)

        # 4. Move to correct position
        # Open3D boxes are created at origin, so we translate them to min_x, min_y, min_z
        mesh.translate([min_x, min_y, min_z])

        # 5. Color it
        color_name = box_data.get('color', 'grey')
        rgb = get_color(color_name)
        mesh.paint_uniform_color(rgb)

        # 6. Compute Normals (Makes lighting look real/3D instead of flat)
        mesh.compute_vertex_normals()

        geometries.append(mesh)

    # Add a Coordinate Frame (Red=X, Green=Y, Blue=Z)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    geometries.append(coord_frame)

    print("Opening Visualizer...")
    print("CONTROLS:")
    print("  [Left Click + Drag]: Rotate")
    print("  [Right Click + Drag]: Pan (Move)")
    print("  [Scroll Wheel]: Zoom")
    print("  [Ctrl + Left Click]: Tilt")

    o3d.visualization.draw_geometries(geometries, window_name="Hotel Building Viewer", width=1200, height=800)


if __name__ == "__main__":
    main()