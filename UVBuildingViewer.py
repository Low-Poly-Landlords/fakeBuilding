import open3d as o3d
import os
import glob
import numpy as np

# --- CONFIGURATION ---
obj_file = "photoreal_room.obj"

# --- AUTO-DETECT TEXTURE ---
image_files = glob.glob("photoreal_room*.png") + glob.glob("photoreal_room*.jpg")

if not image_files:
    print("Error: Could not find the texture image in this folder!")
else:
    texture_file = image_files[0]

    if not os.path.exists(obj_file):
        print(f"Error: Could not find {obj_file}")
    else:
        print(f"Loading {obj_file}...")
        mesh = o3d.io.read_triangle_mesh(obj_file)

        # 1. Check if the OBJ actually has UV wrapping data
        if not mesh.has_triangle_uvs():
            print("CRITICAL ERROR: The OBJ file does not contain UV coordinates!")
        else:
            print(f"Forcing texture load from {texture_file}...")
            tex = o3d.io.read_image(texture_file)
            mesh.textures = [tex]

            # 2. THE C++ CRASH FIX: Assign Material ID 0 to all triangles
            num_triangles = len(mesh.triangles)
            mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(num_triangles, dtype=np.int32))

            print("Success! Texture applied and memory safe.")

            print("Opening Viewer...")
            o3d.visualization.draw_geometries([mesh],
                                              window_name="AAA Photoreal Room",
                                              mesh_show_back_face=True)