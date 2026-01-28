#run this to make a set of "floor plans" for the building. must run after building gen.

import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

INPUT_FILENAME = "hotel_layout.json"
OUTPUT_FILENAME = "hotel_floor_plans.pdf"

# --- CONFIG ---
FLOOR_HEIGHT = 3.5
NUM_FLOORS = 5

# Color Mapping (JSON Color -> Matplotlib Color)
COLOR_MAP = {
    "grey": "#404040",  # Dark Grey Walls
    "dark_grey": "#000000",  # Exterior Walls (Black)
    "blue": "#E0E0FF",  # Light Blue Floor
    "red": "#FFCCCC",  # Red Beds
    "green": "#CCFFCC",  # Green Couches
    "brown": "#D2B48C",  # Tan Desk/Office
    "black": "#202020",  # TVs/Poles
    "white": "#FFFFFF"
}

# Z-Order (What draws on top of what)
Z_LAYERS = {
    "Floor": 0,
    "Wall": 2,
    "Furniture": 3,
    "Parapet": 4,
    "Awning": 5
}


def get_layer(name):
    """ Determines drawing order based on object name """
    if "Floor" in name: return 0
    if "Bed" in name or "Couch" in name or "Desk" in name or "Stand" in name: return 3
    if "Awning" in name: return 5
    if "Wall" in name or "Shaft" in name or "Para" in name: return 2
    return 1


def main():
    print(f"Reading {INPUT_FILENAME}...")
    try:
        with open(INPUT_FILENAME, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: hotel_layout.json not found. Run generate_building_perfect.py first.")
        return

    print(f"Generating PDF: {OUTPUT_FILENAME}...")

    with PdfPages(OUTPUT_FILENAME) as pdf:
        # Loop through each floor to create a page
        for floor_idx in range(NUM_FLOORS):

            # Setup Plot
            fig, ax = plt.subplots(figsize=(11, 8.5))  # Landscape Letter
            ax.set_aspect('equal')
            ax.set_title(f"Floor Plan - Level {floor_idx + 1}", fontsize=16, pad=20)

            # Filter objects for this floor
            # We check if the object's Z-center is within this floor's range
            floor_min_z = floor_idx * FLOOR_HEIGHT
            floor_max_z = (floor_idx + 1) * FLOOR_HEIGHT

            # Special case: Roof objects (Parapets) usually sit exactly on the boundary
            # If it's the top floor, include things slightly above it
            if floor_idx == NUM_FLOORS - 1:
                floor_max_z += 2.0

            floor_objects = []

            for obj in data:
                # Calculate center Z of the object
                z_center = (obj['min_z'] + obj['max_z']) / 2

                # SKIP CEILINGS and ROOFS (They block the view)
                if "Ceil" in obj['name'] or "Roof" in obj['name']:
                    continue

                # Check if object belongs to this floor
                if floor_min_z <= z_center < floor_max_z:
                    floor_objects.append(obj)

            # Sort objects by Z-Layer so furniture draws ON TOP of floors
            floor_objects.sort(key=lambda x: get_layer(x['name']))

            # Draw Objects
            for obj in floor_objects:
                min_x = obj['min_x']
                min_y = obj['min_y']
                width = obj['max_x'] - min_x
                depth = obj['max_y'] - min_y

                c_name = obj.get('color', 'grey')
                face_color = COLOR_MAP.get(c_name, '#808080')

                # Formatting based on type
                edge_color = None
                linewidth = 0

                # Styling Walls
                if "Wall" in obj['name'] or "Shaft" in obj['name']:
                    edge_color = 'black'
                    linewidth = 0.5

                # Styling Furniture
                if get_layer(obj['name']) == 3:
                    edge_color = '#404040'
                    linewidth = 0.5

                rect = Rectangle(
                    (min_x, min_y), width, depth,
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=linewidth,
                    zorder=get_layer(obj['name'])
                )
                ax.add_patch(rect)

            # Add Labels
            # Label the Elevator and Rooms
            if floor_idx > 0:
                ax.text(-5, 0, "ELEV / STAIRS", ha='center', va='center', fontsize=6, color='white', fontweight='bold')
                ax.text(-1.5, 0, "LIFT LOBBY", ha='center', va='center', fontsize=6, color='black')
            else:
                ax.text(5, -5, "RECEPTION", ha='center', va='center', fontsize=8, color='black')

            # Auto-scale axis
            ax.autoscale_view()

            # Add grid and remove axis ticks for cleaner look
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlabel("Meters (X)")
            ax.set_ylabel("Meters (Y)")

            # Save Page
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  - Rendered Page {floor_idx + 1}")

    print("Done!")


if __name__ == "__main__":
    main()