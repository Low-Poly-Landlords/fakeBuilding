import json

OUTPUT_FILENAME = "hotel_layout.json"

# --- CONFIGURATION ---
NUM_FLOORS = 5
ROOMS_PER_SIDE = 4
ROOM_WIDTH = 5.0
ROOM_DEPTH = 6.0
HALLWAY_WIDTH = 2.0
FLOOR_HEIGHT = 3.5

# CALCULATED DIMENSIONS
BUILDING_LENGTH = ROOMS_PER_SIDE * ROOM_WIDTH
BUILDING_WIDTH = (ROOM_DEPTH * 2) + HALLWAY_WIDTH

# Geometry List
BUILDING_BOXES = []


def add_box(name, x, y, z, w, d, h, color="grey"):
    box = {
        "name": name,
        "min_x": x - w / 2, "max_x": x + w / 2,
        "min_y": y - d / 2, "max_y": y + d / 2,
        "min_z": z, "max_z": z + h,
        "color": color
    }
    BUILDING_BOXES.append(box)


def make_room(x_center, y_center, floor_z, is_unique=False):
    # Floor/Ceiling
    add_box("Room_Floor", x_center, y_center, floor_z, ROOM_WIDTH, ROOM_DEPTH, 0.1, color="blue")
    add_box("Room_Ceil", x_center, y_center, floor_z + FLOOR_HEIGHT, ROOM_WIDTH, ROOM_DEPTH, 0.1, color="grey")

    # Side Walls
    add_box("Wall_Left", x_center - (ROOM_WIDTH / 2 - 0.1), y_center, floor_z, 0.2, ROOM_DEPTH, FLOOR_HEIGHT)
    add_box("Wall_Right", x_center + (ROOM_WIDTH / 2 - 0.1), y_center, floor_z, 0.2, ROOM_DEPTH, FLOOR_HEIGHT)

    # Back Wall (Exterior)
    y_offset = (ROOM_DEPTH / 2 - 0.1) if y_center > 0 else -(ROOM_DEPTH / 2 - 0.1)
    add_box("Wall_Ext_Back", x_center, y_center + y_offset, floor_z, ROOM_WIDTH, 0.2, FLOOR_HEIGHT, color="dark_grey")

    # Furniture
    bed_y = y_center + 1.5 if y_center > 0 else y_center - 1.5
    add_box("Bed", x_center, bed_y, floor_z, 1.5, 2.0, 0.6, color="red")

    if is_unique:
        couch_y = y_center - 1.0 if y_center > 0 else y_center + 1.0
        add_box("Couch", x_center - 1.5, couch_y, floor_z, 1.0, 2.0, 0.8, color="green")
        add_box("TV_Stand", x_center + 1.5, couch_y, floor_z, 0.5, 1.5, 0.8, color="black")


def main():
    print(f"Generating Perfected Building (Flush Lobby)...")

    center_x = BUILDING_LENGTH / 2
    center_y = 0

    # 1. SHAFTS
    SHAFT_X_CENTER = -5.0
    SHAFT_DEPTH = 6.0
    TOTAL_HEIGHT = NUM_FLOORS * FLOOR_HEIGHT

    add_box("Shaft_Back", SHAFT_X_CENTER - 2, 0, 0, 0.2, SHAFT_DEPTH, TOTAL_HEIGHT, color="dark_grey")
    add_box("Shaft_Side_Top", SHAFT_X_CENTER, SHAFT_DEPTH / 2, 0, 4.0, 0.2, TOTAL_HEIGHT, color="dark_grey")
    add_box("Shaft_Side_Bot", SHAFT_X_CENTER, -SHAFT_DEPTH / 2, 0, 4.0, 0.2, TOTAL_HEIGHT, color="dark_grey")
    add_box("Shaft_Divider", SHAFT_X_CENTER, 0, 0, 4.0, 0.2, TOTAL_HEIGHT, color="grey")

    for i in range(NUM_FLOORS * 2):
        h = i * (FLOOR_HEIGHT / 2)
        add_box("Stair_Platform", SHAFT_X_CENTER, -1.5, h, 2.0, 2.5, 0.1, color="grey")

    # 2. FLOORS
    for f in range(NUM_FLOORS):
        z_floor = f * FLOOR_HEIGHT

        # --- FLOOR 1: LOBBY ---
        if f == 0:
            # FIX: Use Exact Building Dimensions (No +10 or +5)
            lobby_w, lobby_d = BUILDING_LENGTH, BUILDING_WIDTH

            add_box("Lobby_Floor", center_x, 0, z_floor, lobby_w, lobby_d, 0.1, color="blue")
            add_box("Lobby_Ceil", center_x, 0, z_floor + FLOOR_HEIGHT, lobby_w, lobby_d, 0.1, color="grey")

            add_box("Reception_Desk", 5, 0, z_floor, 1.0, 4.0, 1.1, color="brown")

            # Office
            add_box("Office_Floor", 8.5, 0, z_floor, 4.0, 4.0, 0.1, color="brown")
            add_box("Office_Wall_L", 6.5, 0, z_floor, 0.2, 4.0, FLOOR_HEIGHT, color="grey")
            add_box("Office_Wall_R", 10.5, 0, z_floor, 0.2, 4.0, FLOOR_HEIGHT, color="grey")
            add_box("Office_Wall_T", 8.5, 2, z_floor, 4.0, 0.2, FLOOR_HEIGHT, color="grey")
            add_box("Office_Wall_B", 8.5, -2, z_floor, 4.0, 0.2, FLOOR_HEIGHT, color="grey")
            add_box("Office_Header", 6.5, 0, z_floor + 2.2, 0.2, 1.0, FLOOR_HEIGHT - 2.2, color="grey")

            # Exterior - NOW USING EXACT DIMENSIONS
            # Back Wall (Right)
            add_box("Lobby_Ext_Back", BUILDING_LENGTH, 0, z_floor, 0.2, lobby_d, FLOOR_HEIGHT, color="dark_grey")

            # Side Walls
            add_box("Lobby_Ext_Top", center_x, BUILDING_WIDTH / 2, z_floor, BUILDING_LENGTH, 0.2, FLOOR_HEIGHT,
                    color="dark_grey")
            add_box("Lobby_Ext_Bot", center_x, -BUILDING_WIDTH / 2, z_floor, BUILDING_LENGTH, 0.2, FLOOR_HEIGHT,
                    color="dark_grey")

            # Front / Awning
            wall_len = (BUILDING_WIDTH - 4.0) / 2
            add_box("Lobby_Ext_Front_T", 0, (4.0 / 2 + wall_len / 2), z_floor, 0.2, wall_len, FLOOR_HEIGHT,
                    color="dark_grey")
            add_box("Lobby_Ext_Front_B", 0, -(4.0 / 2 + wall_len / 2), z_floor, 0.2, wall_len, FLOOR_HEIGHT,
                    color="dark_grey")
            add_box("Main_Entrance_Header", 0, 0, z_floor + 2.8, 0.2, 4.0, FLOOR_HEIGHT - 2.8, color="dark_grey")
            add_box("Awning", -1.5, 0, z_floor + 3.0, 3.0, 5.0, 0.2, color="black")
            add_box("Pole_T", -2.8, 2.3, z_floor, 0.1, 0.1, 3.0, color="black")
            add_box("Pole_B", -2.8, -2.3, z_floor, 0.1, 0.1, 3.0, color="black")

        # --- FLOORS 2-5 ---
        else:
            # Lift Lobby
            add_box("Lift_Lobby_Floor", -1.5, 0, z_floor, 3.0, HALLWAY_WIDTH + 2.0, 0.1, color="blue")
            add_box("Lift_Lobby_Ceil", -1.5, 0, z_floor + FLOOR_HEIGHT, 3.0, HALLWAY_WIDTH + 2.0, 0.1, color="grey")
            add_box("Lift_Lobby_Wall_Top", -1.5, (HALLWAY_WIDTH + 2.0) / 2, z_floor, 3.0, 0.2, FLOOR_HEIGHT,
                    color="dark_grey")
            add_box("Lift_Lobby_Wall_Bot", -1.5, -(HALLWAY_WIDTH + 2.0) / 2, z_floor, 3.0, 0.2, FLOOR_HEIGHT,
                    color="dark_grey")

            # Shaft Access
            add_box("Shaft_Wall_Center", -3.0, 0, z_floor, 0.2, 1.0, FLOOR_HEIGHT, color="grey")
            add_box("Lift_Header", -3.0, 1.5, z_floor + 2.2, 0.2, 1.5, FLOOR_HEIGHT - 2.2, color="grey")
            add_box("Lift_Side_Wall", -3.0, 2.5, z_floor, 0.2, 1.0, FLOOR_HEIGHT, color="grey")
            add_box("Stair_Header", -3.0, -1.5, z_floor + 2.2, 0.2, 1.5, FLOOR_HEIGHT - 2.2, color="grey")
            add_box("Stair_Side_Wall", -3.0, -2.5, z_floor, 0.2, 1.0, FLOOR_HEIGHT, color="grey")

            # Main Hall
            add_box("Hall_Floor", center_x, 0, z_floor, BUILDING_LENGTH, HALLWAY_WIDTH, 0.1, color="blue")
            add_box("Hall_Ceil", center_x, 0, z_floor + FLOOR_HEIGHT, BUILDING_LENGTH, HALLWAY_WIDTH, 0.1, color="grey")

            # Closet
            closet_x = BUILDING_LENGTH + 1.5
            add_box("Closet_Floor", closet_x, 0, z_floor, 3.0, HALLWAY_WIDTH, 0.1, color="blue")
            add_box("Closet_Ceil", closet_x, 0, z_floor + FLOOR_HEIGHT, 3.0, HALLWAY_WIDTH, 0.1, color="grey")
            add_box("Closet_Wall_Back", closet_x + 1.5, 0, z_floor, 0.2, HALLWAY_WIDTH, FLOOR_HEIGHT, color="dark_grey")
            add_box("Closet_Wall_L", closet_x, HALLWAY_WIDTH / 2, z_floor, 3.0, 0.2, FLOOR_HEIGHT, color="dark_grey")
            add_box("Closet_Wall_R", closet_x, -HALLWAY_WIDTH / 2, z_floor, 3.0, 0.2, FLOOR_HEIGHT, color="dark_grey")
            add_box("Closet_Header", BUILDING_LENGTH, 0, z_floor + 2.2, 0.2, HALLWAY_WIDTH, FLOOR_HEIGHT - 2.2,
                    color="grey")

            # Rooms
            for r in range(ROOMS_PER_SIDE):
                x_pos = r * ROOM_WIDTH + (ROOM_WIDTH / 2)
                make_room(x_pos, (HALLWAY_WIDTH / 2 + ROOM_DEPTH / 2), z_floor, is_unique=(r == 2))
                make_room(x_pos, -(HALLWAY_WIDTH / 2 + ROOM_DEPTH / 2), z_floor, is_unique=(r == 0))

                door_gap = 1.0
                wall_len = (ROOM_WIDTH - door_gap) / 2
                add_box("Hall_Wall", x_pos - ROOM_WIDTH / 2 + wall_len / 2, HALLWAY_WIDTH / 2, z_floor, wall_len, 0.2,
                        FLOOR_HEIGHT)
                add_box("Hall_Wall", x_pos + ROOM_WIDTH / 2 - wall_len / 2, HALLWAY_WIDTH / 2, z_floor, wall_len, 0.2,
                        FLOOR_HEIGHT)
                add_box("Door_Head", x_pos, HALLWAY_WIDTH / 2, z_floor + 2.2, door_gap, 0.2, FLOOR_HEIGHT - 2.2)
                add_box("Hall_Wall", x_pos - ROOM_WIDTH / 2 + wall_len / 2, -HALLWAY_WIDTH / 2, z_floor, wall_len, 0.2,
                        FLOOR_HEIGHT)
                add_box("Hall_Wall", x_pos + ROOM_WIDTH / 2 - wall_len / 2, -HALLWAY_WIDTH / 2, z_floor, wall_len, 0.2,
                        FLOOR_HEIGHT)
                add_box("Door_Head", x_pos, -HALLWAY_WIDTH / 2, z_floor + 2.2, door_gap, 0.2, FLOOR_HEIGHT - 2.2)

    # 3. ROOF
    parapet_h = 1.0

    # Main Roof
    add_box("Roof_Main", center_x, 0, TOTAL_HEIGHT, BUILDING_LENGTH, BUILDING_WIDTH, 0.2, color="dark_grey")
    # Extensions
    add_box("Roof_Lift", -1.5, 0, TOTAL_HEIGHT, 3.0, HALLWAY_WIDTH + 2.0, 0.2, color="dark_grey")
    closet_x = BUILDING_LENGTH + 1.5
    add_box("Roof_Closet", closet_x, 0, TOTAL_HEIGHT, 3.0, HALLWAY_WIDTH, 0.2, color="dark_grey")

    # Parapets
    add_box("Para_Top", center_x, BUILDING_WIDTH / 2 - 0.1, TOTAL_HEIGHT, BUILDING_LENGTH, 0.2, parapet_h,
            color="dark_grey")
    add_box("Para_Bot", center_x, -BUILDING_WIDTH / 2 + 0.1, TOTAL_HEIGHT, BUILDING_LENGTH, 0.2, parapet_h,
            color="dark_grey")
    add_box("Para_L_Top", 0.1, (BUILDING_WIDTH / 4) + 1, TOTAL_HEIGHT, 0.2, (BUILDING_WIDTH / 2) - 2, parapet_h,
            color="dark_grey")
    add_box("Para_L_Bot", 0.1, -(BUILDING_WIDTH / 4) - 1, TOTAL_HEIGHT, 0.2, (BUILDING_WIDTH / 2) - 2, parapet_h,
            color="dark_grey")
    add_box("Para_Lift_Top", -1.5, 2.0, TOTAL_HEIGHT, 3.0, 0.2, parapet_h, color="dark_grey")
    add_box("Para_Lift_Bot", -1.5, -2.0, TOTAL_HEIGHT, 3.0, 0.2, parapet_h, color="dark_grey")
    add_box("Para_R_Top", BUILDING_LENGTH - 0.1, (BUILDING_WIDTH / 4) + 0.5, TOTAL_HEIGHT, 0.2,
            (BUILDING_WIDTH / 2) - 1, parapet_h, color="dark_grey")
    add_box("Para_R_Bot", BUILDING_LENGTH - 0.1, -(BUILDING_WIDTH / 4) - 0.5, TOTAL_HEIGHT, 0.2,
            (BUILDING_WIDTH / 2) - 1, parapet_h, color="dark_grey")
    add_box("Para_Closet_Top", closet_x, 1.0, TOTAL_HEIGHT, 3.0, 0.2, parapet_h, color="dark_grey")
    add_box("Para_Closet_Bot", closet_x, -1.0, TOTAL_HEIGHT, 3.0, 0.2, parapet_h, color="dark_grey")
    add_box("Para_Closet_End", closet_x + 1.5, 0, TOTAL_HEIGHT, 0.2, HALLWAY_WIDTH, parapet_h, color="dark_grey")
    add_box("Shaft_Cap", SHAFT_X_CENTER, 0, TOTAL_HEIGHT, 4.0, SHAFT_DEPTH, 0.2, color="dark_grey")

    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(BUILDING_BOXES, f, indent=4)
    print(f"Generated {len(BUILDING_BOXES)} objects.")


if __name__ == "__main__":
    main()