import ezdxf
import math
import argparse
import os
from pathlib import Path


def print_dxf_data(filename):
    if not os.path.exists(filename):
        print(f"Error: DXF file '{filename}' not found.")
        return

    try:
        doc = ezdxf.readfile(filename)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return

    msp = doc.modelspace()

    # Grab all the lines in the drawing
    lines = msp.query('LINE')

    if not len(lines):
        print(f"\n--- Wall Measurements for: {os.path.basename(filename)} ---")
        print("No lines found in this DXF file.")
        return

    print(f"\n--- Wall Measurements for: {os.path.basename(filename)} ---")
    print(f"Total Walls Detected: {len(lines)}\n")

    total_perimeter = 0.0

    for i, line in enumerate(lines):
        start = line.dxf.start
        end = line.dxf.end

        # Calculate the length of the line using the distance formula
        length = math.dist((start.x, start.y), (end.x, end.y))
        total_perimeter += length

        # Print the length and the start/end coordinates
        print(
            f"Wall {i + 1:02d}: {length:>6.3f} meters  |  Coordinates: ({start.x:>6.2f}, {start.y:>6.2f}) to ({end.x:>6.2f}, {end.y:>6.2f})")

    print("-" * 50)
    print(f"Total Perimeter: {total_perimeter:.3f} meters\n")


def main():
    parser = argparse.ArgumentParser(description="Output DXF wall lengths to console.")
    # Changed to accept a generic path (file or folder)
    parser.add_argument("input_path", nargs='?', default="BIG SCAN",
                        help="Path to a DXF file or a directory containing DXF files")
    args = parser.parse_args()

    target_path = Path(args.input_path)

    # Check if the input is a single file
    if target_path.is_file():
        if target_path.suffix.lower() == '.dxf':
            print_dxf_data(str(target_path))
        else:
            print(f"Error: The file '{target_path}' is not a .dxf file.")

    # Check if the input is a directory
    elif target_path.is_dir():
        # rglob searches the directory and all subdirectories for .dxf files
        dxf_files = list(target_path.rglob("*.dxf"))

        if not dxf_files:
            print(f"No .dxf files found in directory: {target_path}")
            return

        print(f"Found {len(dxf_files)} .dxf files. Processing...\n{'=' * 50}")
        for dxf_file in dxf_files:
            print_dxf_data(str(dxf_file))

    else:
        print(f"Error: Path '{target_path}' does not exist.")


if __name__ == "__main__":
    main()