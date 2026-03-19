import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
DXF_FILE = "floor_plan.dxf"

def view_dxf(filename):
    """
    Opens and displays a DXF file using matplotlib, ensuring colors are shown.
    """
    if not os.path.exists(filename):
        print(f"Error: DXF file '{filename}' not found.")
        return

    try:
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Error: Could not read DXF file '{filename}'.")
        return
    except ezdxf.DXFError:
        print(f"Error: Invalid or corrupted DXF file '{filename}'.")
        return

    msp = doc.modelspace()

    # Create a matplotlib figure and axes
    fig, ax = plt.subplots()

    # Create a backend for matplotlib
    backend = MatplotlibBackend(ax)

    # The RenderContext holds the drawing state and resources.
    # By default, it resolves colors to black, so we need to configure it.
    context = RenderContext(doc)
    
    # The Frontend manages the drawing process
    frontend = Frontend(context, backend)
    frontend.draw_layout(msp, finalize=True)

    # Set aspect ratio to equal to preserve proportions and show the plot
    ax.set_aspect('equal', 'box')
    ax.set_title(f"DXF Viewer: {os.path.basename(filename)}")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    fig.canvas.manager.set_window_title(f"DXF Viewer: {filename}")
    plt.show()


if __name__ == "__main__":
    print(f"Opening {DXF_FILE} for viewing...")
    view_dxf(DXF_FILE)
