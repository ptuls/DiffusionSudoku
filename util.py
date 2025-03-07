import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image


def visualize_sudoku(
    board, title=None, cmap=None, show_values=True, figsize=(6, 6), outer_grid_size=9
):
    """
    Visualize a Sudoku board with different colors for each number.

    Args:
        board: A 9x9 numpy array or list of lists representing the Sudoku board
        title: Optional title for the plot
        cmap: Optional custom colormap (default is a pastel colormap)
        show_values: Whether to display the numerical values in cells
        figsize: Size of the figure (width, height) in inches

    Returns:
        A PIL Image of the visualization
    """
    # Create a new figure for each board to prevent any sharing
    plt.clf()  # Clear the current figure
    plt.close("all")  # Close all figures
    fig = plt.figure(figsize=(6, 6))

    # Convert to numpy array if it's a list
    if isinstance(board, list):
        board = np.array(board)

    # Create a default colormap if none provided
    if outer_grid_size == 9:
        if cmap is None:
            # Create a colormap with 10 colors (0-9, where 0 is empty)
            colors = [
                "#FFFFFF",  # 0: White (empty)
                "#FFB3BA",  # 1: Light pink
                "#FFDFBA",  # 2: Light orange
                "#FFFFBA",  # 3: Light yellow
                "#BAFFC9",  # 4: Light green
                "#BAE1FF",  # 5: Light blue
                "#D0BAFF",  # 6: Light purple
                "#FFB3F6",  # 7: Light magenta
                "#C4C4C4",  # 8: Light gray
                "#FFD700",  # 9: Gold - changed from light cyan
            ]
            cmap = ListedColormap(colors)
    else:
        # Create a colormap with colors from red to blue for numbers 0 to outer_grid_size
        # White for 0 (empty cells)
        colors = ["#FFFFFF"]
        # Linear interpolation from red to blue for numbers 1 to outer_grid_size
        for i in range(outer_grid_size):
            r = int(255 * (outer_grid_size - i) / outer_grid_size)
            b = int(255 * i / outer_grid_size)
            colors.append(f"#{r:02x}00{b:02x}")
        cmap = ListedColormap(colors)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the board
    im = ax.imshow(board, cmap=cmap, vmin=0, vmax=outer_grid_size)

    inner_grid_size = int(outer_grid_size**0.5)

    # Add grid lines
    for i in range(outer_grid_size + 1):
        lw = 2 if i % inner_grid_size == 0 else 0.5
        ax.axhline(i - 0.5, color="black", linewidth=lw)
        ax.axvline(i - 0.5, color="black", linewidth=lw)

    # Add values to cells if requested
    if show_values:
        for i in range(outer_grid_size):
            for j in range(outer_grid_size):
                if board[i, j] != 0:
                    ax.text(
                        j,
                        i,
                        str(board[i, j]),
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                    )

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title if provided
    if title:
        ax.set_title(title)

    plt.tight_layout()

    # Convert to PIL image
    pil_image = fig_to_pil(fig)
    plt.close(fig)  # Close the figure to avoid displaying it
    return pil_image


def fig_to_pil(fig):
    """Convert a matplotlib figure to a PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img
