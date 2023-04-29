import os
import matplotlib.pyplot as plt

def save_image_grid(images, grid_size = (4,4), grid_path="./grid.jpg"):
  try:
    dirs, _ = os.path.split(grid_path)
    # Create the directories if they don't exist
    if dirs:
        os.makedirs(dirs, exist_ok=True)

    # Create image grid
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            idx = i * grid_size[0] + j
            if idx < len(images):
                axes[i, j].imshow(images[idx])
                axes[i, j].axis('off')
    plt.tight_layout()

    fig.savefig(grid_path, bbox_inches='tight')
    plt.close(fig)
  except Exception as e:
    pass