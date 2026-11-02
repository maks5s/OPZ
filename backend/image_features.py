import numpy as np


def make_masks(img_size: int):
    center = (img_size // 2, img_size // 2)
    radius = int(img_size * 0.35)
    y, x = np.ogrid[:img_size, :img_size]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    circle_mask = dist <= radius
    background_mask = ~circle_mask
    return circle_mask, background_mask, center, radius


def get_template(img_size: int) -> np.ndarray:
    circle_mask, _, _, _ = make_masks(img_size)
    img = np.full((img_size, img_size), 255, dtype=np.uint8)
    img[circle_mask] = 0
    return img


def compute_grid_intensities(
    template: np.ndarray,
    sample: np.ndarray,
    grid_cells: int = 20,
) -> np.ndarray:
    height, width = template.shape[:2]

    diff_image = template.astype(np.float32) - sample.astype(np.float32)

    cell_h = height // grid_cells
    cell_w = width // grid_cells

    features = []

    for row in range(grid_cells):
        for col in range(grid_cells):
            y_start = row * cell_h
            y_end = (row + 1) * cell_h
            x_start = col * cell_w
            x_end = (col + 1) * cell_w

            cell = diff_image[y_start:y_end, x_start:x_end]
            features.append(float(np.mean(cell)))

    return np.array(features, dtype=np.float32)
