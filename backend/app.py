import io
import random
import zipfile
from multiprocessing import Pool, cpu_count
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Defects Dataset Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SettingsModel(BaseModel):
    random_seed: Optional[int] = Field(None, description="Зерно генератора псевдовипадкових чисел")
    img_size: int = Field(200, ge=50, le=1000, description="Розмір сторони квадратного зображення в пікселях")
    grid_cells: int = Field(10, ge=2, le=50,
                            description="Кількість клітинок сітки по стороні (наприклад, 10 для 10x10)")


class DatasetRequest(BaseModel):
    total_samples: int = Field(..., ge=1, le=5000, description="Кількість пар зразків для генерації")


system_config = {
    "random_seed": None,
    "img_size": 200,
    "grid_cells": 10
}


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


def add_random_spots(img_array, mask, count, radius_range, value, img_size):
    valid_coords = np.argwhere(mask)
    if len(valid_coords) == 0:
        return
    for _ in range(count):
        cy, cx = valid_coords[random.randint(0, len(valid_coords) - 1)]
        r = random.randint(*radius_range)
        y_low, y_high = max(0, cy - r), min(img_size, cy + r + 1)
        x_low, x_high = max(0, cx - r), min(img_size, cx + r + 1)

        y_grid, x_grid = np.ogrid[y_low:y_high, x_low:x_high]
        spot = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= r ** 2
        sub_mask = mask[y_low:y_high, x_low:x_high] & spot
        img_array[y_low:y_high, x_low:x_high][sub_mask] = value


def apply_background_noise(img_array, mode, bg_mask, img_size):
    bg_coords = np.argwhere(bg_mask)
    n_bg = len(bg_coords)

    if mode == 0:
        pass
    elif mode == 1:
        idx = bg_coords[random.randint(0, n_bg - 1)]
        img_array[idx[0], idx[1]] = 0
    elif mode == 2:
        idx = bg_coords[random.randint(0, n_bg - 1)]
        img_array[idx[0], idx[1]] = 200
    elif mode == 3:
        samples = bg_coords[np.random.choice(n_bg, random.randint(50, 100), replace=False)]
        img_array[samples[:, 0], samples[:, 1]] = 0
    elif mode == 4:
        samples = bg_coords[np.random.choice(n_bg, random.randint(50, 100), replace=False)]
        img_array[samples[:, 0], samples[:, 1]] = 200
    elif mode == 5:
        add_random_spots(img_array, bg_mask, count=random.randint(4, 20), radius_range=(1, 4), value=0,
                         img_size=img_size)
    elif mode == 6:
       add_random_spots(img_array, bg_mask, count=random.randint(4, 20), radius_range=(1, 4), value=200,
                                     img_size=img_size)
    elif mode == 7:
        img_array[bg_mask] = 200


def apply_defect(img_array, mode, circle_mask, center, radius, img_size):
    circle_coords = np.argwhere(circle_mask)
    n_c = len(circle_coords)

    if mode == 1:
        idx = circle_coords[random.randint(0, n_c - 1)]
        img_array[idx[0], idx[1]] = 255
    elif mode == 2:
        idx = circle_coords[random.randint(0, n_c - 1)]
        img_array[idx[0], idx[1]] = 45
    elif mode == 3:
        samples = circle_coords[np.random.choice(n_c, random.randint(50, 100), replace=False)]
        img_array[samples[:, 0], samples[:, 1]] = 255
    elif mode == 4:
        samples = circle_coords[np.random.choice(n_c, random.randint(50, 100), replace=False)]
        img_array[samples[:, 0], samples[:, 1]] = 45
    elif mode == 5:
        add_random_spots(img_array, circle_mask, count=random.randint(4, 20), radius_range=(1, 4), value=255,
                         img_size=img_size)
    elif mode == 6:
        add_random_spots(img_array, circle_mask, count=random.randint(4, 20), radius_range=(1, 4), value=45,
                         img_size=img_size)
    elif mode == 7:
        fraction = random.uniform(0.20, 0.40)
        hole_r = int(radius * np.sqrt(fraction))
        max_offset = max(1, radius - hole_r)
        angle = random.uniform(0, 2 * np.pi)
        offset = random.uniform(0, max_offset)
        cy = int(center[1] + offset * np.sin(angle))
        cx = int(center[0] + offset * np.cos(angle))

        y_grid, x_grid = np.ogrid[:img_size, :img_size]
        hole_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= hole_r ** 2
        img_array[circle_mask & hole_mask] = 255


def compute_grid_features(template, sample, img_size, grid_cells):
    diff = template.astype(np.float32) - sample.astype(np.float32)
    cell_size = img_size // grid_cells
    features = []

    for i in range(grid_cells):
        for j in range(grid_cells):
            cell = diff[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            features.append(float(np.mean(cell)))

    return features


def generate_pair_worker(args):
    sample_id, bg_mode, defect_mode, img_size, grid_cells, seed = args

    if seed is not None:
        random.seed(seed + sample_id)
        np.random.seed((seed + sample_id) % (2 ** 32))

    circle_mask, bg_mask, center, radius = make_masks(img_size)
    template = get_template(img_size)

    sample_clean = template.copy()
    apply_background_noise(sample_clean, bg_mode, bg_mask, img_size)

    sample_defective = sample_clean.copy()
    apply_defect(sample_defective, defect_mode, circle_mask, center, radius, img_size)

    features_clean = compute_grid_features(template, sample_clean, img_size, grid_cells)
    features_defective = compute_grid_features(template, sample_defective, img_size, grid_cells)

    buf_clean = io.BytesIO()
    Image.fromarray(sample_clean).save(buf_clean, format="PNG")
    clean_bytes = buf_clean.getvalue()
    clean_filename = f"{sample_id:05d}_class0_bg{bg_mode}_def0.png"

    buf_def = io.BytesIO()
    Image.fromarray(sample_defective).save(buf_def, format="PNG")
    def_bytes = buf_def.getvalue()
    def_filename = f"{sample_id:05d}_class1_bg{bg_mode}_def{defect_mode}.png"

    return [
        (clean_filename, clean_bytes, features_clean, 0),
        (def_filename, def_bytes, features_defective, 1)
    ]


def generate_dataset(total_samples: int):
    img_size = system_config["img_size"]
    grid_cells = system_config["grid_cells"]
    seed = system_config["random_seed"]

    tasks = []
    sample_id = 0

    for bg_mode in range(0, 8):
        for defect_mode in range(1, 8):
            tasks.append((sample_id, bg_mode, defect_mode, img_size, grid_cells, seed))
            sample_id += 1

    while len(tasks) < total_samples:
        bg_mode = random.randint(0, 7)
        defect_mode = random.randint(1, 7)
        tasks.append((sample_id, bg_mode, defect_mode, img_size, grid_cells, seed))
        sample_id += 1

    tasks = tasks[:total_samples]

    processes_count = max(1, int(cpu_count() / 2))
    with Pool(processes=processes_count) as pool:
        batch_results = pool.map(generate_pair_worker, tasks)

    return batch_results


@app.post("/settings", summary="1. Встановлення налаштувань системи")
def set_settings(settings: SettingsModel):
    if settings.img_size % settings.grid_cells != 0:
        raise HTTPException(
            status_code=400,
            detail=f"img_size ({settings.img_size}) має націло ділитися на grid_cells ({settings.grid_cells})"
        )

    system_config["random_seed"] = settings.random_seed
    system_config["img_size"] = settings.img_size
    system_config["grid_cells"] = settings.grid_cells

    if settings.random_seed is not None:
        random.seed(settings.random_seed)
        np.random.seed(settings.random_seed % (2 ** 32))

    return {
        "status": "success",
        "message": "Налаштування успішно застосовано",
        "current_settings": system_config
    }


@app.get("/template", summary="2. Отримання еталонного зображення")
def get_template_endpoint():
    size = system_config["img_size"]
    template_arr = get_template(size)
    buffer = io.BytesIO()
    Image.fromarray(template_arr).save(buffer, format="PNG", compress_level=0)

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache"
        }
    )


@app.post("/dataset", summary="3. Генерація датасету (повертає ZIP-архів)")
def generate_dataset_endpoint(req: DatasetRequest):
    batch_results = generate_dataset(req.total_samples)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for pair in batch_results:
            for filename, png_bytes, _, _ in pair:
                zip_file.writestr(filename, png_bytes)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=dataset.zip"}
    )