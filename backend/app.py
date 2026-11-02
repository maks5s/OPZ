import asyncio
import base64
import io
import json
import random
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from image_features import compute_grid_intensities, get_template, make_masks


class SettingsModel(BaseModel):
    random_seed: Optional[int] = Field(None, description="Зерно генератора псевдовипадкових чисел")
    img_size: int = Field(200, ge=50, le=1000, description="Розмір сторони квадратного зображення в пікселях")
    grid_cells: int = Field(10, ge=2, le=50,
                            description="Кількість клітинок сітки по стороні (наприклад, 10 для 10x10)")


class TrainingSettingsModel(BaseModel):
    epochs: int = Field(200, ge=1, description="Кількість епох навчання моделі")
    learning_rate: float = Field(0.001, gt=0, description="Швидкість навчання моделі")
    hidden_layers: int = Field(
        1,
        ge=0,
        le=1,
        description="Кількість прихованих шарів: 0 або 1",
    )
    hidden_units: int = Field(
        16,
        ge=1,
        le=256,
        description="Кількість нейронів у прихованому шарі",
    )


class DatasetRequest(BaseModel):
    total_samples: int = Field(..., ge=1, le=5000, description="Кількість пар зразків для генерації")


DEFAULT_SETTINGS = {
    "random_seed": 47,
    "img_size": 200,
    "grid_cells": 10
}

DEFAULT_TRAINING_SETTINGS = {
    "epochs": 200,
    "learning_rate": 0.001,
    "hidden_layers": 1,
    "hidden_units": 16
}

IMAGE_EXTENSIONS = (".png",)
MAX_REPORTED_MISMATCHES = 5
GRID_COLOR = np.array([96, 126, 78], dtype=np.uint8)
MODEL_DECISION_THRESHOLD = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = DEFAULT_SETTINGS.copy()
    app.state.training_settings = DEFAULT_TRAINING_SETTINGS.copy()
    app.state.template = None
    app.state.model = None
    app.state.model_metadata = None
    app.state.model_metrics = None
    app.state.model_training = False
    app.state.model_training_progress = None
    app.state.model_lock = threading.Lock()
    app.state.log_queue = asyncio.Queue()
    app.state.loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="Defects Dataset Generator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def enqueue_log(message_type: str, text: str, **data):
    message = {
        "type": message_type,
        "text": text,
        "timestamp": datetime.now().astimezone().isoformat(),
        **data
    }
    app.state.loop.call_soon_threadsafe(app.state.log_queue.put_nowait, message)


def template_png_response(template: np.ndarray) -> Response:
    buffer = io.BytesIO()
    Image.fromarray(template).save(buffer, format="PNG", compress_level=0)

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache"
        }
    )


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


def generate_dataset(total_samples: int, settings: dict, on_progress=None):
    img_size = settings["img_size"]
    grid_cells = settings["grid_cells"]
    seed = settings["random_seed"]

    tasks = []
    sample_id = 0

    for bg_mode in range(0, 8):
        if bg_mode in (1,2):
            continue
        for defect_mode in range(1, 8):
            if defect_mode in (1, 2, 7):
                continue
            tasks.append((sample_id, bg_mode, defect_mode, img_size, grid_cells, seed))
            sample_id += 1

    while len(tasks) < total_samples:
        bg_mode = random.randint(0, 7)
        defect_mode = random.randint(1, 7)
        if bg_mode in (1,2) or defect_mode in (1,2,7):
            continue
        tasks.append((sample_id, bg_mode, defect_mode, img_size, grid_cells, seed))
        sample_id += 1

    tasks = tasks[:total_samples]

    processes_count = max(1, int(cpu_count() / 2))
    batch_results = []

    with Pool(processes=processes_count) as pool:
        for completed, result in enumerate(pool.imap_unordered(generate_pair_worker, tasks), start=1):
            batch_results.append(result)
            if on_progress is not None:
                on_progress(completed, total_samples)

    batch_results.sort(key=lambda pair: pair[0][0])

    return batch_results


def render_difference_grid(intensities: np.ndarray, grid_cells: int, img_size: int) -> bytes:
    values = np.asarray(intensities, dtype=np.float32).reshape(grid_cells, grid_cells)
    cell_h = max(1, img_size // grid_cells)
    cell_w = max(1, img_size // grid_cells)

    scale = float(np.max(np.abs(values)))
    if scale < 1e-6:
        scale = 1.0

    neutral = np.array([242, 243, 239], dtype=np.float32)
    positive_color = np.array([184, 62, 49], dtype=np.float32)
    negative_color = np.array([47, 91, 154], dtype=np.float32)

    output = np.empty((img_size, img_size, 3), dtype=np.uint8)
    output[:] = neutral.astype(np.uint8)

    for row in range(grid_cells):
        y_start = row * cell_h
        y_end = (row + 1) * cell_h

        for col in range(grid_cells):
            x_start = col * cell_w
            x_end = (col + 1) * cell_w

            ratio = float(np.clip(values[row, col] / scale, -1.0, 1.0))
            target = positive_color if ratio >= 0 else negative_color
            color = neutral + (target - neutral) * abs(ratio)

            output[y_start:y_end, x_start:x_end] = color.astype(np.uint8)

    for row in range(grid_cells + 1):
        y = min(row * cell_h, img_size - 1)
        output[y, :, :] = GRID_COLOR

    for col in range(grid_cells + 1):
        x = min(col * cell_w, img_size - 1)
        output[:, x, :] = GRID_COLOR

    buffer = io.BytesIO()
    Image.fromarray(output).save(buffer, format="PNG")
    return buffer.getvalue()


def render_source_with_grid(source_rgb: np.ndarray, grid_cells: int) -> bytes:
    height, width = source_rgb.shape[:2]
    cell_h = max(1, height // grid_cells)
    cell_w = max(1, width // grid_cells)

    output = np.asarray(source_rgb, dtype=np.uint8).copy()

    for row in range(grid_cells + 1):
        y = min(row * cell_h, height - 1)
        output[y, :, :] = GRID_COLOR

    for col in range(grid_cells + 1):
        x = min(col * cell_w, width - 1)
        output[:, x, :] = GRID_COLOR

    buffer = io.BytesIO()
    Image.fromarray(output).save(buffer, format="PNG")
    return buffer.getvalue()


def clear_model_state(reason: Optional[str] = None):
    with app.state.model_lock:
        had_model = app.state.model is not None
        app.state.model = None
        app.state.model_metadata = None
        app.state.model_metrics = None
        app.state.model_training_progress = None

    if had_model and reason:
        enqueue_log("model_invalidated", reason)


def build_model_status() -> dict:
    settings = dict(app.state.settings)
    metadata = app.state.model_metadata
    training = bool(app.state.model_training)
    model_ready = app.state.model is not None

    expected_feature_count = int(settings["grid_cells"]) * int(settings["grid_cells"])
    stale = bool(
        model_ready
        and metadata
        and (
            metadata.get("img_size") != settings["img_size"]
            or metadata.get("grid_cells") != settings["grid_cells"]
            or metadata.get("feature_count") != expected_feature_count
        )
    )
    ready = model_ready and metadata is not None and not stale

    if training:
        status = "training"
        message = "Модель навчається"
    elif stale:
        status = "stale"
        message = "Модель навчена для інших налаштувань зображення або сітки"
    elif ready:
        status = "ready"
        message = "Модель готова до роботи"
    else:
        status = "not_ready"
        message = "Модель ще не навчена"

    return {
        "ready": ready,
        "training": training,
        "status": status,
        "message": message,
        "model": metadata,
        "metrics": app.state.model_metrics,
        "progress": app.state.model_training_progress,
        "settings": {
            "img_size": settings["img_size"],
            "grid_cells": settings["grid_cells"]
        }
    }


def validate_training_archive(contents: bytes, expected_size: int) -> dict:
    try:
        archive = zipfile.ZipFile(io.BytesIO(contents))
    except zipfile.BadZipFile:
        detail = "Файл не є коректним ZIP-архівом"
        enqueue_log("model_training_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    image_count = 0
    class_counts = {0: 0, 1: 0}
    unlabeled = []
    mismatches = []

    try:
        for name in archive.namelist():
            if name.endswith("/") or not name.lower().endswith(IMAGE_EXTENSIONS):
                continue

            base_name = name.rsplit("/", 1)[-1]
            normalized_name = base_name.lower()
            if normalized_name == "template.png":
                continue

            image_count += 1

            try:
                with archive.open(name) as member:
                    raw_bytes = member.read()

                with Image.open(io.BytesIO(raw_bytes)) as image:
                    width, height = image.size
                    image.verify()
            except Exception:
                mismatches.append(f"{name} — пошкоджене зображення")
                continue

            if (width, height) != (expected_size, expected_size):
                mismatches.append(
                    f"{name} — {width}×{height}, очікується {expected_size}×{expected_size}"
                )

            if "class1" in normalized_name:
                class_counts[1] += 1
            elif "class0" in normalized_name:
                class_counts[0] += 1
            else:
                unlabeled.append(base_name)
    finally:
        archive.close()

    if image_count == 0:
        detail = "В архіві не знайдено зображень PNG, JPG або BMP"
        enqueue_log("model_training_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    if mismatches:
        reported = mismatches[:MAX_REPORTED_MISMATCHES]
        remaining = len(mismatches) - len(reported)
        detail = (
            f"Архів не пройшов перевірку: {len(mismatches)} з {image_count} зображень "
            f"не відповідають розміру {expected_size}×{expected_size}. "
            + "; ".join(reported)
        )
        if remaining > 0:
            detail += f"; та ще {remaining}"

        enqueue_log("model_training_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    if class_counts[0] < 6 or class_counts[1] < 6:
        detail = (
            "Для навчання потрібно щонайменше 6 зображень класу class0 "
            f"та 6 зображень класу class1 (знайдено: class0 — {class_counts[0]}, "
            f"class1 — {class_counts[1]})"
        )
        enqueue_log("model_training_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    return {
        "images": image_count,
        "class0": class_counts[0],
        "class1": class_counts[1],
        "unlabeled": unlabeled
    }


def extract_training_features(
    contents: bytes,
    template: np.ndarray,
    grid_cells: int,
    expected_size: int
):
    features = []
    labels = []
    skipped = 0

    with zipfile.ZipFile(io.BytesIO(contents), "r") as archive:
        for name in archive.namelist():
            if name.endswith("/") or not name.lower().endswith(IMAGE_EXTENSIONS):
                continue

            base_name = name.rsplit("/", 1)[-1]
            normalized_name = base_name.lower()
            if normalized_name == "template.png":
                continue

            if "class1" in normalized_name:
                label = 1
            elif "class0" in normalized_name:
                label = 0
            else:
                skipped += 1
                continue

            try:
                raw_bytes = archive.read(name)
                with Image.open(io.BytesIO(raw_bytes)) as image:
                    if image.size != (expected_size, expected_size):
                        skipped += 1
                        continue
                    sample = np.asarray(image.convert("L"), dtype=np.uint8)
            except Exception:
                skipped += 1
                continue

            features.append(
                compute_grid_intensities(template, sample, int(grid_cells))
            )
            labels.append(label)

    if not features:
        raise ValueError("У датасеті не знайдено валідних зображень з мітками class0/class1")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32).reshape(-1, 1),
        skipped
    )


@app.get("/settings", summary="1. Отримання поточних налаштувань системи")
def get_settings():
    return dict(app.state.settings)


@app.post("/settings", summary="2. Встановлення налаштувань системи")
def set_settings(settings: SettingsModel):
    if settings.img_size % settings.grid_cells != 0:
        raise HTTPException(
            status_code=400,
            detail=f"img_size ({settings.img_size}) має націло ділитися на grid_cells ({settings.grid_cells})"
        )

    previous_settings = dict(app.state.settings)
    app.state.settings = settings.model_dump()

    if app.state.settings != previous_settings:
        app.state.template = None

    feature_settings_changed = (
        previous_settings["img_size"] != app.state.settings["img_size"]
        or previous_settings["grid_cells"] != app.state.settings["grid_cells"]
    )

    if feature_settings_changed:
        clear_model_state(
            "Налаштування зображення або сітки змінено — навчену модель скинуто"
        )

    if settings.random_seed is not None:
        random.seed(settings.random_seed)
        np.random.seed(settings.random_seed % (2 ** 32))

    enqueue_log("settings_updated", "Налаштування системи оновлено")

    return {
        "status": "success",
        "message": "Налаштування успішно застосовано",
        "current_settings": dict(app.state.settings)
    }


@app.get("/training-settings", summary="3. Отримання налаштувань навчання моделі")
def get_training_settings():
    return dict(app.state.training_settings)


@app.post("/training-settings", summary="4. Встановлення налаштувань навчання моделі")
def set_training_settings(settings: TrainingSettingsModel):
    app.state.training_settings = settings.model_dump()
    enqueue_log("training_settings_updated", "Налаштування навчання оновлено")

    return {
        "status": "success",
        "message": "Налаштування навчання успішно застосовано",
        "current_settings": dict(app.state.training_settings)
    }


@app.post("/template", summary="5. Генерація еталонного зображення")
def generate_template_endpoint():
    size = app.state.settings["img_size"]
    template_arr = get_template(size)
    app.state.template = template_arr
    enqueue_log("template_generated", f"Еталонне зображення {size}×{size} згенеровано")
    return template_png_response(template_arr)


@app.get("/template", summary="6. Отримання еталонного зображення")
def get_template_endpoint():
    if app.state.template is None:
        raise HTTPException(status_code=404, detail="Еталонне зображення ще не згенеровано")

    return template_png_response(app.state.template)


@app.get("/logs/stream", summary="7. Потік системних логів (SSE)")
async def stream_logs():
    async def event_generator():
        while True:
            message = await app.state.log_queue.get()
            yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/dataset", summary="8. Генерація датасету (повертає ZIP-архів)")
def generate_dataset_endpoint(req: DatasetRequest):
    enqueue_log(
        "dataset_started",
        f"Розпочато генерацію датасету: {req.total_samples} пар",
        current=0,
        total=req.total_samples
    )

    def report_progress(current: int, total: int):
        enqueue_log(
            "dataset_progress",
            f"Згенеровано {current} з {total} пар",
            current=current,
            total=total
        )

    try:
        batch_results = generate_dataset(req.total_samples, app.state.settings, report_progress)
    except Exception as error:
        enqueue_log("dataset_error", f"Помилка генерації датасету: {error}")
        raise

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for pair in batch_results:
            for filename, png_bytes, _, _ in pair:
                zip_file.writestr(filename, png_bytes)

    zip_buffer.seek(0)

    enqueue_log(
        "dataset_complete",
        f"Датасет згенеровано: {len(batch_results) * 2} зображень",
        current=req.total_samples,
        total=req.total_samples
    )

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=dataset.zip"}
    )


@app.post("/dataset/validate", summary="9. Перевірка розмірів зображень у ZIP-архіві")
def validate_dataset_archive(file: UploadFile = File(...)):
    expected_size = app.state.settings["img_size"]
    contents = file.file.read()

    try:
        archive = zipfile.ZipFile(io.BytesIO(contents))
    except zipfile.BadZipFile:
        detail = "Файл не є коректним ZIP-архівом"
        enqueue_log("dataset_validation_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    image_count = 0
    mismatches = []

    try:
        for name in archive.namelist():
            if name.endswith("/") or not name.lower().endswith(IMAGE_EXTENSIONS):
                continue

            image_count += 1

            try:
                with archive.open(name) as member:
                    raw_bytes = member.read()

                with Image.open(io.BytesIO(raw_bytes)) as image:
                    width, height = image.size
                    image.verify()
            except Exception:
                mismatches.append(f"{name} — пошкоджене зображення")
                continue

            if (width, height) != (expected_size, expected_size):
                mismatches.append(
                    f"{name} — {width}×{height}, очікується {expected_size}×{expected_size}"
                )
    finally:
        archive.close()

    if image_count == 0:
        detail = "В архіві не знайдено зображень PNG, JPG або BMP"
        enqueue_log("dataset_validation_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    if mismatches:
        reported = mismatches[:MAX_REPORTED_MISMATCHES]
        remaining = len(mismatches) - len(reported)
        detail = (
            f"Архів не пройшов перевірку: {len(mismatches)} з {image_count} зображень "
            f"не відповідають розміру {expected_size}×{expected_size}. "
            + "; ".join(reported)
        )
        if remaining > 0:
            detail += f"; та ще {remaining}"

        enqueue_log("dataset_validation_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    enqueue_log(
        "dataset_validated",
        f"Архів перевірено: {image_count} зображень розміру {expected_size}×{expected_size}"
    )

    return {
        "status": "success",
        "images": image_count,
        "img_size": expected_size
    }


@app.post("/analyze", summary="10. Формування різницевої сітки та зображення з накладеною сіткою")
def analyze_image(file: UploadFile = File(...)):
    settings = dict(app.state.settings)
    expected_size = settings["img_size"]
    grid_cells = settings["grid_cells"]
    template = app.state.template

    if template is None:
        detail = "Спочатку згенеруйте еталонне зображення"
        enqueue_log("analysis_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    contents = file.file.read()
    if not contents:
        detail = "Вхідне зображення порожнє"
        enqueue_log("analysis_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    try:
        with Image.open(io.BytesIO(contents)) as image:
            width, height = image.size

            if (width, height) != (expected_size, expected_size):
                detail = (
                    f"Розмір вхідного зображення {width}×{height} не відповідає "
                    f"очікуваному {expected_size}×{expected_size}"
                )
                enqueue_log("analysis_error", detail)
                raise HTTPException(status_code=400, detail=detail)

            sample = np.asarray(image.convert("L"), dtype=np.uint8)
            source_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except HTTPException:
        raise
    except Exception:
        detail = "Не вдалося прочитати вхідне зображення"
        enqueue_log("analysis_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    if template.shape[:2] != sample.shape[:2]:
        detail = "Розмір еталонного зображення не збігається з поточними налаштуваннями"
        enqueue_log("analysis_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    intensities = compute_grid_intensities(template, sample, int(grid_cells))
    difference_grid_png = render_difference_grid(intensities, int(grid_cells), int(expected_size))
    source_grid_png = render_source_with_grid(source_rgb, int(grid_cells))

    model = app.state.model
    model_metadata = app.state.model_metadata
    prediction = None
    prediction_message = None

    if model is None:
        prediction_message = "Модель ще не навчена — показано лише різницеву сітку"
    elif (
        model_metadata is None
        or model_metadata.get("img_size") != expected_size
        or model_metadata.get("grid_cells") != grid_cells
        or model_metadata.get("feature_count") != int(grid_cells) * int(grid_cells)
    ):
        prediction_message = (
            "Модель навчена для інших налаштувань зображення або сітки — "
            "потрібно повторне навчання"
        )
    else:
        try:
            from model_trainer import predict_sample

            threshold = float(model_metadata.get("threshold", MODEL_DECISION_THRESHOLD))
            probability = float(predict_sample(model, intensities))
            has_defect = probability >= threshold
            confidence = max(probability, 1.0 - probability)
            prediction = {
                "has_defect": bool(has_defect),
                "label": "defect" if has_defect else "clean",
                "label_uk": "Дефект" if has_defect else "Без дефекту",
                "probability": round(probability, 6),
                "confidence": round(float(confidence), 6),
                "threshold": threshold
            }
        except Exception as error:
            prediction_message = "Не вдалося отримати прогноз моделі"
            enqueue_log(
                "analysis_prediction_error",
                f"Помилка прогнозу моделі: {error}"
            )

    if prediction is not None:
        prediction_text = (
            f"Модель: {prediction['label_uk'].lower()}, "
            f"ймовірність дефекту {prediction['probability'] * 100:.1f}%"
        )
    else:
        prediction_text = prediction_message or "Модель ще не навчена"

    enqueue_log(
        "analysis_complete",
        f"Різницеву сітку {grid_cells}×{grid_cells} та зображення із сіткою сформовано. "
        f"{prediction_text}"
    )

    return {
        "grid_cells": grid_cells,
        "img_size": expected_size,
        "difference_grid": base64.b64encode(difference_grid_png).decode("ascii"),
        "source_with_grid": base64.b64encode(source_grid_png).decode("ascii"),
        "model_ready": model is not None and prediction_message is None,
        "prediction": prediction,
        "prediction_message": prediction_message
    }


@app.post("/model/train", summary="11. Навчання моделі на завантаженому датасеті")
def train_model_endpoint(file: UploadFile = File(...)):
    with app.state.model_lock:
        if app.state.model_training:
            raise HTTPException(status_code=409, detail="Навчання моделі вже виконується")
        app.state.model_training = True
        app.state.model_training_progress = None

    started_at = time.perf_counter()

    try:
        settings = dict(app.state.settings)
        training_settings = dict(app.state.training_settings)
        template = app.state.template
        expected_size = settings["img_size"]
        grid_cells = settings["grid_cells"]
        epochs = int(training_settings["epochs"])
        learning_rate = float(training_settings["learning_rate"])
        hidden_layers = int(training_settings["hidden_layers"])
        hidden_units = int(training_settings["hidden_units"])
        random_state = (
            int(settings["random_seed"])
            if settings["random_seed"] is not None
            else 42
        )

        if template is None:
            detail = "Спочатку згенеруйте еталонне зображення"
            enqueue_log("model_training_error", detail)
            raise HTTPException(status_code=400, detail=detail)

        contents = file.file.read()
        if not contents:
            detail = "Файл датасету порожній"
            enqueue_log("model_training_error", detail)
            raise HTTPException(status_code=400, detail=detail)

        archive_info = validate_training_archive(contents, expected_size)
        if archive_info["unlabeled"]:
            enqueue_log(
                "model_training_warning",
                f"Пропущено зображень без мітки class0/class1: "
                f"{len(archive_info['unlabeled'])}"
            )

        hidden_description = (
            f"1 прихований шар ({hidden_units} нейронів)"
            if hidden_layers == 1
            else "без прихованого шару"
        )
        enqueue_log(
            "model_training_started",
            f"Розпочато навчання моделі: {archive_info['class0']} class0, "
            f"{archive_info['class1']} class1, {epochs} епох, {hidden_description}",
            current=0,
            total=epochs
        )

        try:
            from model_trainer import train_defect_model
        except Exception as error:
            detail = f"Не вдалося завантажити модуль навчання: {error}"
            enqueue_log("model_training_error", detail)
            raise HTTPException(status_code=500, detail=detail)

        enqueue_log(
            "model_training_features",
            f"Обчислення ознак різницевої сітки {grid_cells}×{grid_cells}…",
            current=0,
            total=epochs
        )

        features, labels, skipped_features = extract_training_features(
            contents,
            template,
            grid_cells,
            expected_size
        )

        if skipped_features:
            enqueue_log(
                "model_training_warning",
                f"Під час обчислення ознак пропущено зображень: {skipped_features}"
            )

        def report_progress(epoch: int, train_accuracy: float, test_accuracy: float):
            progress = {
                "current": epoch,
                "total": epochs,
                "train_accuracy": round(float(train_accuracy), 2),
                "test_accuracy": round(float(test_accuracy), 2)
            }
            app.state.model_training_progress = progress
            enqueue_log(
                "model_training_progress",
                f"Епоха {epoch}/{epochs}: точність train {train_accuracy:.2f}%, "
                f"test {test_accuracy:.2f}%",
                current=epoch,
                total=epochs,
                train_accuracy=round(float(train_accuracy), 2),
                test_accuracy=round(float(test_accuracy), 2)
            )

        model, metrics = train_defect_model(
            features,
            labels,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            random_state=random_state,
            progress_callback=report_progress
        )

        trained_at = datetime.now().astimezone().isoformat()
        duration_seconds = round(time.perf_counter() - started_at, 3)
        feature_count = int(features.shape[1])

        calculated_threshold = float(metrics.get("threshold", MODEL_DECISION_THRESHOLD))

        architecture = (
            f"dense_{hidden_units}_sigmoid" if hidden_layers == 1 else "linear_sigmoid"
        )
        metadata = {
            "format_version": 1,
            "framework": "tensorflow",
            "architecture": architecture,
            "img_size": expected_size,
            "grid_cells": grid_cells,
            "feature_count": feature_count,
            "hidden_layers": hidden_layers,
            "hidden_units": hidden_units,
            "hidden_activation": "sigmoid",
            "optimizer": "adam",
            "threshold": calculated_threshold,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "trained_at": trained_at,
            "duration_seconds": duration_seconds,
            "source": "trained",
            "dataset_filename": file.filename
        }
        metrics = {
            **metrics,
            "feature_count": feature_count,
            "duration_seconds": duration_seconds,
            "trained_at": trained_at
        }

        with app.state.model_lock:
            current_settings = dict(app.state.settings)
            if (
                current_settings["img_size"] != expected_size
                or current_settings["grid_cells"] != grid_cells
            ):
                detail = (
                    "Налаштування зображення або сітки змінилися під час навчання. "
                    "Модель не збережено, запустіть навчання повторно."
                )
                enqueue_log("model_training_error", detail)
                raise HTTPException(status_code=409, detail=detail)

            app.state.model = model
            app.state.model_metadata = metadata
            app.state.model_metrics = metrics
            app.state.model_training_progress = {
                "current": epochs,
                "total": epochs,
                "train_accuracy": metrics["train_accuracy"],
                "test_accuracy": metrics["test_accuracy"]
            }

        enqueue_log(
            "model_training_complete",
            f"Модель навчено: test accuracy {metrics['test_accuracy']}%, "
            f"F1 {metrics['test_f1']}%, threshold {calculated_threshold:.2f}, "
            f"час {duration_seconds:.2f} с",
            current=epochs,
            total=epochs,
            metrics=metrics,
            model=metadata,
            threshold=MODEL_DECISION_THRESHOLD
        )

        return {
            "status": "success",
            "message": "Модель успішно навчено та збережено",
            "model": metadata,
            "metrics": metrics
        }
    except HTTPException:
        raise
    except Exception as error:
        detail = f"Помилка навчання моделі: {error}"
        enqueue_log("model_training_error", detail)
        raise HTTPException(status_code=500, detail=detail)
    finally:
        with app.state.model_lock:
            app.state.model_training = False
            app.state.model_training_progress = None


@app.get("/model/status", summary="12. Перевірка готовності навченої моделі")
def get_model_status():
    return build_model_status()


def _keras_shape_tuple(shape) -> tuple:
    """Нормалізує input_shape/output_shape Keras до одного кортежу."""
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    if shape is None:
        return ()
    return tuple(shape)


def _format_compatibility_error(metadata: dict, settings: dict) -> str:
    model_img_size = metadata.get("img_size", "невідомо")
    model_grid_cells = metadata.get("grid_cells", "невідомо")
    model_feature_count = metadata.get("feature_count", "невідомо")
    current_img_size = settings["img_size"]
    current_grid_cells = settings["grid_cells"]
    current_feature_count = int(current_grid_cells) * int(current_grid_cells)

    return (
        "Модель несумісна з поточними налаштуваннями:\n"
        f"модель — img_size={model_img_size}, grid_cells={model_grid_cells} "
        f"({model_feature_count} ознак);\n"
        f"поточні — img_size={current_img_size}, grid_cells={current_grid_cells} "
        f"({current_feature_count} ознак).\n"
        "Змініть налаштування або завантажте іншу модель."
    )


@app.get("/model/download", summary="13. Експорт поточної моделі у ZIP-бандл")
def download_model():
    with app.state.model_lock:
        model = app.state.model
        metadata = app.state.model_metadata
        metrics = app.state.model_metrics

    if model is None or metadata is None:
        raise HTTPException(status_code=404, detail="Модель ще не навчена або не завантажена")

    try:
        from model_trainer import export_model_bundle
    except Exception as error:
        detail = f"Не вдалося завантажити модуль експорту моделі: {error}"
        enqueue_log("model_download_error", detail)
        raise HTTPException(status_code=500, detail=detail)

    try:
        contents = export_model_bundle(model, metadata, metrics or {})
    except Exception as error:
        detail = f"Не вдалося експортувати модель: {error}"
        enqueue_log("model_download_error", detail)
        raise HTTPException(status_code=500, detail=detail)

    return Response(
        content=contents,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=defect-model.zip"},
    )


@app.post("/model/load", summary="14. Імпорт моделі з ZIP-бандла")
def load_model_endpoint(file: UploadFile = File(...)):
    with app.state.model_lock:
        if app.state.model_training:
            raise HTTPException(status_code=409, detail="Навчання моделі вже виконується")

    contents = file.file.read()
    if not contents:
        detail = "Файл моделі порожній"
        enqueue_log("model_load_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    try:
        from model_trainer import load_model_bundle
    except Exception as error:
        detail = f"Не вдалося завантажити модуль імпорту моделі: {error}"
        enqueue_log("model_load_error", detail)
        raise HTTPException(status_code=500, detail=detail)

    try:
        model, metadata, metrics = load_model_bundle(contents)
    except ValueError as error:
        detail = f"Не вдалося завантажити модель: {error}"
        enqueue_log("model_load_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    if metadata.get("format_version") != 1:
        detail = (
            "Непідтримувана версія формату моделі: "
            f"{metadata.get('format_version', 'невідомо')}. Очікується 1."
        )
        enqueue_log("model_load_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    if metadata.get("framework") != "tensorflow":
        detail = (
            "Модель має непідтримуваний фреймворк: "
            f"{metadata.get('framework', 'невідомо')}. Підтримується лише tensorflow."
        )
        enqueue_log("model_load_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    settings = dict(app.state.settings)
    current_grid_cells = int(settings["grid_cells"])
    current_feature_count = current_grid_cells * current_grid_cells

    if (
        metadata.get("img_size") != settings["img_size"]
        or metadata.get("grid_cells") != settings["grid_cells"]
        or metadata.get("feature_count") != current_feature_count
    ):
        detail = _format_compatibility_error(metadata, settings)
        enqueue_log("model_load_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    input_shape = _keras_shape_tuple(model.input_shape)
    output_shape = _keras_shape_tuple(model.output_shape)
    expected_input = (None, current_feature_count)
    expected_output = (None, 1)
    if input_shape != expected_input or output_shape != expected_output:
        detail = (
            "Модель має неочікувану форму входу/виходу: "
            f"вхід {input_shape}, вихід {output_shape}; "
            f"очікується вхід {expected_input}, вихід {expected_output}."
        )
        enqueue_log("model_load_error", detail)
        raise HTTPException(status_code=400, detail=detail)

    with app.state.model_lock:
        if app.state.model_training:
            raise HTTPException(status_code=409, detail="Навчання моделі вже виконується")

        current_settings = dict(app.state.settings)
        if (
            current_settings["img_size"] != settings["img_size"]
            or current_settings["grid_cells"] != settings["grid_cells"]
        ):
            detail = (
                "Налаштування зображення або сітки змінилися під час завантаження моделі. "
                "Завантажте модель повторно."
            )
            enqueue_log("model_load_error", detail)
            raise HTTPException(status_code=409, detail=detail)

        metadata["source"] = "loaded"
        app.state.model = model
        app.state.model_metadata = metadata
        app.state.model_metrics = metrics
        app.state.model_training_progress = None

    enqueue_log(
        "model_loaded",
        f"Модель завантажено: img_size={metadata.get('img_size')}, "
        f"grid_cells={metadata.get('grid_cells')} "
        f"({metadata.get('feature_count')} ознак), "
        f"threshold {float(metadata.get('threshold', MODEL_DECISION_THRESHOLD)):.2f}",
        model=metadata,
        metrics=metrics,
    )

    return {
        "status": "success",
        "message": "Модель успішно завантажено",
        "model": metadata,
        "metrics": metrics,
    }
