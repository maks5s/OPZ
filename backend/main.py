import os
import random
import json
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

# --- ПАРАМЕТРИ ЗОБРАЖЕННЯ ТА СІТКИ ---
IMG_SIZE = 200  # 200x200 пікселів
GRID_CELLS = 10  # сітка 10x10 (клітинка 20x20 пікселів)
CENTER = (IMG_SIZE // 2, IMG_SIZE // 2)
RADIUS = 70  # радіус круга

# Створення базової бінарної маски круга (True = всередині круга)
Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
DIST_FROM_CENTER = np.sqrt((X - CENTER[0]) ** 2 + (Y - CENTER[1]) ** 2)
CIRCLE_MASK = DIST_FROM_CENTER <= RADIUS
BACKGROUND_MASK = ~CIRCLE_MASK


def get_template():
    """Створює чистий еталонний масив: білий фон (255), чорний круг (0)."""
    img = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)
    img[CIRCLE_MASK] = 0
    return img


def add_random_spots(img_array, mask, count, radius_range, value):
    """Додає плями в межах заданої маски."""
    valid_coords = np.argwhere(mask)
    if len(valid_coords) == 0:
        return
    for _ in range(count):
        cy, cx = valid_coords[random.randint(0, len(valid_coords) - 1)]
        r = random.randint(*radius_range)
        y_low, y_high = max(0, cy - r), min(IMG_SIZE, cy + r + 1)
        x_low, x_high = max(0, cx - r), min(IMG_SIZE, cx + r + 1)

        y_grid, x_grid = np.ogrid[y_low:y_high, x_low:x_high]
        spot = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= r ** 2
        sub_mask = mask[y_low:y_high, x_low:x_high] & spot
        img_array[y_low:y_high, x_low:x_high][sub_mask] = value


def apply_background_noise(img_array, mode):
    """Накладає фоновий шум згідно з правилами 1-8."""
    bg_coords = np.argwhere(BACKGROUND_MASK)
    n_bg = len(bg_coords)

    if mode == 0:
        pass  # чисто білий фон
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
        # Плями чорні (радіус 2-3 пікселі, сумарно ~10-30 пікселів у плямі)
        add_random_spots(img_array, BACKGROUND_MASK, count=random.randint(4, 20), radius_range=(1, 4), value=0)
    elif mode == 6:
        # Плями темніші (245)
        add_random_spots(img_array, BACKGROUND_MASK, count=random.randint(4, 20), radius_range=(1, 4), value=200)
    elif mode == 7:
        # Всі пікселі фону 245
        img_array[BACKGROUND_MASK] = 200


def apply_defect(img_array, mode):
    """Накладає дефекти всередині фігури згідно з правилами 1-7."""
    circle_coords = np.argwhere(CIRCLE_MASK)
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
        add_random_spots(img_array, CIRCLE_MASK, count=random.randint(4, 20), radius_range=(1, 4), value=255)
    elif mode == 6:
        add_random_spots(img_array, CIRCLE_MASK, count=random.randint(4, 20), radius_range=(1, 4), value=45)
    elif mode == 7:
        # Велика дірка площею 20-40% площі кола
        # Площа круга S = pi*R^2. Радіус дірки r = R * sqrt(fraction)
        fraction = random.uniform(0.20, 0.40)
        hole_r = int(RADIUS * np.sqrt(fraction))

        # Центр дірки генерується так, щоб вона була в межах кола
        max_offset = max(1, RADIUS - hole_r)
        angle = random.uniform(0, 2 * np.pi)
        offset = random.uniform(0, max_offset)
        cy = int(CENTER[1] + offset * np.sin(angle))
        cx = int(CENTER[0] + offset * np.cos(angle))

        y_grid, x_grid = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        hole_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= hole_r ** 2
        img_array[CIRCLE_MASK & hole_mask] = 255


def compute_grid_features(template, sample):
    """Обчислює вектор різниці за клітинками (спосіб 2 зі статті)."""
    # Зображення різниці: значення від 0 до 255
    # diff = np.abs(template.astype(np.int16) - sample.astype(np.int16)).astype(np.float32)
    # diff = diff / 255.0  # нормалізація до [0, 1]
    diff = template.astype(np.float32) - sample.astype(np.float32)

    cell_size = IMG_SIZE // GRID_CELLS
    features = []

    for i in range(GRID_CELLS):
        for j in range(GRID_CELLS):
            cell = diff[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            features.append(float(np.mean(cell)))

    return features


def generate_pair_sample(args):
    sample_id, bg_mode, defect_mode, output_dir = args
    template = get_template()
    sample = template.copy()

    # 1. Накладаємо фоновий шум
    apply_background_noise(sample, bg_mode)
    def_sample = sample.copy()
    apply_defect(def_sample, defect_mode)

    # 3. Витягуємо вектор інтенсивностей клітинок
    features = compute_grid_features(template, sample)

    # 4. Зберігаємо зображення на диск
    filename = f"{sample_id:05d}_class0_bg{bg_mode}_def0.png"
    filepath = os.path.join(output_dir, filename)
    Image.fromarray(sample).save(filepath)

    filename = f"{sample_id:05d}_class1_bg{bg_mode}_def{defect_mode}.png"
    filepath = os.path.join(output_dir, filename)
    Image.fromarray(def_sample).save(filepath)

    return {
        "file": filename,
        "input": features,
        "output": [1]
    }


def main():
    TOTAL_SAMPLES = 100  # Вказуйте будь-яку бажану кількість зразків
    OUTPUT_DIR = "./dataset"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Збереження чистого еталона
    template = get_template()
    Image.fromarray(template).save(os.path.join(OUTPUT_DIR, "template.png"))

    # Формування завдань для процесів (збалансовано 50% норма, 50% дефект)
    # tasks = []
    # for i in range(TOTAL_SAMPLES):
    #     is_defective = (i % 2 == 1)
    #     bg_mode = random.randint(1, 8)
    #     defect_mode = random.randint(1, 7) if is_defective else 0
    #     tasks.append((i, is_defective, bg_mode, defect_mode, OUTPUT_DIR))

    tasks = []
    sample_id = 0

    for bg_mode in range(0, 8):
        for defect_mode in range(1, 8):
            tasks.append((sample_id, bg_mode, defect_mode, OUTPUT_DIR))
            sample_id += 1

    while len(tasks) < TOTAL_SAMPLES:
        bg_mode = random.randint(0, 8)
        defect_mode = random.randint(1, 7)

        tasks.append((sample_id, bg_mode, defect_mode, OUTPUT_DIR))
        sample_id += 1

    tasks = tasks[:TOTAL_SAMPLES]

    print(f"Генерація {TOTAL_SAMPLES} зображень на {cpu_count()} ядрах CPU...")
    with Pool(processes=int(cpu_count()/2)) as pool:
        dataset_records = pool.map(generate_pair_sample, tasks)

    # # Збереження готового датасету для Brain.js
    # with open("dataset_for_brain.json", "w") as f:
    #     json.dump(dataset_records, f, indent=2)

    print("Генерацію завершено! Файли збережено у './dataset', розмітку у 'dataset_for_brain.json'.")


if __name__ == "__main__":
    main()