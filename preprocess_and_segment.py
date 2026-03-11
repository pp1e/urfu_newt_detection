from typing import Tuple

import cv2
import numpy as np
import os
# from skimage import exposure

# Папка с исходными фото и папка для результатов
INPUT_DIR = "data/karelin_newt_small"
OUTPUT_DIR = "data/preprocessed_data"

# ---- ПАРАМЕТРЫ (можно подкручивать) ----
MAX_SIDE = 1400               # ресайз для стабильности
MIN_BODY_AREA_FRAC = 0.01     # минимальная доля площади кадра для «тела»
MORPH_K = 5                   # размер ядра морфологии
ADAPT_BLOCK_FRAC = 0.03       # относительный размер окна адаптивного порога (5% от меньшей стороны)
ADAPT_C = 5                   # смещение адаптивного порога
USE_HOUGH_DISH = True         # пытаться находить круг чашки Петри

def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1

def resize_keep_aspect(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def save_step(steps_dir, name, img):
    p = os.path.join(steps_dir, f"{name}.jpg")
    cv2.imwrite(p, img)

def lab_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    lab = cv2.merge([Lc, A, B])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def find_dish_mask(img_gray):
    # Попытка найти круг чашки Петри (возвращает маску или None)
    h, w = img_gray.shape
    blur = cv2.medianBlur(img_gray, 7)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(h, w)//2,
        param1=120, param2=50,
        minRadius=min(h, w)//4, maxRadius=min(h, w)//2
    )
    mask = np.zeros_like(img_gray)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # берём самый большой круг
        c = max(circles[0, :], key=lambda x: x[2])
        radius = int(c[2] * 0.9)
        cv2.circle(mask, (c[0], c[1]), radius, 255, thickness=-1)
        return mask
    return None

def threshold_combo(gray, roi_mask):
    # Две бинаризации: Otsu и Adaptive. Выберем ту, где «тело» получается адекватной площади.
    g = gray.copy()
    if roi_mask is not None:
        g = cv2.bitwise_and(g, g, mask=roi_mask)

    # Otsu (инвертированный и прямой варианты)
    _, th_otsu_inv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _, th_otsu     = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Adaptive (подбираем окно от размера кадра)
    h, w = g.shape
    block = ensure_odd(int(min(h, w) * ADAPT_BLOCK_FRAC))
    block = max(31, block)  # не слишком маленькое
    th_adapt_inv = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, block, ADAPT_C)
    th_adapt     = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, block, ADAPT_C)

    return {"otsu_inv": th_otsu_inv, "otsu": th_otsu, "adapt_inv": th_adapt_inv, "adapt": th_adapt}

def largest_body_mask(bin_img, image_shape, roi_mask=None) -> Tuple[np.ndarray, dict]:
    """Выберем «тело» по контурам с фильтрацией: площадь, центральность, компактность."""
    h, w = image_shape[:2]
    s = h * w
    B = bin_img.copy()
    if roi_mask is not None:
        B = cv2.bitwise_and(B, B, mask=roi_mask)

    # Морфология
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    B = cv2.morphologyEx(B, cv2.MORPH_OPEN, k, iterations=1)
    B = cv2.morphologyEx(B, cv2.MORPH_CLOSE, k, iterations=3)

    contours, _ = cv2.findContours(B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), np.uint8), {"area": 0, "cx": None, "cy": None}

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < s * MIN_BODY_AREA_FRAC:
            continue
        x,y,wc,hc = cv2.boundingRect(cnt)
        aspect_ratio = max(wc, hc) / min(wc, hc)
        if aspect_ratio < 1.5:
            continue
        # центральность (центр масс ближе к центру кадра)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        center_dist = np.hypot(cx - w//2, cy - h//2) / (np.hypot(w, h))

        # компактность (solidity) — тело более «плотное»
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) or 1
        solidity = float(area) / hull_area

        score = area * (1 - center_dist) * (0.5 + 0.5*solidity)
        candidates.append((score, cnt, area, (cx, cy)))

    if not candidates:
        return np.zeros((h, w), np.uint8), {"area": 0, "cx": None, "cy": None}

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_cnt, area, (cx, cy) = candidates[0]

    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [best_cnt], -1, 255, thickness=-1)
    return mask, {"area": area, "cx": cx, "cy": cy}

def preprocess_and_segment(image_path: str):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    steps_dir = os.path.join(OUTPUT_DIR, f"{basename}_steps")
    os.makedirs(steps_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось открыть {image_path}")
        return
    img = resize_keep_aspect(img)
    save_step(steps_dir, "00_input", img)

    # 1) Денойз + CLAHE по L-каналу (Lab)
    den = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    save_step(steps_dir, "01_denoise", den)
    eq = lab_clahe(den)
    save_step(steps_dir, "02_lab_clahe", eq)

    # 2) Серый + блюр
    gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    save_step(steps_dir, "03_gray_blur", gray)

    # 3) Маска чашки Петри (если найдём)
    roi_mask = None
    if USE_HOUGH_DISH:
        roi_mask = find_dish_mask(gray)
        if roi_mask is not None:
            save_step(steps_dir, "04_dish_mask", roi_mask)

    # 4) Несколько вариантов порога
    th_dict = threshold_combo(gray, roi_mask)
    for k, v in th_dict.items():
        save_step(steps_dir, f"05_thresh_{k}", v)

    # 5) Выбираем лучший порог по «качеству» маски
    best = None
    best_score = -1
    best_info = {}
    for name, th in th_dict.items():
        mask, info = largest_body_mask(th, img.shape, roi_mask)
        score = info["area"]
        if score > best_score:
            best_score = score
            best = (name, th, mask, info)
    best_name, best_th, body_mask, info = best
    save_step(steps_dir, f"06_best_bin_{best_name}", best_th)
    save_step(steps_dir, "07_body_mask", body_mask)

    # 6) Применяем маску к исходнику
    segmented = cv2.bitwise_and(img, img, mask=body_mask)
    save_step(steps_dir, "08_segmented", segmented)

    # 7) Итоговые файлы (сводное изображение)
    out_mask = os.path.join(OUTPUT_DIR, f"{basename}_mask.jpg")
    out_seg  = os.path.join(OUTPUT_DIR, f"{basename}_segmented.jpg")
    cv2.imwrite(out_seg, segmented)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{basename}_mask.jpg"), body_mask)

    print(f"✅ {basename}: выбран порог = {best_name}, площадь тела = {info['area']}")
    return out_seg

if __name__ == "__main__":
    for fn in os.listdir(INPUT_DIR):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            preprocess_and_segment(os.path.join(INPUT_DIR, fn))
