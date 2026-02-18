from typing import Sequence

import numpy as np
import cv2
from PIL import Image


def warp_affine_expand(
    img,
    M,
    border_value: Sequence[float] | float=0,
    flags=cv2.INTER_NEAREST,
):
    h, w = img.shape[:2]

    # 4 угла изображения
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    ones = np.ones((4, 1), dtype=np.float32)
    corners_h = np.hstack([corners, ones])

    # преобразованные углы
    tc = (M @ corners_h.T).T

    min_xy = tc.min(axis=0)
    max_xy = tc.max(axis=0)

    out_w = int(np.ceil(max_xy[0] - min_xy[0]))
    out_h = int(np.ceil(max_xy[1] - min_xy[1]))

    # сдвигаем матрицу, чтобы всё было внутри
    M2 = M.copy()
    M2[0, 2] -= min_xy[0]
    M2[1, 2] -= min_xy[1]

    rotated = cv2.warpAffine(
        img,
        M2,
        (out_w, out_h),
        flags=flags,
        borderValue=border_value
    )

    return rotated


def warp_belly_to_rect(
    image_np: np.ndarray,
    target_size: tuple[int, int],
) -> Image.Image | None:
    """
    Выполняет нелинейное выпрямление брюшка:
    по каждой горизонтальной строке растягивает область между левым и правым краем маски
    в фиксированную ширину target_size.
    """

    # 1. Получаем бинарную маску из RGB-изображения:
    # считаем пиксель "брюшком", если он не полностью чёрный.
    mask_bool = image_np.sum(axis=2) > 0
    if not mask_bool.any():
        return None

    # 2. Вычисляем главную ось объекта через PCA
    # Нужно, чтобы повернуть брюшко вертикально.

    # Получаем координаты всех пикселей маски
    ys, xs = np.nonzero(mask_bool)

    # Формируем массив точек (x, y)
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))

    # Центрируем точки (для корректной ковариации)
    mean = coords.mean(axis=0)
    centered = coords - mean

    # Ковариационная матрица
    cov = np.cov(centered, rowvar=False)

    # Собственные значения и векторы
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Главная ось — собственный вектор с максимальным собственным значением
    major = eigvecs[:, 1]

    # Угол наклона главной оси
    angle_rad = np.arctan2(major[1], major[0])
    rot_deg = np.degrees(angle_rad) - 90

    # 3. Поворот изображения и маски так, чтобы брюшко стало вертикальным
    h, w = mask_bool.shape
    cy = float(ys.mean())
    cx = float(xs.mean())
    center = (cx, cy)
    rot_mat = cv2.getRotationMatrix2D(center, rot_deg, 1.0)

    # Поворачиваем изображение
    rot_image = warp_affine_expand(
        image_np,
        M=rot_mat,
        flags=cv2.INTER_LINEAR,
        border_value=(0, 0, 0),
    )

    # Поворачиваем маску
    rot_mask = warp_affine_expand(
        (mask_bool.astype(np.uint8) * 255),
        M=rot_mat,
        flags=cv2.INTER_NEAREST,
        border_value=0,
    ) > 0

    # 4. Проверяем ориентацию и
    # доворачиваем на 90° в случае необходимости

    ys_r, xs_r = np.nonzero(rot_mask)
    if len(ys_r) == 0:
        return None

    h_rot, w_rot = rot_mask.shape

    # 5. Для каждой строки ищем левую и правую границу маски

    # Минимальный и максимальный X для каждой строки
    x_min = np.full(h_rot, np.inf)
    x_max = np.full(h_rot, -np.inf)

    ys_nonzero, xs_nonzero = np.nonzero(rot_mask)
    if len(ys_nonzero) == 0:
        return None

    for y, x in zip(ys_nonzero, xs_nonzero):
        if x < x_min[y]:
            x_min[y] = x
        if x > x_max[y]:
            x_max[y] = x

    # Строки, где вообще есть брюшко
    valid_rows = np.where(x_max >= 0)[0]
    if len(valid_rows) == 0:
        return None

    y_start, y_end = valid_rows.min(), valid_rows.max()

    # 6. Подготавливаем выходное изображение фиксированного размера

    target_width, target_height = target_size
    out = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Строим равномерное соответствие между строками выходного изображения
    # (целевое количество строк — target_height)
    # и строками исходного изображения в диапазоне от y_start до y_end]
    src_y_f = np.linspace(y_start, y_end, target_height)

    # 7. Основной этап растяжения
    for yi, sy in enumerate(src_y_f):
        sy_idx = int(round(sy))
        sy_idx = max(0, min(sy_idx, h_rot - 1))

        # Если строка пустая — берем ближайшую непустую
        if x_max[sy_idx] < 0:
            nearest = valid_rows[np.abs(valid_rows - sy_idx).argmin()]
            sy_idx = int(nearest)

        # Берём строку исходного изображения
        row = rot_image[sy_idx]

        row_w = row.shape[0] # ширина строки после всех поворотов

        # Левая и правая границы брюшка в этой строке
        x0 = int(x_min[sy_idx])
        x1 = int(x_max[sy_idx])

        x0 = max(0, min(x0, row_w - 1))
        x1 = max(0, min(x1, row_w - 1))

        if x1 <= x0:
            continue

        # Создаём равномерную сетку по ширине
        xs_src = np.linspace(x0, x1, target_width)

        # 8. Интерполяция каждого цветового канала

        for c in range(3):
            out[yi, :, c] = np.interp(
                xs_src,
                np.arange(row_w),
                row[:, c]
            ).astype(np.uint8)

    return Image.fromarray(out)


def extract_belly_from_prediction(
    original: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int] = (80, 320),
    auto_rotate: bool = True,
    warp: bool = True,
) -> Image.Image:

    # 1. Приведение маски к бинарному виду
    mask_gray = mask.astype(np.uint8)
    mask_bool = mask_gray > 0

    if not mask_bool.any():
        raise ValueError("Mask is empty — no belly detected")

    # 2. Поиск bounding-box по маске
    ys, xs = np.nonzero(mask_bool)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    # 3. Обнуление фона вне маски
    masked = np.zeros_like(original)
    masked[mask_bool] = original[mask_bool]
    crop = masked[y0:y1, x0:x1]

    belly_img = Image.fromarray(crop)

    # 4. Варпинг по медиальной оси
    if warp:
        warped = warp_belly_to_rect(np.array(belly_img), target_size)
        if warped is not None:
            return warped

    # 5. Fallback
    return belly_img.resize(target_size, Image.BILINEAR)
