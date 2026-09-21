# ============================================================
# check_dataset.py - 划分前检查 YOLO 标签
# 检查图片与标签是否配对、类别 ID 是否越界、归一化框是否超出图片,
# 并统计每类框数和空背景图片数量。
# 通过只说明结构和数值合法, 不说明框画得正确, 仍需人工抽图复核。
# 用法: 修改下方调参区, 然后直接运行  python check_dataset.py
# ============================================================

import math
from collections import Counter
from pathlib import Path

# ==================== 调参区 ====================
IMAGES_DIR   = Path("converted/images")  # 图片目录
LABELS_DIR   = Path("converted/labels")  # 标签目录 (与图片同名的 .txt)
CLASSES_FILE = Path("classes.txt")       # 类别清单 (一行一类, 行号从 0 开始)
MAX_ERRORS   = 20                        # 最多打印多少条错误, 剩余只报数量
# ================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names(path):
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError("classes file is empty")
    return names


def check_label(path, class_count, counts):
    errors = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{path}:{line_number}: expected 5 fields")
            continue
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric label value")
            continue
        values = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{path}:{line_number}: non-finite box value")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"{path}:{line_number}: class ID {class_id} is outside 0..{class_count - 1}")
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{path}:{line_number}: normalized values are outside YOLO bounds")
        # Center and size can each be legal while their combined box crosses an edge.
        if x_center - width / 2 < -1e-6 or x_center + width / 2 > 1 + 1e-6:
            errors.append(f"{path}:{line_number}: box crosses the left or right image edge")
        if y_center - height / 2 < -1e-6 or y_center + height / 2 > 1 + 1e-6:
            errors.append(f"{path}:{line_number}: box crosses the top or bottom image edge")
        counts[class_id] += 1
    return errors


def main():
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir() or not CLASSES_FILE.is_file():
        raise FileNotFoundError("IMAGES_DIR, LABELS_DIR, and CLASSES_FILE paths must exist")
    if MAX_ERRORS < 1:
        raise ValueError("MAX_ERRORS must be positive")

    class_names = load_class_names(CLASSES_FILE)
    image_stems = {
        path.stem for path in IMAGES_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    label_paths = {path.stem: path for path in LABELS_DIR.glob("*.txt") if path.is_file()}
    errors = []
    for stem in sorted(image_stems - label_paths.keys()):
        errors.append(f"missing label for image stem: {stem}")
    for stem in sorted(label_paths.keys() - image_stems):
        errors.append(f"label has no matching image: {label_paths[stem]}")

    counts = Counter()
    background_count = 0
    for stem in sorted(image_stems & label_paths.keys()):
        label_path = label_paths[stem]
        if not label_path.read_text(encoding="utf-8").strip():
            background_count += 1
        errors.extend(check_label(label_path, len(class_names), counts))

    if errors:
        for error in errors[:MAX_ERRORS]:
            print(f"ERROR: {error}")
        if len(errors) > MAX_ERRORS:
            print(f"ERROR: {len(errors) - MAX_ERRORS} more errors omitted")
        raise SystemExit(1)

    print(f"images={len(image_stems)} background_images={background_count} boxes={sum(counts.values())}")
    for class_id, name in enumerate(class_names):
        print(f"class_id={class_id} name={name} boxes={counts[class_id]}")


if __name__ == "__main__":
    main()
