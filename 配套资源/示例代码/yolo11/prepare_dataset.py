# ============================================================
# prepare_dataset.py - 按固定随机种子划分 train/val/test
# 把图片-标签对复制到 dataset/images/{train,val,test} 和
# dataset/labels/{train,val,test}; 不移动、不修改源文件。
# 它按"图片"随机划分, 不认识视频来源: 连续视频帧先按前缀
# 手工分组后再用, 否则近重复帧会泄漏进不同子集。
# 用法: 修改下方调参区, 然后直接运行  python prepare_dataset.py
# ============================================================

import random
import shutil
from pathlib import Path

# ==================== 调参区 ====================
IMAGES_DIR  = Path("converted/images")  # 图片目录 ( voc_to_yolo.py 的输出 )
LABELS_DIR  = Path("converted/labels")  # 标签目录 ( 与图片同名的 .txt )
OUTPUT_DIR  = Path("dataset")           # 输出目录 ( 须为空; data.yaml 里的 path 指向它 )
TRAIN_RATIO = 0.8                       # 训练集比例
VAL_RATIO   = 0.1                       # 验证集比例
TEST_RATIO  = 0.1                       # 测试集比例 ( 三项之和必须等于 1 )
SEED        = 42                        # 随机种子; 同一种子得到同一划分, 便于复现
# ================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_pairs(images_dir, labels_dir):
    pairs = []
    missing = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            pairs.append((image_path, label_path))
        else:
            missing.append(image_path.name)
    if missing:
        raise ValueError(f"Missing labels for: {', '.join(missing[:10])}")
    if not pairs:
        raise ValueError("No image-label pairs found")
    return pairs


def split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed):
    if any(ratio < 0 for ratio in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("split ratios cannot be negative")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train + val + test must equal 1")
    # Use a local random generator so the same seed reproduces the same split.
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    splits = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
    for split_name, ratio in (("train", train_ratio), ("val", val_ratio), ("test", test_ratio)):
        if ratio > 0 and not splits[split_name]:
            raise ValueError(f"{split_name} would be empty; add data or change the split ratios")
    return splits


def copy_splits(splits, output):
    for split_name, pairs in splits.items():
        image_output = output / "images" / split_name
        label_output = output / "labels" / split_name
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in pairs:
            # Preserve the source dataset; the output is a separate experiment input.
            shutil.copy2(image_path, image_output / image_path.name)
            shutil.copy2(label_path, label_output / label_path.name)
        print(f"{split_name}={len(pairs)}")


def validate_output(images, labels, output):
    images = images.resolve()
    labels = labels.resolve()
    output = output.resolve()
    if output in {images, labels} or images in output.parents or labels in output.parents:
        raise ValueError("output cannot be the same as or inside the source directories")
    if output.exists():
        # Repo doc files (README.md, .gitkeep) are not user data; everything else blocks reuse.
        existing = [
            path
            for path in output.rglob("*")
            if path.is_file() and path.name not in {".gitkeep", "README.md"}
        ]
        if existing:
            raise FileExistsError("output already contains dataset files; choose a new or empty directory")


def main():
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        raise FileNotFoundError("IMAGES_DIR and LABELS_DIR must both be directories")
    validate_output(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR)
    pairs = collect_pairs(IMAGES_DIR, LABELS_DIR)
    splits = split_pairs(pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)
    copy_splits(splits, OUTPUT_DIR)


if __name__ == "__main__":
    main()
