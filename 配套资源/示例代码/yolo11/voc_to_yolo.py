# ============================================================
# voc_to_yolo.py - Pascal VOC 转 YOLO 标签
# 把 EasyData/EasyDL 导出的 Pascal VOC XML 框转成 YOLO 文本标签,
# 图片按复制处理, 不移动、不删除原始导出数据。
# 缺少 XML 的图片默认直接报错; 只有逐张确认是"无目标负样本"后,
# 才把 MISSING_AS_BACKGROUND 改为 True。
# 用法: 修改下方调参区, 然后直接运行  python voc_to_yolo.py
# ============================================================

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ==================== 调参区 ====================
IMAGES_DIR           = Path("voc/Images")       # EasyData 导出的图片目录
ANNOTATIONS_DIR      = Path("voc/Annotations")  # 同一次导出的 VOC XML 标注目录
CLASSES_FILE         = Path("classes.txt")      # 类别清单, 一行一个类别, 行号从 0 开始
OUTPUT_DIR           = Path("converted")        # 输出目录 (生成 images/ 和 labels/, 须为空)
MISSING_AS_BACKGROUND = False                   # True = 无 XML 的图片按空背景输出。
                                                # 漏标图片被当成负样本会直接教错模型
# ================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names(path):
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError("classes file is empty")
    if len(names) != len(set(names)):
        raise ValueError("classes file contains duplicate names")
    return names


def collect_images(directory):
    images = [path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError("image filenames must have unique stems")
    if not images:
        raise ValueError("no supported images found")
    return images


def read_number(parent, name, source):
    element = parent.find(name)
    if element is None or element.text is None:
        raise ValueError(f"missing {name} in {source}")
    return float(element.text)


def convert_box(width, height, xmin, ymin, xmax, ymax, source):
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in {source}")
    if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
        raise ValueError(f"box is outside the image in {source}: {(xmin, ymin, xmax, ymax)}")
    # VOC stores pixel corners; YOLO stores a normalized center and size.
    return (
        (xmin + xmax) / 2.0 / width,
        (ymin + ymax) / 2.0 / height,
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


def convert_annotation(xml_path, class_ids):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing size in {xml_path}")
    width = read_number(size, "width", xml_path)
    height = read_number(size, "height", xml_path)

    rows = []
    for obj in root.findall("object"):
        name_element = obj.find("name")
        box = obj.find("bndbox")
        if name_element is None or name_element.text is None or box is None:
            raise ValueError(f"incomplete object in {xml_path}")
        class_name = name_element.text.strip()
        if class_name not in class_ids:
            raise ValueError(f"unknown class {class_name!r} in {xml_path}")
        normalized = convert_box(
            width,
            height,
            read_number(box, "xmin", xml_path),
            read_number(box, "ymin", xml_path),
            read_number(box, "xmax", xml_path),
            read_number(box, "ymax", xml_path),
            xml_path,
        )
        rows.append(f"{class_ids[class_name]} " + " ".join(f"{value:.8f}" for value in normalized))
    return rows


def ensure_empty_output(output):
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        raise FileExistsError("output already contains files; choose a new or empty directory")
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)


def main():
    if not IMAGES_DIR.is_dir() or not ANNOTATIONS_DIR.is_dir():
        raise FileNotFoundError("IMAGES_DIR and ANNOTATIONS_DIR must both be directories")
    if not CLASSES_FILE.is_file():
        raise FileNotFoundError(CLASSES_FILE)

    class_names = load_class_names(CLASSES_FILE)
    class_ids = {name: index for index, name in enumerate(class_names)}
    images = collect_images(IMAGES_DIR)
    # A missing XML file may mean either a true negative or an export mistake.
    missing = [image.name for image in images if not (ANNOTATIONS_DIR / f"{image.stem}.xml").is_file()]
    if missing and not MISSING_AS_BACKGROUND:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"images without XML annotations: {preview}; "
            "confirm them, then set MISSING_AS_BACKGROUND = True"
        )

    ensure_empty_output(OUTPUT_DIR)
    background_count = 0
    box_count = 0
    for image in images:
        xml_path = ANNOTATIONS_DIR / f"{image.stem}.xml"
        rows = convert_annotation(xml_path, class_ids) if xml_path.is_file() else []
        if not rows:
            background_count += 1
        box_count += len(rows)
        # Copy instead of moving so the EasyData/VOC export remains recoverable.
        shutil.copy2(image, OUTPUT_DIR / "images" / image.name)
        (OUTPUT_DIR / "labels" / f"{image.stem}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )

    print(
        f"images={len(images)} boxes={box_count} background_images={background_count} "
        f"classes={len(class_names)} output={OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
