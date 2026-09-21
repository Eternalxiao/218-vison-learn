# ============================================================
# prepare_quant_images.py - 准备模型量化用的校准图片
# 从训练图片中按固定随机种子抽样, 复制并缩放到板端输入尺寸。
# 量化图片要代表部署时的输入分布 (含无目标画面、明暗变化、
# 远近目标), 不是挑检测效果最好的图片。
# 用法: 修改下方调参区, 然后直接运行  python prepare_quant_images.py
# ============================================================

import random
from pathlib import Path

from PIL import Image, ImageOps

# ==================== 调参区 ====================
INPUT_DIR  = Path("dataset/images/train")  # 抽样来源 (应来自训练集)
OUTPUT_DIR = Path("quant_images")          # 输出目录 (须为空; 交给转换平台)
WIDTH      = 320                           # 输出宽 (与板端输入一致)
HEIGHT     = 224                           # 输出高 (与板端输入一致)
LIMIT      = 100                           # 抽样张数; 0 = 全部使用
SEED       = 42                            # 随机种子; 同一种子抽到同一批图
# ================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(INPUT_DIR)
    if WIDTH < 1 or HEIGHT < 1 or LIMIT < 0:
        raise ValueError("WIDTH/HEIGHT must be positive and LIMIT cannot be negative")
    if OUTPUT_DIR.exists() and any(path.is_file() for path in OUTPUT_DIR.rglob("*")):
        raise FileExistsError("output already contains files; choose a new or empty directory")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = [path for path in sorted(INPUT_DIR.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise ValueError("no supported images found")
    # A seeded sample avoids taking only one scene because its filename sorts first.
    if LIMIT and LIMIT < len(images):
        images = random.Random(SEED).sample(images, LIMIT)
        images.sort()
    for index, source in enumerate(images):
        with Image.open(source) as image:
            converted = ImageOps.exif_transpose(image).convert("RGB")
            converted.resize((WIDTH, HEIGHT)).save(OUTPUT_DIR / f"quant_{index:05d}.png")
    print(f"saved={len(images)} output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
