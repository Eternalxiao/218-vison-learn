# ============================================================
# feature_pipeline.py - 传统视觉处理链 (保存每个中间结果)
# 对一张图片依次执行: 颜色空间转换 → 滤波 → 二值化 → 形态学 →
# 轮廓筛选, 并把每个阶段的中间图都保存到输出目录, 方便观察
# "哪一步开始不对"。
# 用法: 修改下方调参区, 然后直接运行  python feature_pipeline.py
# 每次只改一组参数, 再比较最早发生变化的中间图。 -- 本文效果与 “配套资源\工具\视觉流程调试器” 目标相同

# ============================================================

from pathlib import Path

import cv2
import numpy as np

# ==================== 调参区 ====================
INPUT_IMAGE = Path("sample.png")  # 输入图片 (可先用 make_sample_image.py 生成)
OUTPUT_DIR  = Path("output")      # 中间结果保存目录

CONFIG = {
    # --- 颜色空间: ranges 阈值需要 "hsv" 或 "lab" ---
    "color_space": "hsv",           # "gray" / "hsv" / "lab"

    # --- 滤波 ---
    "filter": "gaussian",           # "none" / "gaussian" / "median"
    "filter_kernel": 3,             # 滤波核大小 (奇数)

    # --- 二值化 ---
    "threshold_mode": "ranges",     # "ranges" / "fixed" / "otsu" / "adaptive"
    # ranges: 多段颜色区间, 每项 [最小值x3, 最大值x3]
    #   HSV 每项为 [Hmin, Smin, Vmin, Hmax, Smax, Vmax]
    #   LAB 每项为 [Lmin, Amin, Bmin, Lmax, Amax, Bmax]
    "ranges": [
        [0, 100, 80, 12, 255, 255],
        [168, 100, 80, 179, 255, 255],
    ],
    "fixed_threshold": 127,         # threshold_mode = "fixed" 时使用
    "adaptive_block_size": 31,      # adaptive 模式块大小 (奇数, >= 3)
    "adaptive_c": 5,                # adaptive 模式常数项

    # --- 形态学 (按 腐蚀 → 膨胀 → 开 → 闭 的顺序执行) ---
    "morph_kernel": 3,              # 结构元素大小 (奇数)
    "erode_iterations": 0,          # 腐蚀次数
    "dilate_iterations": 0,         # 膨胀次数
    "open_iterations": 1,           # 开运算次数
    "close_iterations": 1,          # 闭运算次数

    # --- 轮廓筛选 ---
    "min_contour_area": 100,        # 面积小于该值 (px^2) 的轮廓直接丢弃
}
# ================================================


def odd_kernel(value, name):
    value = int(value)
    if value < 1 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
    return value


def convert_color(image, color_space):
    conversions = {
        "gray": cv2.COLOR_BGR2GRAY,
        "hsv": cv2.COLOR_BGR2HSV,
        "lab": cv2.COLOR_BGR2LAB,
    }
    try:
        return cv2.cvtColor(image, conversions[color_space])
    except KeyError as exc:
        raise ValueError("color_space must be gray, hsv, or lab") from exc


def filter_image(image, mode, kernel_size):
    if mode == "none":
        return image.copy()
    kernel_size = odd_kernel(kernel_size, "filter_kernel")
    if mode == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    if mode == "median":
        return cv2.medianBlur(image, kernel_size)
    raise ValueError("filter must be none, gaussian, or median")


def multi_range_mask(image, ranges):
    if image.ndim != 3:
        raise ValueError("ranges mode needs hsv or lab color_space")
    if not ranges:
        raise ValueError("ranges mode needs at least one range")
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for values in ranges:
        if len(values) != 6:
            raise ValueError("each range must contain six values")
        lower = np.array(values[:3], dtype=np.uint8)
        upper = np.array(values[3:], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(image, lower, upper))
    return mask


def threshold_image(color_image, gray_image, config):
    mode = config.get("threshold_mode", "ranges")
    if mode == "ranges":
        return multi_range_mask(color_image, config.get("ranges", []))
    if mode == "fixed":
        _, mask = cv2.threshold(gray_image, int(config.get("fixed_threshold", 127)), 255, cv2.THRESH_BINARY)
        return mask
    if mode == "otsu":
        _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    if mode == "adaptive":
        block = odd_kernel(config.get("adaptive_block_size", 31), "adaptive_block_size")
        if block < 3:
            raise ValueError("adaptive_block_size must be at least 3")
        return cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            float(config.get("adaptive_c", 5)),
        )
    raise ValueError("threshold_mode must be ranges, fixed, otsu, or adaptive")


def run_pipeline(image, config):
    color = convert_color(image, config.get("color_space", "hsv"))
    filtered = filter_image(color, config.get("filter", "none"), config.get("filter_kernel", 3))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_filtered = filter_image(gray, config.get("filter", "none"), config.get("filter_kernel", 3))
    binary = threshold_image(filtered, gray_filtered, config)
    kernel_size = odd_kernel(config.get("morph_kernel", 3), "morph_kernel")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    eroded = cv2.erode(binary, kernel, iterations=int(config.get("erode_iterations", 0)))
    dilated = cv2.dilate(eroded, kernel, iterations=int(config.get("dilate_iterations", 0)))
    opened = cv2.morphologyEx(
        dilated, cv2.MORPH_OPEN, kernel, iterations=int(config.get("open_iterations", 0))
    )
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel, iterations=int(config.get("close_iterations", 0))
    )

    contour_view = image.copy()
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= float(config.get("min_contour_area", 100))]
    cv2.drawContours(contour_view, kept, -1, (0, 255, 0), 2)
    return {
        "00-original": image,
        "01-color": color,
        "02-filtered": filtered,
        "03-binary": binary,
        "04-eroded": eroded,
        "05-dilated": dilated,
        "06-opened": opened,
        "07-closed": closed,
        "08-contours": contour_view,
    }


def main():
    image = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {INPUT_IMAGE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, stage in run_pipeline(image, CONFIG).items():
        output = OUTPUT_DIR / f"{name}.png"
        if not cv2.imwrite(str(output), stage):
            raise RuntimeError(f"Could not write image: {output}")
        print(output)


if __name__ == "__main__":
    main()
