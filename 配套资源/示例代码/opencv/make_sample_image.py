# ============================================================
# make_sample_image.py - 生成一张练习用图片
# 画一张含矩形、圆形、三角形和文字的合成图, 供 OpenCV 和
# 浏览器调试器练习使用。
# 用法: 修改下方调参区, 然后直接运行  python make_sample_image.py
# ============================================================

from pathlib import Path

import cv2
import numpy as np

# ==================== 调参区 ====================
OUTPUT = Path("sample.png")  # 输出图片路径
# ================================================


def main():
    canvas = np.full((360, 560, 3), 235, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 50), (220, 190), (20, 20, 220), -1)
    cv2.circle(canvas, (390, 125), 72, (20, 180, 20), -1)
    points = np.array([[140, 270], [250, 220], [315, 325]], dtype=np.int32)
    cv2.fillPoly(canvas, [points], (220, 60, 20))
    cv2.putText(canvas, "TILearn", (350, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), canvas):
        raise RuntimeError(f"Could not write image: {OUTPUT}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
