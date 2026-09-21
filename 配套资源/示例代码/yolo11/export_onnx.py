# ============================================================
# export_onnx.py - 把权重导出为固定输入尺寸的 ONNX
# MaixCAM 转换流程期望固定输入, 且宽高是 32 的倍数; 导出时不带
# NMS, 后处理由 MaixPy 的 YOLO11 封装完成。
# 用法: 修改下方调参区, 然后直接运行  python export_onnx.py
# ============================================================

# ==================== 调参区 ====================
WEIGHTS      = r"runs\train\baseline\weights\best.pt"  # 要导出的权重
IMGSZ_HEIGHT = 224   # 输入高 (必须是 32 的倍数; 本项目板端输入为 224)
IMGSZ_WIDTH  = 320   # 输入宽 (必须是 32 的倍数; 本项目板端输入为 320)
DEVICE       = "cpu" # 导出设备; 一般用 "cpu"
OPSET        = 17    # ONNX opset 版本; 转换平台不兼容时以平台文档为准
# ================================================


def main():
    # MaixCAM conversion expects a fixed input whose dimensions match the model stride.
    if any(size < 1 or size % 32 for size in (IMGSZ_HEIGHT, IMGSZ_WIDTH)):
        raise ValueError("both image dimensions must be positive multiples of 32")
    # Import lazily so opening this file works before the training environment is installed.
    from ultralytics import YOLO

    path = YOLO(WEIGHTS).export(
        format="onnx",
        imgsz=[IMGSZ_HEIGHT, IMGSZ_WIDTH],
        dynamic=False,
        simplify=True,
        opset=OPSET,
        batch=1,
        device=DEVICE,
        nms=False,
    )
    print(path)


if __name__ == "__main__":
    main()
