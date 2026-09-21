# ============================================================
# predict.py - 用权重在图片/视频/摄像头上做预测
# 逐帧打印类别 ID、置信度和归一化 xywh, 便于与 YOLO 标签、
# ONNX 输出和板端输出对比; 也可保存画框结果或弹窗实时查看。
# 用法: 修改下方调参区, 然后直接运行  python predict.py
# ============================================================

# ==================== 调参区 ====================
WEIGHTS = r"runs\train\baseline\weights\best.pt"  # 权重; ONNX 同图比较时改 .onnx
SOURCE  = "dataset/images/test"  # 图片/视频/目录路径, 或摄像头编号 "0"
IMGSZ   = 640                    # 推理输入尺寸
CONF    = 0.5                    # 置信度阈值 (0~1)
DEVICE  = "0"                    # "0" = 第一块 GPU; 没有 NVIDIA 显卡改 "cpu"
SHOW    = False                  # True = 弹窗实时显示画面
SAVE    = True                   # True = 保存画框结果到 PROJECT/NAME
PROJECT = "runs/predict"         # 输出根目录
NAME    = "baseline-test"        # 实验名
# ================================================


def parse_source(value):
    # Ultralytics expects an integer for a camera and a string for files or URLs.
    return int(value) if str(value).isdecimal() else value


def main():
    if not 0 <= CONF <= 1:
        raise ValueError("CONF must be between 0 and 1")

    from ultralytics import YOLO

    # Streaming avoids collecting an entire video or camera session in memory.
    results = YOLO(WEIGHTS).predict(
        source=parse_source(SOURCE),
        imgsz=IMGSZ,
        conf=CONF,
        device=DEVICE,
        show=SHOW,
        save=SAVE,
        project=PROJECT,
        name=NAME,
        stream=True,
        verbose=False,
    )
    for frame_index, result in enumerate(results):
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x_center, y_center, width, height = box.xywhn[0].tolist()
            print(
                f"frame={frame_index} class_id={class_id} confidence={confidence:.4f} "
                f"xywhn={x_center:.4f},{y_center:.4f},{width:.4f},{height:.4f}"
            )


if __name__ == "__main__":
    main()
