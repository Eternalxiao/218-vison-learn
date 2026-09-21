# ============================================================
# validate.py - 独立验证训练好的权重
# 在指定数据集划分上重新计算 mAP50 / mAP75 / mAP50-95,
# 并保存混淆矩阵和预测图, 用于独立于训练过程的验证。
# 用法: 修改下方调参区, 然后直接运行  python validate.py
# ============================================================

# ==================== 调参区 ====================
WEIGHTS = r"runs\train\baseline\weights\best.pt"  # 要验证的权重
DATA    = "data.yaml"     # 数据集配置 (与训练时同一份)
SPLIT   = "val"           # 在哪个划分上验证: "val" / "test" / "train"
IMGSZ   = 640             # 验证输入尺寸
BATCH   = 16              # 批大小
DEVICE  = "0"             # "0" = 第一块 GPU; 没有 NVIDIA 显卡改 "cpu"
IOU     = 0.6             # 推理时非极大值抑制(NMS)的 IoU 阈值
RECT    = False           # True = 矩形验证 (减少填充, 与训练设置不一致时慎用)
PROJECT = "runs/val"      # 输出根目录
NAME    = "baseline"      # 实验名
# ================================================


def main():
    # Import lazily so dataset checks do not require a GPU environment.
    from ultralytics import YOLO

    metrics = YOLO(WEIGHTS).val(
        data=DATA,
        split=SPLIT,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        iou=IOU,
        rect=RECT,
        project=PROJECT,
        name=NAME,
        plots=True,
    )
    # These summaries complement, rather than replace, the saved error images and plots.
    print(f"mAP50-95={metrics.box.map:.6f} mAP50={metrics.box.map50:.6f} mAP75={metrics.box.map75:.6f}")


if __name__ == "__main__":
    main()
