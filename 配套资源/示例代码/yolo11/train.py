# ============================================================
# train.py - YOLO11 训练 / 续训
# 每次实验的所有取舍 (数据、模型、轮数、显存相关参数) 都集中写在
# 下面的调参区里, 改完直接运行  python train.py
# 冒烟训练: EPOCHS=3, BATCH=4, WORKERS=0, NAME="smoke"
# 续训:    MODEL 指向该实验的 last.pt, RESUME=True
# 输出目录是 PROJECT/NAME, 里面的 args.yaml 记录了本次全部参数。
# ============================================================

# ==================== 调参区 ====================
DATA         = "data.yaml"           # 数据集配置 (路径与类别顺序)
MODEL        = "yolo11n.pt"          # 官方预训练权重; 续训时指向 last.pt
EPOCHS       = 300                   # 训练轮数 (冒烟训练先改 3)
IMGSZ        = 640                   # 训练输入尺寸 (会保持宽高比缩放填充)
BATCH        = 16                    # 批大小; 显存不足先减小它
WORKERS      = 2                     # 数据加载进程数; Windows 首次运行改 0
DEVICE       = "0"                   # "0" = 第一块 GPU; 没有 NVIDIA 显卡改 "cpu"
OPTIMIZER    = "auto"                # 只有做对照实验时才显式选择 SGD 等
CACHE        = "none"                # "ram" 用内存换速度 / "disk" 磁盘缓存 / "none"
CLOSE_MOSAIC = 10                    # 最后 N 个 epoch 关闭 mosaic 增强
PATIENCE     = 100                   # 早停耐心值
SAVE_PERIOD  = -1                    # 每 N 个 epoch 额外保存权重; -1 = 不启用
TIME_HOURS   = None                  # 限制训练时长(小时); 启用时可能跑不满 EPOCHS
SEED         = 0                     # 随机种子
PROJECT      = "runs/train"          # 输出根目录
NAME         = "baseline"            # 实验名 (冒烟训练改 "smoke")
RESUME       = False                 # True = 从 last.pt 恢复优化器、调度器和轮数
# ================================================


def main():
    # Import lazily so opening this file still works before Ultralytics is installed.
    from ultralytics import YOLO

    if EPOCHS < 1 or BATCH == 0 or WORKERS < 0:
        raise ValueError("EPOCHS must be positive, BATCH cannot be zero, and WORKERS cannot be negative")
    if CLOSE_MOSAIC < 0 or PATIENCE < 0 or TIME_HOURS is not None and TIME_HOURS <= 0:
        raise ValueError("CLOSE_MOSAIC/PATIENCE cannot be negative and TIME_HOURS must be positive")

    model = YOLO(MODEL)
    # Keep every experiment choice visible instead of inheriting personal paths or devices.
    options = dict(
        data=DATA,
        epochs=EPOCHS,
        device=DEVICE,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        optimizer=OPTIMIZER,
        cache=False if CACHE == "none" else CACHE,
        close_mosaic=CLOSE_MOSAIC,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,
        seed=SEED,
        resume=RESUME,
        plots=True,
    )
    if TIME_HOURS is not None:
        options["time"] = TIME_HOURS
    model.train(**options)


if __name__ == "__main__":
    main()
