# YOLO11 教学脚本

这些脚本覆盖完整的 YOLO11 工作流：视频抽帧、VOC 标签转换、空背景处理、数据集划分、训练与续训、验证、摄像头/图片预测、ONNX 导出和量化图片准备。所有输入、输出和训练选项都写在每个脚本开头的调参区里，改完直接运行；数据处理脚本不会移动或删除源数据。

## 从原始素材到板端模型

1. [`video2frame.py`](video2frame.py)：按间隔抽帧并调整为部署画面尺寸；使用 `imencode().tofile()` 兼容 Windows 中文路径。
2. [`classes.txt`](classes.txt)：一行一个真实目标类别，行号从 0 开始；负样本不写 `background` 类。
3. [`voc_to_yolo.py`](voc_to_yolo.py)：把 EasyData/EasyDL 导出的 Pascal VOC XML 转成 YOLO 标签，缺少 XML 的图片只有在显式确认后才按空背景处理。
4. [`check_dataset.py`](check_dataset.py)：检查图片与标签是否配对、类别是否越界、归一化框是否超出图片。
5. [`prepare_dataset.py`](prepare_dataset.py)：固定随机种子复制出 `train/val/test`，不移动源数据。
6. [`train.py`](train.py)：训练或显式续训；`cache`、`time`、`close_mosaic`、`optimizer` 和周期保存都可由参数控制。
7. [`validate.py`](validate.py) 与 [`predict.py`](predict.py)：分别生成验证指标/图表，以及检查图片、视频或摄像头中的具体检测框。
8. [`export_onnx.py`](export_onnx.py)：导出固定输入尺寸、无 NMS 的 ONNX。
9. [`prepare_quant_images.py`](prepare_quant_images.py)：从训练图片中按固定随机种子选取并缩放代表性量化图片。

完整解释见[03 阶段学习路线](../../../学习路线/03-YOLO11训练与部署/README.md)，EasyData、标注规范和数据划分见[数据准备](../../../学习路线/03-YOLO11训练与部署/02-数据准备.md)。

## 一次最小闭环

以下步骤假设当前目录就是 `配套资源\示例代码\yolo11`，EasyData 导出结果已经整理为 `voc\Images` 与 `voc\Annotations`。每一步都先打开脚本、把调参区改成当次实验的值，再直接运行：

```powershell
python video2frame.py           # INPUT_VIDEO / OUTPUT_DIR / INTERVAL / WIDTH / HEIGHT / PREFIX
python voc_to_yolo.py           # IMAGES_DIR / ANNOTATIONS_DIR / OUTPUT_DIR；负样本确认后 MISSING_AS_BACKGROUND = True
python check_dataset.py         # IMAGES_DIR / LABELS_DIR / CLASSES_FILE
python prepare_dataset.py       # OUTPUT_DIR / TRAIN_RATIO / VAL_RATIO / TEST_RATIO / SEED
python train.py                 # 冒烟: EPOCHS=3, BATCH=4, WORKERS=0, NAME="smoke"；默认值即 baseline
python validate.py              # WEIGHTS / SPLIT / IOU
python predict.py               # WEIGHTS / SOURCE / CONF / SHOW / SAVE
python export_onnx.py           # WEIGHTS / IMGSZ_HEIGHT / IMGSZ_WIDTH
python prepare_quant_images.py  # INPUT_DIR / OUTPUT_DIR / LIMIT / SEED
```

先修改 `classes.txt` 与 `data.yaml`，确保类别名称和顺序完全相同。只有逐张确认“无 XML 的图片确实没有目标”后，才能把 `MISSING_AS_BACKGROUND` 改为 `True`；漏标图片被误当成负样本会直接教错模型。

`dataset/` 只保留目录说明，这里不发布我个人的数据集。所有数据准备脚本只向新目录复制或写入文件，输出目录已有内容时会停止，避免覆盖一次已经完成的转换或划分。
