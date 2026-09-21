# ONNX、量化与 MaixCAM Pro 部署

部署不是把文件后缀从 `.pt` 改成 `.cvimodel`。这一段数据流中至少有四种产物：

```text
best.pt
→ 固定输入 best.onnx
→ MaixCAM Pro 执行的 model.cvimodel
→ 描述模型与预处理的 model.mud
→ nn.YOLO11 加载 .mud 并完成板端后处理
```

任何一步改变输入尺寸、颜色顺序、归一化或类别顺序，PC 与板端就可能得到不同结果。

以下命令从教学脚本目录运行：

```powershell
Set-Location 配套资源\示例代码\yolo11
```

## 1. 在训练前就考虑板端输入

本篇示例使用 320×224。建议两个尺寸都是 32 的倍数并且接近常见摄像头画面比例。但这个数不是所有项目的固定答案；更换分辨率会同时影响小目标信息、计算量、量化图片和摄像头配置。

训练默认使用 `imgsz=640` 时，Ultralytics 会在保持宽高比的前提下缩放和填充图片。导出 320×224 是为了固定板端输入，并不表示采集阶段应该把任意比例图片强行拉伸。训练前应尽量使用与部署相近的画面比例，并在 PC 导出前后做同图比较。

## 2. 从 best.pt 导出固定输入 ONNX

打开 [`export_onnx.py`](../../配套资源/示例代码/yolo11/export_onnx.py) 确认调参区（默认就是 `best.pt`、224×320、`DEVICE = "cpu"`、`OPSET = 17`），然后直接运行：

```powershell
python export_onnx.py
```

它明确设置：

- 输入尺寸写在 `IMGSZ_HEIGHT = 224`、`IMGSZ_WIDTH = 320`，先高后宽，都必须是 32 的倍数。
- `dynamic=False`，导出固定输入尺寸。
- `batch=1`，对应板端逐帧推理。
- `nms=False`，由 MaixPy 的 YOLO11 封装完成后处理。
- `OPSET = 17`；若当前转换平台提示不兼容，应以平台文档为准并记录改动。

## 3. ONNX 先在 PC 上做同图比较

选择一组固定图片，至少包含：普通正样本、小目标、遮挡、易误检背景和无目标画面。先保存 PyTorch 权重预测（[`predict.py`](../../配套资源/示例代码/yolo11/predict.py)调参区：`WEIGHTS` 指向 `best.pt`，`DEVICE = "cpu"`，`SAVE = True`，`PROJECT = "runs/compare"`，`NAME = "pytorch"`）：

```powershell
python predict.py
```

再使用导出的 ONNX（把调参区的 `WEIGHTS` 改成 `best.onnx`，`NAME = "onnx"`）：

```powershell
python predict.py
```

比较类别、框位置、漏检和误检。浮点实现之间置信度不必逐位相等，但若类别顺序错位、框整体偏移或大量结果消失，应停在 ONNX 阶段排查，不要继续量化。

还可以用 Netron 查看 ONNX 输入形状是否为 `1×3×224×320`，记录输入和输出节点名。手动转换时这些信息会直接用于转换参数。

## 4. 量化图片要代表输入分布

打开 [`prepare_quant_images.py`](../../配套资源/示例代码/yolo11/prepare_quant_images.py) 确认调参区（默认 `INPUT_DIR` 为训练图片、`WIDTH = 320`、`HEIGHT = 224`、`LIMIT = 100`、`SEED = 42`），然后直接运行：

```powershell
python prepare_quant_images.py
```

它把校准图片复制到新目录并缩放为 320×224；设置 `LIMIT` 时按固定随机种子抽样，而不是只取文件名最前面的一批。

量化图片应来自训练来源并覆盖：

- 常见背景与无目标画面。
- 明暗变化、阴影和反光。
- 近处大目标和远处小目标。
- 多目标、遮挡和画面边缘。

不要只选检测效果最好的图片。量化集的任务是代表部署输入分布，不是展示模型成绩。也不要用 test 结果反复挑选“最有利”的量化图片。

脚本会直接缩放到目标尺寸。若源图片比例与 320×224 不一致，应先决定转换工具实际采用的裁剪或填充规则，不能一边在 PC 上 letterbox、一边给转换器使用拉伸后的量化图。

## 5. 转换与部署

### 普通 YOLO11 优先使用网页转换

Sipeed 当前[网页转换 YOLO 模型](https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/online_converter.html)支持 YOLO11，并会为 MaixCAM/MaixCAM Pro 生成 `.mud + .cvimodel`。普通检测模型先使用网页转换；网页不支持模型结构或需要自定义节点时，再进入[手动转换文档](https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam.html)。

转换任务至少要记录：

```text
来源权重：runs/train/baseline/weights/best.pt
ONNX 文件：best.onnx
输入形状：1×3×224×320
模型类型：YOLO11 detect
类别顺序：与 classes.txt / data.yaml 一致
量化图片：quant_images 的版本与数量
转换平台或工具版本：实际页面/容器版本
输出文件：model.mud + model.cvimodel
```

### 个人部署

对于我们直接使用来说整个流程非常简单，搭建虚拟机-> 拉取对应的docker容器->根据项目情况改写自动化脚本，所以这一步的难点不是转换，而是对虚拟机、docker的了解与使用，请仔细阅读官网中对应的部分，文件[VisionUbuntuClone](../../配套资源/工具/)提供个人 VMware 的 clone，密码为 `fxq76836`，但是依旧建议自己学习搭建虚拟机和docker的相关知识。

文章[一篇 YOLOv5 转换记录](https://blog.csdn.net/m0_75041317/article/details/142930573)可用于理解个人部署的的详细步骤这里不进行深入讲解。

## 6. `.mud` 与 `.cvimodel` 怎样配对

仓库中的 [`TIbegin_int8.mud`](../../配套资源/模型/TIbegin_int8.mud)展示了 `.mud` 与 `.cvimodel` 的配对关系：

```ini
[basic]
type = cvimodel
model = TIbegin_int8.cvimodel

[extra]
model_type = yolo11
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
labels = ...
```

- `model` 必须指向同一模型包中的 `.cvimodel` 文件。
- `model_type` 决定 MaixPy 使用哪种模型封装和后处理。
- `input_type`、`mean`、`scale` 必须与转换时预处理一致。
- `labels` 的数量与顺序必须和训练数据一致。

这份归档中的历史标签拼写按事实保留，不应复制成新模型的标签。新模型应从自己的 `classes.txt` 生成并逐项核对。

## 7. 上传模型并运行板端示例

把同一转换任务生成的 `.mud` 与 `.cvimodel` 一起上传到 MaixCAM Pro，例如：

```text
/root/models/my_yolo11.mud
/root/models/my_yolo11.cvimodel
```

然后修改[板端推理示例](../../配套资源/示例代码/maixpy/yolo11_infer.py)中的模型路径。脚本的核心数据流是：

```python
detector = nn.YOLO11(model=MODEL_PATH, dual_buff=True)
cam = camera.Camera(detector.input_width(), detector.input_height())
frame = cam.read()
objects = detector.detect(frame, conf_th=0.5, iou_th=0.45)
```

相机尺寸从模型读取，可以减少手写尺寸与模型输入不一致的机会。`dual_buff=True` 会影响取图和推理的流水方式；它不是模型正确性的证明，调试时仍要记录实际画面和结果。

## 8. 同图对照决定问题在哪一层

用同一张原始图片依次运行：

1. `best.pt`。
2. `best.onnx`。
3. `.mud + .cvimodel` 的板端模型。

每一层保存：输入图片、输入尺寸、类别与置信度、框坐标和阈值。可以按下面顺序判断：

- `.pt` 已经错：回到数据和训练。
- `.pt` 正常、ONNX 错：检查导出尺寸、输出和 ONNX 运行时。
- ONNX 正常、板端错：检查量化图片、颜色顺序、`mean/scale`、输出节点和 `.mud`。
- 固定图片一致、摄像头现场错：检查摄像头画面比例、曝光、对焦和现场数据分布。

最后再记录取图、推理、绘制和显示耗时。只有总帧率无法判断瓶颈位于哪一步。

## 当前验证边界

仓库中的[模型说明](../../配套资源/模型/README.md)、`TIbegin_int8.mud`、同名 `.cvimodel` 和板端示例形成了可读的文件闭环，但本次没有在所有 MaixCAM Pro 固件与设备组合上重新加载并做同图对照。因此这里只说明检查方法，不声称归档模型在任意环境开箱即用。

外部链接最后核验于 2026-09-17。
