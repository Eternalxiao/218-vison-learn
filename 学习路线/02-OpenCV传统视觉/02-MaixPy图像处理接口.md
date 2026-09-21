# 02：在 MaixCAM 上使用 MaixPy 图像处理接口

这篇的前提是你已经读完[滤波、形态学与轮廓](01-滤波、形态学与轮廓.md)，能从中间图判断自己需要平滑像素、修补二值区域，还是筛选目标。这里不再重新讲高斯滤波、开闭运算的原理，只解决两个问题：OpenCV 中学过的操作在 MaixPy 里怎样调用，以及怎样避免为了使用 OpenCV 而增加不必要的格式转换和计算开销。

MaixCAM 可以运行 Python 版 OpenCV，但 Sipeed 官方说明指出，OpenCV 函数基本由 CPU 计算，而 `maix` 模块中的许多函数经过硬件加速。因此设备端应先检查 `maix.image.Image` 是否已经提供所需能力；只有缺少对应操作、需要复杂 OpenCV 处理，或已经测量确认方案可接受时，才转换到 NumPy/OpenCV。

## 先看清 MaixPy 与 OpenCV 的三个差异

### 核尺寸的写法不同

MaixPy 的 `size` 不是 OpenCV 的 `(宽, 高)`。`size=1` 表示 3×3 核，`size=2` 表示 5×5 核，实际边长为 `2 × size + 1`。因此：

- OpenCV 的 `cv2.GaussianBlur(img, (3, 3), 0)` 对应从 `img.gaussian(1)` 开始比较。
- OpenCV 的 3×3 腐蚀核对应从 `img.erode(1)` 开始比较。

这只是核尺寸的对应，不表示两端的边界处理、参数和计算结果逐像素完全一致。迁移后仍要保存同一输入的结果做对比。

### 不少方法会直接修改当前图像

MaixPy 官方 API 明确说明 `erode()`、`dilate()` 会原地修改图像；`binary()` 的 `copy` 默认也是 `False`。为了同时显示原图与中间结果，调试阶段先复制，再处理：

```python
filtered = frame.copy()
filtered.gaussian(1)

binary = filtered.binary(TARGET_LAB, copy=True)
```

若直接对 `frame` 连续调用多个方法，前一步的信息会被覆盖，随后很难判断是哪一步开始出错。

### 阈值与形态学参数不能按名字混用

在 `FMT_RGB888` 图像上，`binary()` 和 `find_blobs()` 的六个阈值是 LAB 顺序：`Lmin, Lmax, Amin, Amax, Bmin, Bmax`；灰度图使用 `[Lmin, Lmax]`。OpenCV 的 RGB、BGR 或 HSV 数值不能直接粘贴过来。

`erode()`、`dilate()`、`open()` 和 `close()` 还提供名为 `threshold` 的参数，它表示核内非零像素数量的判定条件，不是颜色阈值，也不是 OpenCV 的 `iterations`。初次迁移时保留默认值，只调整 `size`，避免同时引入两种变化。

## 把已学操作映射到 MaixPy

### 滤波接口

- OpenCV 的均值滤波对应 `img.mean(size)`。
- 高斯滤波对应 `img.gaussian(size)`；`unsharp=True` 会改为非锐化掩模用途，不应在普通去噪时顺手开启。
- 中值滤波对应 `img.median(size, percentile=0.5)`；官方文档提示它较慢，是否满足实时性要在设备上测量。
- 双边滤波对应 `img.bilateral(size, color_sigma, space_sigma)`；两项 sigma 的含义与取值不要从 OpenCV 示例直接照搬。

`img.morph(size, kernel, ...)` 是自定义卷积接口，不是“形态学”的总入口。腐蚀、膨胀、开运算和闭运算分别使用下面四个独立方法。

### 二值化与形态学接口

- `img.binary(thresholds, copy=True)`：按一组或多组阈值生成便于观察的黑白图。
- `img.erode(size)`：执行已经在上一篇学过的腐蚀。
- `img.dilate(size)`：执行膨胀。
- `img.open(size)`：依次执行腐蚀与膨胀。
- `img.close(size)`：依次执行膨胀与腐蚀。

这里故意不再写“哪个操作适合哪种现象”。选择依据与信息损失已经在上一篇说明；本篇只负责把确定的操作翻译成 MaixPy 调用。

## 一条便于观察的 MaixPy 处理链

下面的阈值和面积只用于展示代码结构，不是可直接用于项目的参数。先用自己的静态图片测量 LAB 范围，再放进实时循环。

```python
from maix import app, camera, display, image

# Example ranges only; measure them from the actual scene.
TARGET_LAB = [[20, 80, -20, 20, 20, 80]]
WHITE_LAB = [[90, 100, -10, 10, -10, 10]]

cam = camera.Camera(320, 240, image.Format.FMT_RGB888)
disp = display.Display()

while not app.need_exit():
    frame = cam.read()

    # Preserve the camera frame for drawing and side-by-side debugging.
    filtered = frame.copy()
    filtered.median(1)

    binary = filtered.binary(TARGET_LAB, copy=True)
    binary.close(1)

    blobs = binary.find_blobs(
        WHITE_LAB,
        pixels_threshold=100,
        area_threshold=100,
        merge=False,
    )

    for blob in blobs:
        x, y, w, h = blob.rect()
        frame.draw_rect(x, y, w, h, image.COLOR_RED)

    disp.show(frame)
```

先分别显示 `frame`、`filtered` 和 `binary`，确认每一步符合预期后，再保留形态学与候选筛选。示例使用 `merge=False`，是为了避免一开始就把相邻候选合并；是否需要合并由任务决定。

## 什么时候可以直接使用 `find_blobs()`

`find_blobs()` 已经封装了按阈值寻找连通色块并返回 `Blob` 对象的过程。它不是 OpenCV `findContours()` 的同名替代，也不会返回完整的轮廓点数组。

颜色范围清楚、画面较干净时，可以直接在原图上查找：

```python
blobs = frame.find_blobs(
    TARGET_LAB,
    roi=[40, 30, 240, 160],
    pixels_threshold=100,
    area_threshold=100,
    merge=False,
)
```

只有当阈值图确实出现需要修补的散点、孔洞或断裂时，才先 `binary()` 和形态学，再在白色区域中查找。`roi` 的格式是 `[x, y, w, h]`；`pixels_threshold` 按有效像素数量过滤，`area_threshold` 按外接区域面积过滤；`merge=True` 会合并满足相交与间距条件的候选。形态学和 `merge` 都可能造成合并，因此不要在没有中间图证据时同时加强它们。

## 怎样判断封装接口是否真的更合适

不要用“感觉更快”代替测量，也不要引用别人的固定帧率。分辨率、固件、函数组合和显示过程都会影响耗时。可以在同一设备、同一输入和同一分辨率下测量目标处理段：

```python
from maix import time

start = time.ticks_ms()
filtered = frame.copy()
filtered.median(1)
binary = filtered.binary(TARGET_LAB, copy=True)
elapsed_ms = time.ticks_ms() - start
print("processing_ms:", elapsed_ms)
```

比较时一次只替换一个实现，并把图像转换、算法处理和显示耗时分开记录。若 MaixPy 封装已经满足结果和时延要求，就没有必要为了统一写法转成 OpenCV。

## 什么时候再转到 OpenCV

下列情况可以考虑 OpenCV：MaixPy 没有对应算法；需要复杂的轮廓层级、特征组合或现成 OpenCV 模块；或者要让设备端与已经验证的 PC 方案保持一致。

先使用容易理解的复制方式：

```python
from maix import image

frame_bgr = image.image2cv(frame, ensure_bgr=True, copy=True)
```

`ensure_bgr=True` 会按 OpenCV 常用顺序准备数据；`copy=True` 会分配并复制缓冲区，安全但有开销。官方文档中的 `copy=False` 可以减少复制，却要求原图在数组使用期间保持有效，而且修改数组会影响原图；确认结果正确并测量到转换确实是瓶颈后，再研究这种共享内存写法。

## 学到什么程度算完成

完成本篇后，你应能：

- 根据上一篇的中间图结论，选择一个对应的 MaixPy 方法，而不是重新猜算法。
- 解释 `size=1` 与 OpenCV 3×3 核的关系，并知道两端结果仍需实测对比。
- 在处理前复制图像，分别观察原图、滤波图和二值图。
- 判断应直接使用 `find_blobs()`，还是先生成二值图并做一次有依据的形态学修补。
- 在转用 OpenCV 前说明缺少什么能力，并测量转换和处理各自的耗时。

## API 核对入口

本文接口与参数根据 MaixPy v4 官方文档核对，核对日期为 2026-09-17。固件升级后，实际写代码前仍应重新查看：

- [MaixPy `maix.image` API](https://wiki.sipeed.com/maixpy/api/maix/image.html)
- [MaixCAM 使用 OpenCV 的官方说明](https://wiki.sipeed.com/maixpy/doc/zh/vision/opencv.html)
- [本仓库的找色块最小例程](../../配套资源/示例代码/maixpy/find_color_blocks.py)
