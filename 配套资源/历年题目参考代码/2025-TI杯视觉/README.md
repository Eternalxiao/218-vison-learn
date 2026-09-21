# 2025 TI 杯视觉代码

这是 2025 期间编写和整理的 MaixCAM 视觉侧代码。

## 文件入口

- `main.py`：取图、状态切换、检测和显示的主循环。
- `SimpleShapeDetector.py`：基础几何形状检测。
- `OverRectLapDetector.py`：重叠矩形相关判断。
- `calibration_manager.py`：标定参数与像素/距离关系的管理。
- `constants.py`：分辨率、阈值和其他现场参数。
- `MyUI.py`：MaixCAM 触摸显示界面。
- `MyUART.py`：视觉结果发送。

## 适合学习什么

这个目录适合观察一个视觉工程如何从单个检测函数逐渐扩展成取图、标定、识别、显示和发送组成的程序。阈值、尺寸、距离和路径是当时环境参数，换相机位置或分辨率后必须重新验证。

推荐先读[OpenCV 传统视觉](../../../学习路线/02-OpenCV传统视觉/README.md)，再从 `main.py` 的图像读取位置沿调用关系阅读。
