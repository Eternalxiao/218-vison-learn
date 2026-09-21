# MaixPy 重点示例

## 建议先运行

- `find_color_blocks.py`：最小找色块闭环，输入为摄像头图像，输出为画框、中心坐标和 FPS。
- `yolo11_infer.py`：加载配套 `.mud + .cvimodel`，显示检测框、类别和置信度。

这两个示例需要复制到 MaixCAM 设备中运行。先确认当前 MaixPy 文档中的 API 与设备固件一致，再修改阈值、分辨率和模型路径。

## 整理的工具

- `MaixCAM--note.py`：个人学习时摄像头、绘图和常见图像函数的学习记录。
- `MyGetPixel.py`：交互取色工具。 -- 一般不会用到
- `MyOpenCV.py`：MaixPy 图像与 OpenCV 处理的封装尝试。 -- 初学时的个人对一些算法的复现只作为学习理解使用
- `MyUART.py`：多种串口发送格式的历史实现。
- `MyUI.py`：设备触摸界面封装。只例举了最常使用组件的个人封装实现，希望深入学习可以参考项目[MaixPy-UI-Lib](https://github.com/aristorechina/MaixPy-UI-Lib.git)

这些文件保留了不是统一设计的库。阅读时关注数据流和具体函数，不建议整目录复制到新项目。
