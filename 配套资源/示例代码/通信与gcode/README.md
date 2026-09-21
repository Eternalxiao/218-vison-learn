# 通信、坐标与 G-code 示例

## 数据格式回环

```powershell
python protocol.py
```

`protocol.py` 演示逗号字符串、换行 JSON 和教学用二进制帧。二进制解析器可以连续接收不完整片段，并从缓冲区中依次取出完整帧。

## 坐标校验

`coordinate_validator.py` 把"像素坐标 → 机械坐标"摊成可单独运行的四步：① 读入已知点（检测像素 + 实测机械坐标）→ ② 畸变矫正（可选）→ ③ 按轴交换、比例、方向符号、中心偏移做映射 → ④ 残差汇总。映射公式、每个参数怎么标定出来、残差形态怎么读，见[01-坐标标定](../../../学习路线/04-坐标转换、标定与现场调试/01-坐标标定.md)；② 使用的简化径向畸变模型与 MaixPy `lens_corr`、OpenCV 完整内参标定的边界，见[02-相机畸变矫正](../../../学习路线/04-坐标转换、标定与现场调试/02-相机畸变矫正.md)。

打开 `coordinate_validator.py`，把调参区的 `POINTS_FILE`、畸变参数（`DIST_K1`、`DIST_K2`、`DIST_CENTER_U`、`DIST_CENTER_V`）和映射参数（`PX_PER_MM`、`CENTER_U`、`CENTER_V`、`OFFSET_X`、`OFFSET_Y`、`X_SIGN`、`Y_SIGN`）换成自己的标定值，然后直接运行：

```powershell
python coordinate_validator.py
```

输出包括每点的原始像素、矫正后像素、预测坐标、`dx/dy`、欧氏误差、RMSE 和最大误差。仓库带两份点表：

- `sample_points.csv` 与映射公式完全一致，畸变关闭时误差应为 0；换成实测数据后才有校准意义。
- `sample_points_distorted.csv` 是同一批理想点预先套上 `k1=-3e-7` 桶形畸变后的检测值。把 `POINTS_FILE` 指向它并保持 `DIST_K1=0` 运行，可以看到中心点仍准、误差随离中心距离增大；再把 `DIST_K1` 改成 `-3e-7` 复验，RMSE 应回到约 0。这组对照就是"误差随离中心距离增大 → 检查畸变"的可运行版本。

## 安全生成 G-code

打开 `gcode_demo.py`，把调参区的 `U`、`V` 和映射参数改成目标点，然后直接运行：

```powershell
python gcode_demo.py
```

程序只把帧文本打印到终端，不打开串口。`M106/M107/M280` 等项目相关命令没有放入这个最小示例。

## 自动测试

```powershell
python -m unittest test_protocol_and_coordinate.py
```
