# PC 端 OpenCV 特征处理链可视化

先生成一张练习图片（输出路径在 `make_sample_image.py` 调参区的 `OUTPUT` 里改）：

```powershell
python 配套资源\示例代码\opencv\make_sample_image.py
```

再执行处理链。输入图片、输出目录和处理链的全部参数都写在 `feature_pipeline.py` 开头的调参区（`INPUT_IMAGE`、`OUTPUT_DIR`、`CONFIG` 字典），改完直接运行：

```powershell
python 配套资源\示例代码\opencv\feature_pipeline.py
```

输出目录会保存原图、颜色空间、滤波、二值化、腐蚀、膨胀、开运算、闭运算和轮廓结果。调参时每次只改 `CONFIG` 里的一组参数，并比较最早发生变化的中间图。

`CONFIG["ranges"]` 是二维列表。HSV 每项为 `[Hmin, Smin, Vmin, Hmax, Smax, Vmax]`；LAB 每项为 `[Lmin, Amin, Bmin, Lmax, Amax, Bmax]`。
