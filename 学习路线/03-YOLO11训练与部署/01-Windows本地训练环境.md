# Windows 本地训练环境

这篇不提供一条对所有电脑都成立的 CUDA 安装命令。训练环境会同时受到显卡型号、驱动、Python、PyTorch 和 Ultralytics 版本影响；真正需要掌握的是怎样分层验证，而不是记住一次安装成功时的版本组合。

## 1. 先决定在哪里训练

本地训练适合反复查看图片、修改标签和做小规模实验，学习阶段主要在本地训练；云服务器适合没有可用 NVIDIA 显卡或需要较长训练时使用。无论选哪一种，代码、数据版本和命令都应能被记录并复现。

本仓库默认使用 Python，因为 Ultralytics 的训练、验证和导出接口都直接提供 Python API。MaixCAM Pro 板端使用 MaixPy，属于部署环境，不要把 PC 的 `pip` 包复制到板端。

## 2. 建立独立 Python 环境

> 这一步是为了环境隔离，建议使用 conda 进行不同环境的隔离，这不属于训练环境的搭建（使用base环境同样可以），如果你不懂什么是 conda，什么是python的环境隔离，请自行查找资料学习本文不深入讲解，请不要将 conda、cuda 等混为一谈。

在仓库根目录打开 PowerShell：

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 阻止激活脚本，可以直接使用 `.venv\Scripts\python.exe` 执行后续命令，而不是先关闭系统安全策略。

项目的 PC 依赖列在 [`requirements.txt`](../../requirements.txt)：NumPy、OpenCV、Pillow 和 Ultralytics。先根据 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择与驱动匹配的 PyTorch，再安装仓库依赖：

```powershell
python -m pip install -r requirements.txt
```

不要从几篇不同年份的教程各抄一条 CUDA、cuDNN 和 PyTorch 命令混装。官方页面生成的安装命令是当前版本关系的起点。

## 3. 分四层检查 GPU

### 驱动层

```powershell
nvidia-smi
```

能看到显卡、驱动版本和显存占用，说明操作系统能访问 NVIDIA 驱动；它不证明当前 Python 环境中的 PyTorch 已经安装了 CUDA 支持。

### Python 与 pip 层

```powershell
python --version
python -m pip --version
python -c "import sys; print(sys.executable)"
```

三条命令中的 Python 路径应指向同一个虚拟环境。常见问题是 `python` 与 `pip` 分别来自两个环境，所以安装成功后仍然导入失败。

### PyTorch 层

```powershell
python -c "import torch; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

- `torch.version.cuda` 是这个 PyTorch 安装包面向的 CUDA 运行时版本。
- `torch.cuda.is_available()` 才回答当前 Python 能否使用 CUDA。
- `nvcc --version` 只说明本机 CUDA Toolkit，不能替代上面的 PyTorch 检查。

### Ultralytics 层

```powershell
python -c "import ultralytics; print(ultralytics.__version__)"
```

把以上输出保存到实验记录。之后换电脑、换云服务器或升级包时，先比较环境信息，再判断是不是代码问题。

## 4. 先冒烟训练，不要直接长时间训练

```powershell
Set-Location 配套资源\示例代码\yolo11
```

教学脚本不带命令行参数：打开脚本，文件开头就是调参区，改完直接运行。先打开 [`train.py`](../../配套资源/示例代码/yolo11/train.py) 逐行看一遍调参区，确认每个参数的含义都说得清楚，脚本说明见[YOLO11 教学脚本 README](../../配套资源/示例代码/yolo11/README.md)。

准备好最小数据集后，把调参区改成冒烟配置——`EPOCHS = 3`、`BATCH = 4`、`WORKERS = 0`、`NAME = "smoke"`，直接运行：

```powershell
python train.py
```

它的目的不是得到可用模型，而是尽早发现路径、标签、显存和多进程问题。CPU 环境把 `DEVICE = "0"` 改为 `DEVICE = "cpu"`。

## 5. Windows 多进程为什么容易报错

Windows 创建数据加载进程时会重新导入主模块，因此训练入口必须放在：

```python
if __name__ == "__main__":
    main()
```

配套 [`train.py`](../../配套资源/示例代码/yolo11/train.py)已经这样组织。首次运行用 `WORKERS = 0`，数据与训练都正常后再改成 2、4 等较小值。`WORKERS` 的合适值取决于 CPU、内存和数据所在磁盘，不存在适用于所有 Windows 电脑的固定推荐值。

## 6. 显存不足时按顺序缩小问题

出现 CUDA out of memory 时，一次只改一个因素：

1. 先降低 `BATCH`。
2. 再确认没有其他程序占用大量显存。
3. 必要时使用更小的模型，例如 `yolo11n.pt`。
4. 最后才考虑降低训练 `IMGSZ`，因为它会同时改变小目标信息和计算量。

`CACHE = "ram"` 会把图片缓存进内存，不是显存；机器内存不足时改为 `"disk"` 或保持默认 `"none"`。先完成一轮基线训练，再单独比较每个加速选项的收益和资源占用。

## 7. 云服务器需要注意事项

虽然本文是教学如何搭建本地训练环境，但是需要强调的是这只是为学习了解整个流程，以及在平时学习期间避免不必要的支出。但实际比赛的时候建议租借高性能的服务器，因为本地训练的时间较长而且比赛期间电脑要用来改代码、建模型不适合用来跑训练任务。

这里给出四点使用云服务训练的建议

- 数据集不是越大越好，如果是比赛期间训练，建议不超过8k，因为下载上传给服务器要消耗大量时间
- 由于训练是长时间的后台跑训练，所以一定要使用`tmux、 screen、 nohup`等命令，让程序不随着终端关闭而停止执行，这一点非常重要！！！
- 比赛期间一定不要为了省一点钱提前释放服务器，因为训练完成后的第一步是把模型下载到本地，释放后就没有了。
- 建议比赛前使用一次云服务器提前适应整个流程（虽然很简单必要性不是很大）

把数据上传到云端。训练时至少考虑：

- 实例 GPU 型号和数量，建议双卡。
- 驱动、Python、PyTorch、Ultralytics 版本。
- 数据集版本、`data.yaml`、训练命令和随机种子。
- 输出目录以及下载回本地的 `args.yaml`、曲线、`last.pt`、`best.pt`。

云端训练结束前把产物下载并校验，一定不要把唯一权重留在临时实例磁盘上。

## 最小验收

1. 当前虚拟环境能导入 `torch` 和 `ultralytics`。
2. 能准确说明当前训练将使用 CPU 还是哪一张 GPU。
3. 能打开每个教学脚本，找到文件开头的调参区并说明每个参数的含义。
4. 用自己的最小数据集完成 1～3 epoch 冒烟训练。
5. `runs\train\smoke` 中能找到本次参数、图表以及 `weights\last.pt`、`weights\best.pt`。

安装参数会更新，以 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)和 [Ultralytics 快速开始](https://docs.ultralytics.com/quickstart/)为准。外部链接最后核验于 2026-09-17。
