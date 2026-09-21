# ============================================================
# coordinate_validator.py - 像素坐标 → 机械坐标: 步骤、公式与校验
#
# 完整流程分四步 (对照 学习路线/04-坐标转换、标定与现场调试/
# 01-坐标标定.md):
#   ① 读入已知点: CSV 每行 = 检测像素 (u, v) + 实测机械坐标
#   ② 畸变矫正 (可选): 把检测像素从镜头拉歪的位置换回无畸变位置
#   ③ 像素 → 机械: 按轴交换、比例、方向符号、中心偏移做映射
#   ④ 残差汇总: 每点 dx/dy/欧氏误差, 全场 RMSE 和最大误差
#
# ③ 的映射公式 (注意 v 算 x、u 算 y, 因为安装关系含约 90° 旋转):
#   x = offset_x + x_sign × (v' - center_v) / px_per_mm
#   y = offset_y + y_sign × (u' - center_u) / px_per_mm
#
# ② 用只含 k1/k2 的简化径向畸变模型, 详细推导见 undistort_point
# 的注释。它等价于"先矫正整幅图像、再做检测"的点级版本: 图像级
# 矫正把每个像素按同一模型重采样, 点级矫正是把检测出来的 (u, v)
# 直接换回无畸变位置, 对同一个点二者结果一致。
#
# 用法: 修改下方调参区, 然后直接运行  python coordinate_validator.py
#   - sample_points.csv: 与映射公式完全一致, 畸变关闭时误差应为 0
#   - sample_points_distorted.csv: 同一批理想点预先套上了 k1=-3e-7
#     的桶形畸变。先保持 DIST_K1=0 运行, 会看到中心点仍准、误差随
#     离中心距离增大; 再把 DIST_K1 改成 -3e-7 复验, RMSE 应回到约 0。
# ============================================================

import csv
import math
from dataclasses import dataclass
from pathlib import Path

# ==================== 调参区 ====================
POINTS_FILE = Path("sample_points.csv")  # 已知点清单, 列: u, v, measured_x, measured_y

# --- ② 畸变矫正参数 (k1=k2=0 表示关闭这一步, 像素原样进入映射) ---
DIST_K1 = 0.0         # 一阶径向畸变系数, 量纲约为 1/像素²; 桶形畸变为负
DIST_K2 = 0.0         # 二阶径向畸变系数, 留 0 即只用 k1 一项
DIST_CENTER_U = 320.0 # 畸变中心 u: 未做完整相机标定时近似取图像中心
DIST_CENTER_V = 224.0 # 畸变中心 v (它是镜头主点概念, 与下面的映射参考中心不是一回事)

# --- ③ 像素→机械映射参数 (标定改这里, 不要改公式) ---
PX_PER_MM = 2.0    # 每毫米像素数, 由已知边长方块的像素边长 ÷ 物理边长得到
CENTER_U  = 320.0  # 图像参考中心 u (像素)
CENTER_V  = 224.0  # 图像参考中心 v (像素)
OFFSET_X  = 271.0  # 图像中心对应的机械 X (mm)
OFFSET_Y  = 221.0  # 图像中心对应的机械 Y (mm)
X_SIGN    = -1     # X 方向符号, 只能取 -1 或 1
Y_SIGN    = 1      # Y 方向符号, 只能取 -1 或 1
# ================================================


def undistort_point(u, v, k1, k2, center_u, center_v):
    """② 畸变矫正: 把有畸变图像上检测到的 (u, v) 换回无畸变像素位置。

    镜头造成的径向畸变有一个正向模型 (无畸变 → 有畸变):
        x  = u - center_u,  y  = v - center_v    (平移到以畸变中心为原点)
        r2 = x² + y²                            (该点到中心的距离平方)
        x_d = x × (1 + k1·r2 + k2·r2²)          (同一放大系数乘在两个分量上,
        y_d = y × (1 + k1·r2 + k2·r2²)           离中心越远被挪动得越多)
        有畸变像素 = (center_u + x_d, center_v + y_d)

    k1 < 0 是桶形畸变: 边缘的点被向中心压缩, 画面里的直线在边缘外凸;
    k1 > 0 是枕形畸变, 挪动方向相反。这就是"中心准、边缘放射状渐偏"
    误差的来源——沿半径方向、大小随 r2 增长。

    本函数要求解的是反问题 (有畸变 → 无畸变): 已知 x_d, 求 x。把正向
    式改写成 x = x_d / (1 + k1·r2 + k2·r2²) 后, 右边仍含未知的 r2, 没有
    闭式解, 用不动点迭代: 先拿 x_d 近似当 x 算出系数, 代回更新 x, 重复
    到数值不再变化。畸变不重时几轮就收敛, 这里固定迭代 20 轮。
    """
    x_d = u - center_u
    y_d = v - center_v
    x, y = x_d, y_d
    for _ in range(20):
        r2 = x * x + y * y
        factor = 1 + k1 * r2 + k2 * r2 * r2
        x = x_d / factor
        y = y_d / factor
    return center_u + x, center_v + y


@dataclass(frozen=True)
class Mapping:
    """③ 像素→机械映射: 一组标定参数 + 一个固定公式。

    px_per_mm 统一作用于两个轴, 隐含"相机垂直于工作平面、无透视"的
    前提; x_sign/y_sign 只能取 ±1, 处理机械轴与图像轴方向相反的情况。
    """

    px_per_mm: float
    center_u: float
    center_v: float
    offset_x: float
    offset_y: float
    x_sign: int
    y_sign: int

    def __post_init__(self):
        if self.px_per_mm <= 0:
            raise ValueError("px_per_mm must be positive")
        if self.x_sign not in {-1, 1} or self.y_sign not in {-1, 1}:
            raise ValueError("x_sign and y_sign must be -1 or 1")

    def pixel_to_machine(self, u, v):
        # 读法以 x 为例, 从内往外: v - center_v 得"离中心多少像素",
        # ÷ px_per_mm 得"离中心多少毫米", × x_sign 换到机械正方向,
        # + offset_x 平移到机械原点。y 同理, 但由 u 参与——轴交换,
        # 因为相机相对机械转了约 90° 安装。
        x = self.offset_x + self.x_sign * (v - self.center_v) / self.px_per_mm
        y = self.offset_y + self.y_sign * (u - self.center_u) / self.px_per_mm
        return x, y


def read_points(path):
    points = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            points.append(tuple(float(row[key]) for key in ("u", "v", "measured_x", "measured_y")))
    if not points:
        raise ValueError("point file contains no rows")
    return points


def validate_points(mapping, points):
    """④ 残差校验: 对每个已知点比较预测坐标与实测坐标。

    dx/dy 保留符号, 能看出每个点往哪边偏; error 是该点的欧氏距离
    误差, RMSE 反映总体精度, MAX 暴露最差的点。残差形态怎么读
    (整体平移/随距离增长/镜像) 见 01-坐标标定.md 的"怎么看残差"。
    输入的 points 应当是已经过 ② 矫正的像素坐标 + 实测机械坐标。
    """
    results = []
    for u, v, measured_x, measured_y in points:
        predicted_x, predicted_y = mapping.pixel_to_machine(u, v)
        dx = predicted_x - measured_x
        dy = predicted_y - measured_y
        results.append(
            {
                "u": u,
                "v": v,
                "predicted_x": predicted_x,
                "predicted_y": predicted_y,
                "dx": dx,
                "dy": dy,
                "error": math.hypot(dx, dy),
            }
        )
    rmse = math.sqrt(sum(item["error"] ** 2 for item in results) / len(results))
    maximum = max(item["error"] for item in results)
    return results, rmse, maximum


def main():
    # ③ 的映射参数来自标定, 整个校验过程不再改动它们
    mapping = Mapping(
        PX_PER_MM,
        CENTER_U,
        CENTER_V,
        OFFSET_X,
        OFFSET_Y,
        X_SIGN,
        Y_SIGN,
    )
    # ① 读入已知点: (u, v) 是有畸变图像上的检测值
    points = read_points(POINTS_FILE)
    # ② 先矫正像素, 再进入映射; 关闭时 (k1=k2=0) 矫正后即原值
    if DIST_K1 == 0 and DIST_K2 == 0:
        print("distortion correction: OFF (k1=0, k2=0)")
    else:
        print(f"distortion correction: ON k1={DIST_K1:g} k2={DIST_K2:g} center=({DIST_CENTER_U:g},{DIST_CENTER_V:g})")
    corrected = []
    for u, v, measured_x, measured_y in points:
        cu, cv = undistort_point(u, v, DIST_K1, DIST_K2, DIST_CENTER_U, DIST_CENTER_V)
        corrected.append((cu, cv, measured_x, measured_y))
    # ③④ 用矫正后的点做映射并与实测对比
    results, rmse, maximum = validate_points(mapping, corrected)
    for (u, v, _, _), item in zip(points, results):
        print(
            "u={u:.1f} v={v:.1f} corrected=({cu:.1f},{cv:.1f}) predicted=({predicted_x:.3f},{predicted_y:.3f}) "
            "residual=({dx:.3f},{dy:.3f}) error={error:.3f} mm".format(cu=item["u"], cv=item["v"], **item)
        )
    print(f"RMSE={rmse:.3f} mm")
    print(f"MAX={maximum:.3f} mm")


if __name__ == "__main__":
    main()
