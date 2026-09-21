"""
G代码生成器 - 支持像素坐标转换与串口发送
适用于 MaixPy 环境，控制 XYZ 三轴机械臂 / SCARA 机械臂
"""

from maix import uart, time
import json
import math

class GCodeGenerator:
    """
    G代码生成器，支持：
    - 像素坐标到实际坐标的线性映射（标定）
    - 生成 G0/G1 移动指令
    - 抬刀/下刀（Z轴控制）
    - 通过串口发送 G 代码
    - 保存 G 代码到文件（可选）
    - SCARA 机械臂角度模式逆解（可选）
    """

    def __init__(self,
                 feed_rate=1000,
                 z_up=5.0,
                 z_down=80.0,
                 Z_start = 70.0,
                 x_range=(0.0, 200.0),
                 y_range=(0.0, 200.0),
                 pixel_range=(0, 320, 0, 224),

                 # === SCARA 参数===
                 link1=125.0,
                 link2=125.0,
                 offset_x=60.0,
                 offset_y=-100.0,
                 axis_scaling=(1.0, 1.0, 1.0),
                 angle_mode=False):
        """
        初始化生成器
        :param serial_port: 串口设备路径
        :param baudrate: 波特率
        :param feed_rate: 进给速度 (mm/min)
        :param z_up: 抬刀高度 (mm)
        :param z_down: 下刀深度 (mm)
        :param x_range: 实际 X 轴范围 (min, max)
        :param y_range: 实际 Y 轴范围 (min, max)
        :param pixel_range: 像素范围 (min_u, max_u, min_v, max_v)
        :param link1: SCARA 主臂长度 (mm)
        :param link2: SCARA 副臂长度 (mm)
        :param offset_x: SCARA 原点 X 偏移 (mm)
        :param offset_y: SCARA 原点 Y 偏移 (mm)
        :param axis_scaling: 各轴缩放系数 (x_scale, y_scale, z_scale)
        :param angle_mode: 是否使用角度模式（True: 输出关节角度 G95；False: 传统笛卡尔坐标）
        """

        self.feed_rate = feed_rate
        self.Z_start = Z_start
        self.z_up = z_up
        self.z_down = z_down

        # 标定参数（像素 -> 实际坐标）
        self.pixel_min_u = pixel_range[0]
        self.pixel_max_u = pixel_range[1]
        self.pixel_min_v = pixel_range[2]
        self.pixel_max_v = pixel_range[3]
        self.real_min_x = x_range[0]
        self.real_max_x = x_range[1]
        self.real_min_y = y_range[0]
        self.real_max_y = y_range[1]

        # SCARA 参数
        self.link1 = link1
        self.link2 = link2
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.axis_scaling = axis_scaling
        self.angle_mode = angle_mode

        # G代码存储
        self.gcode_lines = [
            "G28 XYZ",
            "G90",
            f"G1 F{self.feed_rate}",
            f"G1 Z{self.Z_start}",
            "M400",
            "G92 X-15 Y18"
            
        ]

    # ==================== 坐标转换 ====================
    def set_calibration(self,
                        pixel_min_u, pixel_max_u, real_min_x, real_max_x,
                        pixel_min_v, pixel_max_v, real_min_y, real_max_y):
        """
        手动设置标定参数（线性映射）
        """
        self.pixel_min_u = pixel_min_u
        self.pixel_max_u = pixel_max_u
        self.real_min_x = real_min_x
        self.real_max_x = real_max_x
        self.pixel_min_v = pixel_min_v
        self.pixel_max_v = pixel_max_v
        self.real_min_y = real_min_y
        self.real_max_y = real_max_y

    def pixel_to_real(self, u, v, z=0.0):
        """
        将像素坐标 (u, v) 转换为实际坐标 (x, y, z)
        线性映射公式：real = min_real + (pixel - min_pixel) / (max_pixel - min_pixel) * (max_real - min_real)
        :param u: 像素列坐标（X方向）
        :param v: 像素行坐标（Y方向）
        :param z: 自定义 Z 轴高度（若不提供则使用当前下刀/抬刀高度）
        :return: (x, y, z)
        """
        # 处理除零情况（若像素范围为零，则直接返回中点）
        if self.pixel_max_u - self.pixel_min_u == 0:
            x = (self.real_min_x + self.real_max_x) / 2.0
        else:
            x = self.real_min_x + (u - self.pixel_min_u) / (self.pixel_max_u - self.pixel_min_u) * (self.real_max_x - self.real_min_x)

        if self.pixel_max_v - self.pixel_min_v == 0:
            y = (self.real_min_y + self.real_max_y) / 2.0
        else:
            y = self.real_min_y + (v - self.pixel_min_v) / (self.pixel_max_v - self.pixel_min_v) * (self.real_max_y - self.real_min_y)

        print(f"将像素坐标 (u, v) 转换为实际坐标 (x, y, z): {u=},{v=},{x=},{y=},{z=}")
        return (x, y, z)

    # ==================== SCARA 逆解 ====================
    def scara_inverse_kinematics(self, x, y):
        """
        将绝对坐标 (x, y) 转换为 SCARA 关节角度 (theta, psi)
        :param x: 目标 X 坐标（mm）
        :param y: 目标 Y 坐标（mm）
        :return: (theta_deg, psi_deg) 角度值（度）
        """
        # 转换为假定坐标系（相对于旋转中心）
        px =  -x * self.axis_scaling[0] + self.offset_x
        py =  y * self.axis_scaling[1] + self.offset_y

        L1 = self.link1
        L2 = self.link2
        L1_2 = L1 * L1
        L2_2 = L2 * L2

        # 计算副臂角度余弦值
        if abs(L1 - L2) < 1e-6:
            C2 = (px*px + py*py) / (2.0 * L1_2) - 1.0
        else:
            C2 = (px*px + py*py - L1_2 - L2_2) / (2.0 * L1 * L2)

        # 防止浮点误差导致越界
        C2 = max(-1.0, min(1.0, C2))
        S2 = math.sqrt(1.0 - C2*C2)

        psi = math.atan2(S2, C2)
        psi_deg = psi * 180.0 / math.pi

        K1 = L1 + L2 * C2
        K2 = L2 * S2
        theta = math.atan2(px, py) - math.atan2(K1, K2)
        theta_deg = -theta * 180.0 / math.pi

        return (theta_deg, 180-psi_deg)

    def set_angle_mode(self, enabled):
        """动态切换角度模式"""
        self.angle_mode = enabled

    # ==================== G代码生成 ====================
    def start_gcode(self, origin_x=0.0, origin_y=0.0, origin_z=0.0):
        """
        生成 G 代码起始部分
        :param origin_x, origin_y, origin_z: 设置当前点为原点 (G92)
        """
        self.gcode_lines = []

    def add_comment(self, comment):
        """添加注释行"""
        self.gcode_lines.append(f"; {comment}")

    def add_gcode_line(self, line):
        """添加自定义 G 代码行"""
        self.gcode_lines.append(line)

    def move_to(self, x, y, z=None, is_cutting=False):
        """
        移动到指定实际坐标
        :param x, y: 目标坐标 (mm)
        :param z: 目标 Z 坐标，若为 None 则根据 is_cutting 决定下刀或保持当前
        :param is_cutting: 是否为切削移动（若为 True 且未指定 z，则移动到 z_down）
        """
        if z is None:
            z = self.z_down if is_cutting else self.z_up

        if self.angle_mode:
            # 角度模式：先逆解得到关节角度，再生成 G 代码
            print(f"准备逆向解析世界坐标{x=},{y=}")
            theta, psi = self.scara_inverse_kinematics(x, y)
            self.gcode_lines.append(f"G1 X{theta:.3f} Y{psi:.3f}")
            self.gcode_lines.append("M400")
            print("添加G代码：" + f"G1 X{theta:.3f} Y{psi:.3f} Z{z:.3f}")
        else:
            # 笛卡尔模式
            if is_cutting:
                # 切削移动：使用 G1 直线插补，下刀到 z_down
                self.gcode_lines.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f}")
            else:
                # 快速定位：先抬刀再移动
                self.gcode_lines.append(f"G0 Z{self.z_up}")
                self.gcode_lines.append(f"G0 X{x:.3f} Y{y:.3f}")
                if z != self.z_up:
                    self.gcode_lines.append(f"G0 Z{z:.3f}")

    def move_to_pixel(self, u, v, z=None, is_cutting=False):
        """
        通过像素坐标移动到实际位置
        """
        x, y, _ = self.pixel_to_real(u, v)
        self.move_to(x, y, z, is_cutting)

    def pen_down(self):
        """下刀（移动到 z_down）"""
        self.gcode_lines.append(f"G1 Z{self.z_down:.3f}")
        self.gcode_lines.append("M400")
        self.gcode_lines.append("M150 U255")
        self.gcode_lines.append("G4 P500")

    def pen_up(self):
        """抬刀（移动到 z_up）"""
        self.gcode_lines.append(f"G1 Z{self.z_up:.3f}")

    def pen_drop(self):
        """放棋子"""
        self.gcode_lines.append("M150 U0")
        self.gcode_lines.append("G4 P500")

    def end_gcode(self, home_x=0.0, home_y=0.0):
        """
        生成 G 代码结束部分
        :param home_x, home_y: 回原点位置
        """
        self.gcode_lines.extend([
            f"G1 Z{self.Z_start}",
            "M400",
            "G28 XY",
            "G92 X-15 Y18"
        ])
    
    # ==================== 文件保存 ====================
    def save_gcode(self, filename):
        """
        将生成的 G 代码保存到文件
        """
        try:
            with open(filename, 'w') as f:
                f.write("\n".join(self.gcode_lines))
            print(f"G 代码已保存到: {filename}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False

    # ==================== 调试与重置 ====================
    def clear(self):
        """清空当前 G 代码缓冲区"""
        self.gcode_lines = []

    def get_gcode(self):
        """返回当前 G 代码字符串（用于调试）"""
        return "\n".join(self.gcode_lines)
