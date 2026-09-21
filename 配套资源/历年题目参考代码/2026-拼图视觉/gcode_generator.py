"""
G代码生成器 - 统一仿射变换版
像素(u,v) -> 机械坐标(x,y) 一步完成:
  x_robot = offset_x + (v - center_v) / px_per_mm
  y_robot = offset_y + (u - center_u) / px_per_mm
轴交换(v->x, u->y)隐含90度坐标系旋转
"""


class GCodeGenerator:

    def __init__(self, cfg):
        cal = cfg["calibration"]
        mech = cfg["mechanical"]

        # 统一仿射参数 (3个字段, 5个数值)
        self.px_per_mm = cal["px_per_mm"]
        self.center_u = cal["pixel_center"][0]
        self.center_v = cal["pixel_center"][1]
        self.offset_x = cal["robot_offset"][0]
        self.offset_y = cal["robot_offset"][1]
        # 符号翻转: 相机v->机械x / 相机u->机械y 的正方向由物理安装决定
        # 若PNT校验发现上下/左右点镜像偏移, 用Offset页X±/Y±翻转
        self.x_sign = cal.get("x_sign", 1)
        self.y_sign = cal.get("y_sign", 1)

        # 机械参数
        self.feed_rate = mech["feed_rate"]
        self.z_safe = mech["z_up"]       # 安全高度 (负值, 如-1)
        self.z_work = mech["z_down"]     # 工作高度 (0)
        self.e_home = mech.get("e_home", 90.0)

        # G代码缓冲
        self.gcode_lines = []

    # ==================== 坐标转换 (核心) ====================

    def pixel_to_robot(self, u, v):
        """
        像素(u,v) -> 机械坐标mm(x,y)
        统一仿射: 缩放 + 90度轴交换 + 平移, 一步到位
        """
        mm_per_px = 1.0 / self.px_per_mm
        x = self.offset_x + self.x_sign * (v - self.center_v) * mm_per_px
        y = self.offset_y + self.y_sign * (u - self.center_u) * mm_per_px
        return (x, y)

    # ==================== 基础指令 ====================

    def start_gcode(self):
        """上电初始化: 回零 + 绝对定位 + 舵机归位"""
        self.gcode_lines = [
            "G92 X0 Y0 Z0",
            "G90",
            "G1 Z%.1f F%d" % (self.z_safe, self.feed_rate),
            "G4 P500",
            "M280 P0 S%.0f" % self.e_home,
        ]

    def end_gcode(self):
        """完成: 抬到安全高度 + 回零"""
        self.gcode_lines.extend([
            "M280 P0 S%.0f" % self.e_home,
            "G1 X0 Y0 Z%.1f F%d" % (self.z_safe, self.feed_rate),
            "M400",
        ])

    def move_to(self, x, y):
        """移动到机械坐标(mm)"""
        self.gcode_lines.append("G1 X%.3f Y%.3f F%d" % (x, y, self.feed_rate))
        self.gcode_lines.append("M400")

    def move_to_pixel(self, u, v):
        """像素坐标 -> 机械坐标 -> 移动"""
        x, y = self.pixel_to_robot(u, v)
        self.move_to(x, y)

    # ==================== Z轴 ====================

    def lower(self):
        """下压到工作高度"""
        self.gcode_lines.append("G1 Z%.1f F%d" % (self.z_work, self.feed_rate))
        self.gcode_lines.append("M400")

    def raise_up(self):
        """抬到安全高度"""
        self.gcode_lines.append("G1 Z%.1f F%d" % (self.z_safe, self.feed_rate))
        self.gcode_lines.append("M400")

    # ==================== 电磁铁 ====================

    def magnet_on(self):
        """吸附上电"""
        self.gcode_lines.append("M106 S255")
        self.gcode_lines.append("G4 P500")

    def magnet_off(self):
        """释放"""
        self.gcode_lines.append("M107")
        self.gcode_lines.append("G4 P500")

    # ==================== 旋转 (舵机) ====================

    def _servo_angle_seq(self, delta_theta):
        """
        将旋转偏移delta_theta(度,±180)转为舵机绝对角度序列
        返回: [angle] 或 [limit_angle, final_angle] (超范围分步)
        """
        if abs(delta_theta) <= 90:
            return [self.e_home + delta_theta]
        elif delta_theta > 90:
            return [180.0, delta_theta]
        else:
            return [0.0, delta_theta + 180.0]

    def rotate(self, delta_theta):
        """
        旋转碎片, 超范围自动分步(放下-归零-重取)
        调用时物体应已被吸住并抬起
        """
        angles = self._servo_angle_seq(delta_theta)
        if len(angles) == 1:
            self.gcode_lines.append("M280 P0 S%.3f" % angles[0])
            self.gcode_lines.append("G4 P500")
        else:
            # 第一步: 转到极限
            self.gcode_lines.append("M280 P0 S%.3f" % angles[0])
            self.gcode_lines.append("G4 P500")
            # 放下 -> 释放 -> 抬起
            self.lower()
            self.magnet_off()
            self.raise_up()
            # 舵机归零
            self.gcode_lines.append("M280 P0 S%.0f" % self.e_home)
            self.gcode_lines.append("G4 P500")
            # 重新吸取
            self.lower()
            self.magnet_on()
            self.raise_up()
            # 第二步: 转到剩余角度
            self.gcode_lines.append("M280 P0 S%.3f" % angles[1])
            self.gcode_lines.append("G4 P500")

    # ==================== 组合动作 ====================

    def pick_place_rotate(self, from_px, to_px, delta_theta):
        """
        完整动作: 抓取 -> 旋转 -> 放置
        from_px: (u, v) 碎片像素坐标
        to_px:   (u, v) 目标像素坐标
        delta_theta: 度, 旋转量
        """
        self.move_to_pixel(from_px[0], from_px[1])
        self.lower()
        self.magnet_on()
        self.raise_up()
        self.rotate(delta_theta)
        self.move_to_pixel(to_px[0], to_px[1])
        self.lower()
        self.magnet_off()
        self.raise_up()
        # place后舵机归零: 保证下一片拾取时从 e_home 起始,
        # 避免携带上一片的旋转角度 (M280为绝对指令, 多一条指令换确定性)
        self.gcode_lines.append("M280 P0 S%.0f" % self.e_home)
        self.gcode_lines.append("G4 P500")

    # ==================== 标定辅助 ====================

    def calib_move(self):
        """
        标定用: 移动到robot_offset位置(像素中心对应的物理点)
        注意: 不含G92, 原点由Home(start_gcode)统一建立, 避免重复MOVE累积跑偏
        使用前必须先按一次 Home
        """
        self.gcode_lines = [
            "G90",
            "G1 Z%.1f F%d" % (self.z_safe, self.feed_rate),
            "M400",
            "G1 X%.3f Y%.3f F%d" % (self.offset_x, self.offset_y, self.feed_rate),
            "M400",
        ]

    def calib_point(self, u, v):
        """PNT多点校验: 移动到像素(u,v)对应的机械坐标 (使用前必须先按Home)"""
        x, y = self.pixel_to_robot(u, v)
        self.gcode_lines = [
            "G90",
            "G1 Z%.1f F%d" % (self.z_safe, self.feed_rate),
            "M400",
            "G1 X%.3f Y%.3f F%d" % (x, y, self.feed_rate),
            "M400",
        ]

    # ==================== 工具 ====================

    def add_comment(self, comment):
        self.gcode_lines.append("; %s" % comment)

    def clear(self):
        self.gcode_lines = []

    def get_gcode(self):
        return "\n".join(self.gcode_lines)
