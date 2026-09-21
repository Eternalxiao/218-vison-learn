# ============================================================
# gcode_demo.py - 安全生成 G-code 帧文本 (不打开串口)
# 把一个像素点按标定参数换算成机械坐标, 生成一次移动的 G-code,
# 按 M400/G4 边界分块, 并加上 EEEF/FFFE 帧头帧尾, 只打印不发送。
# 用法: 修改下方调参区, 然后直接运行  python gcode_demo.py
# ============================================================

from coordinate_validator import Mapping

# ==================== 调参区 ====================
U = 340.0        # 目标像素 u (横坐标)
V = 244.0        # 目标像素 v (纵坐标)
FEED_RATE = 3000 # G1 进给速度

# --- 像素→机械坐标映射参数 (与 coordinate_validator.py 保持一致) ---
PX_PER_MM = 2.0    # 每毫米像素数
CENTER_U  = 320.0  # 图像中心 u (像素)
CENTER_V  = 224.0  # 图像中心 v (像素)
OFFSET_X  = 271.0  # 图像中心对应的机械 X (mm)
OFFSET_Y  = 221.0  # 图像中心对应的机械 Y (mm)
X_SIGN    = -1     # X 方向符号, 只能取 -1 或 1
Y_SIGN    = 1      # Y 方向符号, 只能取 -1 或 1
# ================================================


def generate_move(mapping, u, v, feed_rate):
    x, y = mapping.pixel_to_machine(u, v)
    return ["G90", f"G1 X{x:.3f} Y{y:.3f} F{feed_rate}", "M400", "G4 P500"]


def split_blocks(lines):
    blocks = []
    current = []
    for line in lines:
        current.append(line)
        if line.startswith("M400") or line.startswith("G4"):
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def frame_block(block):
    return "EEEF\r\n" + "\r\n".join(block) + "\r\nFFFE\r\n"


def main():
    mapping = Mapping(
        PX_PER_MM,
        CENTER_U,
        CENTER_V,
        OFFSET_X,
        OFFSET_Y,
        X_SIGN,
        Y_SIGN,
    )
    for index, block in enumerate(split_blocks(generate_move(mapping, U, V, FEED_RATE)), 1):
        print(f"--- block {index} ---")
        print(frame_block(block), end="")


if __name__ == "__main__":
    main()
