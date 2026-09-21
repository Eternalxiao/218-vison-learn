from http.client import PRECONDITION_FAILED

from maix import uart, pinmap
import json
from constants import ERRORS
import gcode_generator

class MyUtils:
    """
        自定义的串口类
    """

    def __init__(self):

        device = "/dev/ttyS0"   # usb转接口 - 反接
        self.serial0 = uart.UART(device, 115200)

        # 创建G代码生成器（配置像素坐标范围、世界坐标、SCARA参数）
        self.gcode = gcode_generator.GCodeGenerator(
            feed_rate=1000,
            Z_start = 60.0,
            z_up=68.0,
            z_down=78.0,
            x_range=(0, 270),
            y_range=(0, 185),
            pixel_range=(0, 320, 0, 224),

            # === SCARA 参数===
            link1=180.0,
            link2=180.0,
            offset_x=150.0,
            offset_y=150.0,
            angle_mode=True          # 启用角度模式
        )

        # 发送上电后的g代码
        self.send_gcode()
        
    def sending_coordinate_data(self, date_dir: dict) -> bool:
        """
            串口传输数据函数
            date_dir:   存有from和to数据字典，值为未修正的坐标
            通过串口得到并且发送 G 代码
            :param lines: 要发送的行列表，若为 None 则发送 self.gcode_lines
            :param delay_ms: 每行发送后的等待时间（毫秒），避免机械臂缓冲区溢出
            :return: 是否全部发送成功
        """
        
        # print(f"接收到{date_dir=}")
        new_data = self.modify_coordinates(date_dir, self.alter_coordinate)

        # 定义数据字典，使用 json 模式传输
        # new_data = {
        #     'from': [a, b],
        #     'to': [c, d]
        # }
        print(f"对像素数数据{new_data=}进行g代码生成并发送")

        # 1. 开始生成
        self.gcode.start_gcode(origin_x=0, origin_y=0, origin_z=0)

        # 2. 像素坐标 -> 移动（抬刀移动）
        self.gcode.move_to_pixel(new_data["from"][0], new_data["from"][1], is_cutting=False)

        # 3. 下刀并移动到另一个像素点
        self.gcode.pen_down()
        self.gcode.pen_up()
        self.gcode.move_to_pixel(new_data["to"][0], new_data["to"][1], is_cutting=True)

        # 4. 抬刀结束
        self.gcode.pen_drop()
        self.gcode.end_gcode(home_x=0, home_y=0)

        # 5. 准备发送
        self.send_gcode()


    def send_gcode(self):
        """发送g代码"""
        lines = self.gcode.gcode_lines

        if lines is None:
            lines = self.gcode_lines
        if not lines:
            print("没有 G 代码可发送")
            return False
        try:
            print("发送g代码\n")
            for line in lines:
                # 发送一行，末尾添加换行符
                print(line)
                self.serial0.write_str(line + "\n")
                # time.sleep_ms(delay_ms)
            print(f"\n成功发送 {len(lines)} 行 G 代码")
            self.gcode.clear()   # 清除缓存
            return True
        except Exception as e:
            print(f"串口发送失败: {e}")
            return False
        
    # 实现坐标的同步并且不修改字典的键值
    def modify_coordinates(self, original_dict: dict, adjust_func) -> dict:
        """
        修改字典中每个点的坐标，保持键不变
        :param original_dict: 原始字典
        :param adjust_func: 调整函数，输入为(x,y)，返回新坐标(new_x, new_y)
        :return: 修改后的新字典
        """
        #       {新键: 新值 for 键, 值 in 字典.items()}
        return {key: adjust_func(point) for key, point in original_dict.items()}

    # 坐标调整函数，输入为(x,y)，返回新坐标(new_x, new_y)
    def alter_coordinate(self, point:tuple) -> tuple:

        x_error, y_error = ERRORS
        x, y = point

        new_x = x - x_error
        new_y = y_error - y

        new_y = y
        
        return (new_x, new_y)
