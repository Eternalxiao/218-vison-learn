from maix import uart, pinmap
import json
from constants import ERRORS

class MyUtils:
    """
        自定义的串口类
    """
    def sending_coordinate_data(self, date_dir: dict) -> bool:
        """
            串口传输数据函数
            date_dir:   存有from和to数据字典，值为未修正的坐标
        """

        device = "/dev/ttyS0"   # usb转接口 - 反接
        serial0 = uart.UART(device, 115200)

        # 和组控同步中心点 -- 修改坐标值

        new_data = self.modify_coordinates(date_dir, self.alter_coordinate)

        # 定义数据字典，使用 json 模式传输
        # new_data = {
        #     'from': [a, b],
        #     'to': [c, d]
        # }

        json_data = json.dumps(new_data)  # 初始化数据为 json 格式
        if serial0.write_str(json_data) < 0:
            return False
        return True
        
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
