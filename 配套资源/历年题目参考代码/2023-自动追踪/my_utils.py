from maix import uart, pinmap
import json
from struct import pack
# from constants import ERRORS
ERRORS = [0, 0]

class MyUtils:
    """
        自定义的串口类
    """
    def __init__(self) -> None:
        pinmap.set_pin_function("A16", "UART0_RX")
        pinmap.set_pin_function("A17", "UART0_TX")
        _device = "/dev/ttyS0"
        self.serial0 = uart.UART(_device, 115200)
        self.flag = 0

    def sending_coordinate_data(self, date_dir: dict) -> bool:
        """
            串口传输数据函数,使用Json封装
            date_dir:   存有六个点的数据字典，值为未修正的坐标
        """

        # 定义数据字典，使用 json 模式传输
        # new_data = {
        #     'red_dot': js_data['red_dot'],
        #     'green_dot': js_data['green_dot'],
        #     'first_dot': broders_data['first'],
        #     'second_dot': broders_data['second'],
        #     'third_dot': broders_data['third'],
        #     'forth_dot': broders_data['forth']
        # }

        json_data = json.dumps(date_dir)  # 初始化数据为 json 格式
        if self.serial0.write_str(json_data) < 0:
            return False
        return True
        
    def send_pack_data(self, data_dir: list) -> bool:
        """
            串口传输edge函数,使用帧头封装
            date_dir:   存有六个点的数据字典，值为未修正的坐标
        """
        edge_index_dict = {
            0: 'first',
            1: 'second',
            2: 'third',
            3: 'forth'
        }
        
        edge_required_keys = {'first', 'second', 'third', 'forth'}
        
        bytes_content_edge = None
        bytes_content_laser = None
        
        if edge_required_keys.issubset(data_dir.keys()):
            # 初始化帧头
            bytes_content_edge = b'\xD1\xD2'
            
            for index in range(4):
                x, y = data_dir[edge_index_dict[index]]
                if x and y:
                    print(f"edge {index} {x} {y}")
                    bytes_content_edge  += pack(">HH", int(x), int(y))
            
            bytes_content_edge += b'\xFE\xFF'
            
        if 'red_dot' in data_dir.keys():
            # 初始化帧头
            bytes_content_laser = b'\xE1\xE2'
            
            # 添加红色激光点坐标
            x, y = data_dir['red_dot']
            if x and y:
                print(f"\nred {x} {y}\n")
                bytes_content_laser += pack(">HH", int(x), int(y))
            
            # 添加绿色激光点
            if 'green_dot' in data_dir.keys():
                x, y = data_dir['green_dot']
                if x and y:
                    print(f"\ngreen {x} {y}\n")
                    bytes_content_laser += pack(">HH", int(x), int(y))
            
            bytes_content_laser += b'\xFE\xFF'

        # 添加帧尾后发送
        if bytes_content_edge != None:
            self.serial0.write(bytes_content_edge)
            self.flag += 1
        if bytes_content_laser != None:
            self.serial0.write(bytes_content_laser)
            self.flag += 1
        
        if self.flag > 0:
            self.flag = 0
            return True
        else:
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

    def received_callback(self, cbf):
        """
            设置收到信息的回调
        """
        self.serial0.set_received_callback(cbf)