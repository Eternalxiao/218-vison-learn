from threading import Thread
from time import sleep
from struct import pack
from maix import uart, pinmap
import json

class MyUtils:
    """
        自定义的串口类
    """
    
    def __init__(self) -> None:
        pinmap.set_pin_function("A16", "UART0_RX")
        pinmap.set_pin_function("A17", "UART0_TX")
        _device = "/dev/ttyS0"
        self.serial0 = uart.UART(_device, 115200)

    def send_data(self, date_dir: dict) -> bool:
        """
            串口传输数据函数 -- json分装
        """
        json_data = json.dumps(date_dir)  # 初始化数据为 json 格式
        if self.serial0.write_str(json_data) < 0:
            return False
        return True
    
    def send_str_data(self, spd_dif: int) -> bool:
        """
            串口传输函数,使用帧头封装,使用字符串发送
        """
        content = None
        
        # 初始化数据
        content = "AA{:03d}BB".format(int(spd_dif))
        
        # 进行发送
        if len(content) == 7:
            self.serial0.write_str(content)
            # print(f"发送：{content}")
            return True
        return False
    
    def send_pack_data(self, angle) -> bool:
        """
            串口传输使用帧头封装
        """

        content_angle = None
        angle_tmp = int(angle * 100)
        # print(f"发送{angle_tmp}")
        
        # 初始化帧头
        content_angle = b'\xAA'
        content_angle += pack(">h", angle_tmp)
        content_angle += b'\xBB'
        
        if content_angle != None:
            self.serial0.write(content_angle)
            # print(f"发送{content_angle}")
            return True
        return False

    def get_data(self):
        data = self.serial0.read()
        if data:
            print("Received, type: {}, len: {}, data: {}".format(type(data), len(data), data))
    
    def received_callback(self, cbf):
        """
            设置收到信息的回调
        """
        self.serial0.set_received_callback(cbf)

class SendThread(Thread):
    """
        串口发送线程类
    """
    def __init__(self, data_lock, _global_variable, active_flag, cbf):
        """
        初始化发送线程
        
        参数:
        data_lock: 线程锁，用于保护共享数据
        _global_variable: 对全局变量的引用
        active_flag: 线程活动标志
        """
        Thread.__init__(self)
        self.uart = MyUtils()
        self.data_lock = data_lock
        self._global_variable = _global_variable
        self.active_flag = active_flag
        self.daemon = True  # 设置为守护线程，主程序退出时自动结束

        self.ReceiveCallBack = cbf      # 记录回调函数
    def run(self):
        """
            线程主函数，执行数据发送, 启动线程接收函数
        """
        print("串口线程启动")
        
        self.uart.received_callback(self.ReceiveCallBack)

        while self.active_flag:
            try:
                # 获取锁，保护共享数据
                with self.data_lock:
                    current_data = self._global_variable[0]
                    
                # 发送数据
                # self.uart.send_str_data(current_data)
                self.uart.send_pack_data(current_data)
                # print(f"发送预本次数据发{current_data}")

                # 休眠20ms
                sleep(0.02)
            except Exception as e:
                print(f"发送线程错误: {e}")
                break

        print("串口发送线程退出")

        