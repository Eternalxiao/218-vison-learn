from maix import uart, pinmap
import json

class MyUtils:
    """
        自定义的串口类
    """
    
    def __init__(self) -> None:
        # 初始化数据发送串口
        pinmap.set_pin_function("A16", "UART0_RX")
        pinmap.set_pin_function("A17", "UART0_TX")
        _device0 = "/dev/ttyS0"
        self.serial0 = uart.UART(_device0, 9600)

        # 初始化电流接收串口
        pinmap.set_pin_function("A18", "UART1_RX")
        pinmap.set_pin_function("A19", "UART1_TX")
        _device1 = "/dev/ttyS1"
        self.serial1 = uart.UART(_device1, 9600)

    def send_data(self, date_dict: dict) -> bool:
        """
            串口传输数据函数 -- json分装
        """
        json_data = json.dumps(date_dict)  # 初始化数据为 json 格式
        if self.serial0.write_str(json_data) < 0:
            return False
        return True
     
    def _print(self, info):
        self.serial0.write_str(info)
        
    def get_data(self):
        data = self.serial0.read()
        if data:
            print("Received, type: {}, len: {}, data: {}".format(type(data), len(data), data))
    
    def received_callback(self, cbf):
        """
            设置收到信息的回调
        """
        print(f"shezhihuidiao")
        self.serial0.set_received_callback(cbf)
