from MyUI import GUI
from MyOpenCV import MyCV2
import look_for_data, my_utils
from maix import uart, app, time, camera, app, time, image, pinmap, gpio
from queue import Queue
import json

# Infoseek

class BaseInfo:
    # 初始化检测模型
    def __init__(self) -> None:
        # 初始化角点信息查找模块
        self.seek = look_for_data.InforSeek()
        
        # 实例化串口模块
        self.myutils = my_utils.MyUtils()   

        # # 初始化串口的回调函数
        self.myutils.received_callback(data_processing)
        # 初始化照明引脚
        pinmap.set_pin_function("B3", "GPIOB3")
        self.led = gpio.GPIO("GPIOB3", gpio.Mode.OUT)
        self.led.value(0)
        
    # 寻找角点坐标
    def find_corners(self, img:image.Image) -> dict:
        """
            根据传入的图片进行角点的查找
        """
        return self.seek.look_for_edge(img)

    # 寻找红、绿激光点信息
    def find_laser_spot(self, img:image.Image) -> dict:

        return self.seek.look_for_laser_spot(img)

    # 指定位置显示指定字符串
    def show_info(self, img:image.Image, place: list, content: str, color = image.Color.from_rgb(0, 0, 255), myscale=0.6):
        img.draw_string(
            int(place[0]), int(place[1]),         # x 和 y 是文字的左上角坐标
            content, # 要写的文字
            color=color, # color 是文字的颜色
            scale=myscale     # 放大字体
        )
        img.draw_cross(
            int(place[0]), int(place[1]),
            image.Color.from_rgb(255, 0, 0), 
            size=5, 
            thickness=2
        )

    def create_button(self, gui: GUI, label: str, x: int, y: int, cbf) -> int:
        btn_id = gui.createButton(x, y, _btn_width, _btn_height) 
        gui.setItemLabel(btn_id, label)   # 设置标签
        gui.setItemCallback(btn_id, self.btn_pressed) # 设置回调
        return btn_id

    def btn_pressed(self, btn_id, state):
        '''
            界面上按键的装填改变回调函数
        '''
        global _to_show_binary_img, _to_get_pixel, _btn_id_binary, _btn_id_pixel
        global _btn_id_LED, _btn_id_EXIT

        if state == 0: #只响应触摸抬起的动作（抬起时响应）
            return 
        
        if btn_id == _btn_id_LED:
            self.led.toggle()
        elif btn_id == _btn_id_RESET:
            app.switch_app('laseltrack')
            pass

    # 发送信息
    def send_data(self, data: dict) -> bool:
        if (self.myutils.send_pack_data(data)):
            return True
        return False

def data_processing(serial : uart.UART, bytes_data : bytes):
    """
        指挥函数：分析接受的数据，通过使用 queue.Queue 指挥主线程行为
    """

    print(f"on_received start\n")

    dir_data = {}   # 储存解析的数据

    try:
        # 将接收到的数据转化为字符串
        all_str_data = bytes_data.decode("utf-8")
        print(f"收到信息{all_str_data=}，开始json解析\n")

        # 进行字符切片获取到 {} 内的数据
        left_start = all_str_data.find('{')
        right_end = all_str_data.find('}') + 1
        str_data = all_str_data[left_start: right_end]

        dir_data = json.loads(str_data)
    except:
        print(f"json解析错误")  
    else:
        print(f"json解析成功, 为{dir_data}\n")

        result = {key: values for key, values in dir_data.items() if values[0] != 0}
        if queue.qsize() < 1:
            print(f'放入{result=}')
            queue.put(result)

# 初始化串口及指挥线程
queue = Queue()

#屏幕宽度和高度
_image_width  = 160
_image_height = 160
_btn_width  = _image_width//5
_btn_height = _image_height//5

# 初始化按钮框的id 和 flag
_btn_id_RESET = -1
_btn_id_LED = -1

data_dir = {}   # 初始化发送数据字典
info = BaseInfo()   # 实例化基本信息类

def main():
    global _btn_id_LED, _btn_id_RESET
    print(app.get_app_config_path())
    # _gray_flag = 1

    cam = camera.Camera(_image_width, _image_height)    # 初始化摄像头分辨率
    cam.skip_frames(30)     # 跳过开头的30帧
    
    gui = GUI() # 初始化ui显示
    cv = MyCV2()
    # 设置按钮框
    # _image_height - _btn_height 是为了将按钮表示在下边栏
    _btn_id_RESET = info.create_button(gui, '重启', 0, 0, info.btn_pressed) # 左上0
    _btn_id_LED = info.create_button(gui, '照明', 0, _btn_height, info.btn_pressed) # 左上1

    # # 初始化阈值范围
    # last_x = -1
    # last_y = -1
    # threshold = get_configured_threshold()
    # print(threshold)

    while not app.need_exit():
        time.fps_start()                         
        img = cam.read()

        img_show = img.copy()

        # 初始化字典
        edge_corners_dict = dict()
        laser_spot_dict = dict()
        data_dict = dict()

        if queue.qsize() > 0:
            order = queue.get()
            print(f"得到指令：{order=}")
            if 'scan' in order:
                # 尝试得到角点坐标字典
                edge_corners_dict = info.find_corners(img)
                if len(edge_corners_dict) != 4:
                    queue.put({'scan': 1})
                    gui.run(img_show)
                    continue
                
        # 得到激光点信息字典
        laser_spot_dict = info.find_laser_spot(img)

        # 调试使用
        # gui.show(laser_spot_dict)

        # 合成为最后的数据字典
        data_dict = edge_corners_dict | laser_spot_dict
        
        # 图片中批注信息
        for key, value in data_dict.items():
            info.show_info(img_show, value, f"{key}")

        # if len(edge_corners_dict) == 4:
        #     img_show.draw_string(
        #     0, _image_height-_btn_height,         # x 和 y 是文字的左上角坐标
        #     "edge YES", # text 是要写的文字
        #     image.Color.from_rgb(0, 255, 0), # color 是文字的颜色
        #     scale=1     # 放大字体
        #     )
        # if len(laser_spot_dict) > 1:
        #     img_show.draw_string(
        #     0, _image_height-2*_btn_height,         # x 和 y 是文字的左上角坐标
        #     "have laser", # text 是要写的文字
        #     image.Color.from_rgb(0, 255, 0), # color 是文字的颜色
        #     scale=1     # 放大字体
        #     )

        if len(data_dict.keys()) > 0:
            # 发送数据
            # print(f"本次发送的keys为：{data_dict.keys()}")
            if (info.send_data(data_dict)):
                print(f"YES，{data_dict}")
                pass
            else:
                print("FAIL")

        gui.run(img_show)
        time.sleep_ms(20)


if __name__ == '__main__':
    main()