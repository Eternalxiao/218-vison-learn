from constants import BOARD_THRESHOLD, WHITE_THRESHOLD, BLACK_THRESHOLD
import look_for_data, my_utils, topic_request
from maix import uart, app, time, camera, display, app, time, image, touchscreen
from queue import Queue
import json

class BaseInfo:
    """
        获得无论是否完成哪一个题目都需要的基本信息 
        如棋盘的中心坐标列表、黑色和白色棋子坐标列表
        对获取的图片进行批注
    """
    def __init__(self) -> None:

        # 初始化信息查找模块
        self.seek = look_for_data.GetBoard()
        # 实例化串口模块
        self.myutils = my_utils.MyUtils()
        # 初始化颜色阈值
        self.board_threshold = BOARD_THRESHOLD
        self.white_threshold = WHITE_THRESHOLD
        self.black_threshold = BLACK_THRESHOLD
    
    def find_chess_info(self, img:image.Image) -> dict:
        """
            获取棋子信息，返回信息字典
        """

        # 获得黑色棋子坐标信息
        black_chess_list = []   # 初始化黑色棋子中心坐标列表
        black_chess_list = self.seek.find_chess(img, self.black_threshold)
        
        # 获得白色棋子坐标信息
        white_chess_list = []   # 初始化白色棋子中心坐标列表
        white_chess_list = self.seek.find_chess(img, self.white_threshold)
        
        chess_info = {
            "black_chess": black_chess_list,
            "white_chess": white_chess_list
        }

        # 批注图片
        # 标注白色棋子
        for white in white_chess_list:
            self.draw_str(img, white, f"white")
        # 标注黑色棋子
        for black in black_chess_list:
            self.draw_str(img, black, f"black")

        return chess_info

    def find_board(self):
        """
            5秒时间，平均滤波获得棋盘信息
        """
        start_time = time.time()    # 开始时间
        current_time = time.time()  # 结束时间
        timeout = 5     # 时间跨度
        board_info = [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ]   # 棋盘信息
        count = 0   # 查找次数
        flag = 0
        # 初始化累加记录
        sum_board_centre_list = [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ]

        # 寻找五秒棋盘信息
        while current_time - start_time < timeout:
            print(f"开始第{count=}次")
            _img = cam.read()
            img = _img.crop(80, 40, 160, 160)
            disp.show(img)
            # 获取棋盘中心坐标列表
            board_centre_list = self.seek.find_board(img, self.board_threshold)
            if len(board_centre_list) < 3:
                current_time = time.time()
                continue
            
            # 将坐标值累加记录
            for i in range(3):
                for j in range(3):
                    sum_board_centre_list[i][j][0] += board_centre_list[i][j][0]
                    sum_board_centre_list[i][j][1] += board_centre_list[i][j][1]
            count += 1
            flag = 1
            current_time = time.time()
        if flag:
            for i in range(3):
                for j in range(3):
                    board_info[i][j][0] = sum_board_centre_list[i][j][0] / count
                    board_info[i][j][1] = sum_board_centre_list[i][j][1] / count

            

        return {"board":board_info}
    
    # 指定位置显示指定字符串
    def draw_str(self, img:image.Image, place, content: str, color = image.Color.from_rgb(0, 0, 255), myscale=1):
        img.draw_string(
            int(place[0]), int(place[1]),         # x 和 y 是文字的左上角坐标
            content, # 要写的文字
            color=color, # color 是文字的颜色
            scale=myscale     # 放大字体
        )

    # 发送信息
    def sending_data(self, data: dict) -> bool:
        if (self.myutils.sending_coordinate_data(data)):
            return True
        return False

def data_processing(serial : uart.UART, bytes_data : bytes):
    """
        指挥函数：分析接受的数据，通过使用 queue.Queue 指挥主线程行为
    """

    print(f"on_received start")
    # print("received:", bytes_data)
    
    dir_data = {}   # 储存解析的数据

    try:
        # 将接收到的数据转化为字符串
        all_str_data = bytes_data.decode("utf-8")
        print(f"收到信息{all_str_data=}，开始json解析")
        # 进行字符切片获取到 {} 内的数据
        left_start = all_str_data.find('{')
        right_end = all_str_data.find('}') + 1
        str_data = all_str_data[left_start: right_end]

        dir_data = json.loads(str_data)
    except:
        print(f"json解析错误")  
    else:
        # print(f"json解析成功")

        result = {key: values for key, values in dir_data.items() if values[0] != 0}
        if queue.qsize() < 1:
            queue.put(result)
    
    # print("发送信息，开始干活---->>>")
    # event.set()
    # time.sleep_ms(200)

    # send back
    # while True:
    #     json_data = json.dumps(dir_data)  # 初始化数据为 json 格式
    #     serial.write_str(json_data)
    #     time.sleep_ms(1000) # sleep to make CPU free

# 确保映射的安全
def safe_mapping(chess_list, color_name, color_code):
    """带安全校验的映射函数"""
    # 前置条件断言
    assert isinstance(chess_list, list), "需要list类型输入"
    assert color_code in (1, 2), "非法棋子颜色编码"
    
    if chess_list:  # 仅在列表非空时执行
        print(f"开始映射{color_name}棋子...")
        return True
    return False

def is_in_button(x, y, btn_pos):
    return x > btn_pos[0] and x < btn_pos[0] + btn_pos[2] and y > btn_pos[1] and y < btn_pos[1] + btn_pos[3]


# 初始化串口及指挥线程
device = "/dev/ttyS0"
queue = Queue()
serial = uart.UART(device, 115200)

serial.set_received_callback(data_processing)

# 初始化相机和显示器
cam = camera.Camera(320, 224, fps=30)
disp = display.Display()        
cam.skip_frames(30)     # 跳过开头的30帧   
# 初始化触摸屏
ts = touchscreen.TouchScreen()

# 初始化题目逻辑
topic = topic_request.SerialNumber()

# # 初始化事件
# event = threading.Event()

# serial.write_str("hello\r\n")

# 初始化发送数据字典
data_dir = {}

info = BaseInfo()   # 实例化基本信息类

base_info = {}
# 开局尝试寻找棋盘
try:
    base_info = info.find_board()
except ValueError as e:
    print(f"{e}")
    serial.write_str("{'board': 'no board info'}")


print("sent hello")
print("wait data")

while not app.need_exit():
    _img = cam.read()
    img = _img.crop(80, 40, 160, 160)
    _img = img  # 用于屏幕显示

    # draw exit button
    exit_label = "< Exit"   # draw content
    size = image.string_size(exit_label)    # 确定框大小
    exit_btn_pos = [0, 0, 8*2 + size.width(), 12 * 2 + size.height()]   # rect 的 x, y, w, h
    img.draw_string(8, 12, exit_label, image.COLOR_WHITE)   # 显示在屏幕上
    img.draw_rect(exit_btn_pos[0], exit_btn_pos[1], exit_btn_pos[2], exit_btn_pos[3],  image.COLOR_WHITE, 2)

    # 判断是否
    x, y, pressed = ts.read()
    if is_in_button(x, y, exit_btn_pos):
        app.set_exit_flag(True)
    img.draw_circle(x, y, 1, image.Color.from_rgb(255, 255, 255), 2)
    
    # 寻找基本信息
    chess_list = info.find_chess_info(img)
    try:
        if (len(chess_list["black_chess"]) < 1) and (len(chess_list["white_chess"]) < 1):
            print("失败寻找棋子")
            continue
    except KeyError as e:
        print(f"没找到{e}")

    base_info = base_info | info.find_chess_info(img)

    
    # base_info = {
    #         "board": board_centre_list,
    #         "black_chess": black_chess_list,
    #         "white_chess": white_chess_list
    #     }

    # 标注九宫格
    for row, values in enumerate(base_info["board"]):
        for col, value in enumerate(values):
            info.draw_str(_img, value, f"{row*3 + col + 1}")
    
    if queue.qsize() < 1:
        print(f"未收到指示，无操作")
        disp.show(img)
        fps = time.fps()
        print(f"time: {1000/fps:.02f}ms, fps: {fps:.02f}\n")
        # time.sleep_ms(200)
        continue

    order = queue.get()
    print(f"得到指令：{order=}")
    
    if "scan" in order:
        base_info.update(info.find_board())
        data_dir = {"game": "scan succeed"}
    elif "one" in order:
        print("执行one, start")
        data_dir = topic.one(base_info["board"], base_info["black_chess"])
        print("执行one, end")
    elif ("two" in order) or ("three" in order):
        print("执行two or three, start")
        cell = order['two'][2]
        if order['two'][1] == 1:
            # 白色棋子
            chess_list = base_info["white_chess"]
        else:
            # 黑色棋子
            chess_list = base_info["black_chess"]

        data_dir = topic.two_and_three(cell, base_info["board"], chess_list)
        print("执行two or three, end")
# # =============== 加入人机博弈过程 ================
    elif "four" in order:
        print("执行four, start")
        
        # 初始化对弈棋盘列表
        board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
        ]

        # 将棋子映射到对弈棋盘列表
        if safe_mapping(base_info["white_chess"], "白色", 1):
            board = info.seek.map_all_chess_to_board(base_info["white_chess"], board, base_info["board"], 1)
        if safe_mapping(base_info["black_chess"], "黑色", 2):
            board = info.seek.map_all_chess_to_board( base_info["black_chess"], board, base_info["board"], 2)

        # 获取并发送本次移动信息
        data_dir = topic.four(base_info["board"], base_info["black_chess"], board)
        
        print("执行four, end")
    elif ("five" in order) or ("six" in order):
        print("执行five, start")
        
        # 初始化对弈棋盘列表
        board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
        ]

        # 将棋子映射到对弈棋盘列表
        if safe_mapping(base_info["white_chess"], "白色", 2):
            board = info.seek.map_all_chess_to_board(base_info["white_chess"], board, base_info["board"], 2)
        if safe_mapping(base_info["black_chess"], "黑色", 1):
            board = info.seek.map_all_chess_to_board( base_info["black_chess"], board, base_info["board"], 1)

        # 获取并发送本次移动信息
        data_dir = topic.five(base_info["board"], base_info["white_chess"], board)
        print("执行five, end")
    if "game" in data_dir:
        print(f"游戏异常")
        serial.write_str(f"{data_dir}")
        disp.show(img)
        fps = time.fps()
        print(f"time: {1000/fps:.02f}ms, fps: {fps:.02f}\n")
        # time.sleep_ms(200)
        continue

    # 图像批注from-to信息，后发送
    for key, value in data_dir.items():
        info.draw_str(_img, value, key, color=image.Color.from_rgb(255, 0, 255), myscale=2)
    else:
        print(f"发送坐标数据")
        info.sending_data(data_dir)
        for key, value in data_dir.items():
            print(f"{key}:, {value}")

    disp.show(_img)

    fps = time.fps()
    print(f"time: {1000/fps:.02f}ms, fps: {fps:.02f}\n")
    time.sleep(2)    # 正常接收，此时机械结构正在运行，可以休息
