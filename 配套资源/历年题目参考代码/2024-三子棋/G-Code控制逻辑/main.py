from constants import MODEL_DIR, MUD_FILE

import look_for_data, my_utils, topic_request
from maix import uart, app, time, camera, display, app, time, image, touchscreen, nn
from queue import Queue
import json, os, cv2, struct
import numpy as np

class BaseInfo:
    """
        获得无论是否完成哪一个题目都需要的基本信息 
        如棋盘的中心坐标列表、黑色和白色棋子坐标列表
        对获取的图片进行批注
    """
    
    # 初始化检测模型
    def __init__(self) -> None:

        # 初始化信息查找模块
        self.seek = look_for_data.GetBoard()
        # 实例化串口模块
        self.myutils = my_utils.MyUtils()


        # 闭运算卷积核 -- 矩形
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))      

    def find_chess_info(self, img:image.Image) -> dict:
        """
            神经网络获取棋子信息，返回信息字典
        """
        black_chess_list = []   # 初始化黑色棋子中心坐标列表
        white_chess_list = []   # 初始化白色棋子中心坐标列表
        objs = chess_detector.detect(img, conf_th = 0.5, iou_th = 0.45)
        
        for obj in objs:
            if chess_detector.labels[obj.class_id] == 'Black':
                cx = obj.x + obj.w/2.0
                cy = obj.y + obj.h/2.0
                black_chess_list.append([cx, cy])
            if chess_detector.labels[obj.class_id] == 'White':
                cx = obj.x + obj.w/2.0
                cy = obj.y + obj.h/2.0
                white_chess_list.append([cx, cy])

        
        chess_info = {
            "black_chess": black_chess_list,
            "white_chess": white_chess_list
        }
        
        # 批注图片
        # 标注白色棋子
        for white in white_chess_list:
            self.show_info(img, white, f"white")
        # 标注黑色棋子
        for black in black_chess_list:
            self.show_info(img, black, f"black")

        return chess_info

    def find_board(self, img) -> dict:
        """
            opencv的方式识别到棋盘信息
        """
        centers = []      # 中心点列表

        img_cv = image.image2cv(img, copy=False)   # 转换为 opencv 格式

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # 高斯模糊去噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 边缘检测，阈值 0，150
        edged = cv2.Canny(blurred, 50, 150)

        # 膨胀处理
        dilated = cv2.dilate(edged, self.kernel, iterations=1)

        # 腐蚀处理
        eroded = cv2.erode(dilated, self.kernel, iterations=1)

        # img = image.cv2image(eroded, False, False)
        # disp.show(img)

        # 查找最外围的轮廓
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))

        if len(contours) > 0:
            # 筛选出最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 计算参数，多边形逼近
            """
            cv2.approxPolyDP(curve, epsilon, closed) -- 返回近似后的多边形点集
            # # curve: 输入的轮廓点集
            # # epsilon: 近似精度，值越小，近似越精确
            """
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # 显示轮廓和角点
            if len(approx) == 4:
                corners = approx.reshape((4, 2))

                # 按照顺序对角点排序（左上，右上，右下，左下）
                rect = np.zeros((4, 2), dtype="int")
                points_sum = corners.sum(axis=1)
                rect[0] = corners[np.argmin(points_sum)]
                rect[2] = corners[np.argmax(points_sum)]
                points_diff = np.diff(corners,axis=1)
                rect[1] = corners[np.argmin(points_diff)]
                rect[3] = corners[np.argmax(points_diff)]
                corners = rect

                # 绘制四边路径
                cv2.drawContours(img_cv, [approx], -1, (0,255,0), 2)
                img = image.cv2image(img_cv, False, False)

                # 找出九宫格坐标
                centers = self.find_grid_centers(corners)
                # 定义新顺序的映射规则（新索引对应的原索引）
                # new_order = [6, 3, 0, 7, 4, 1, 8, 5, 2]
                new_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                # 生成重新排序后的列表
                centers = [centers[i] for i in new_order]
                centers = [centers[i:i+3] for i in range(0, len(centers), 3)]


                # # 画出中心点
                # if len(centers) == 9:
                #     for i in range(9):
                #         x, y = centers[i][0], centers[i][1]
                #         cv2.circle(img_cv, (x, y), 2, (0,255,0), -1)
                #         img = image.cv2image(img_cv, False, False)

                #         img.draw_string(x, y, f"{i+1}", image.COLOR_WHITE)

                #         # 设定棋子的判定区域，
                #         width = 15
                #         img.draw_rect(x-width//2, y-width//2, width, width, image.COLOR_WHITE)

                #         # 直方图均衡化 & 获取直方图 & 中位值/均值/众数
                #         img = img.histeq(adaptive=True)
                #         hist = img.get_histogram(thresholds=[[0, 100, -128, 127, -128, 127]], roi=[x-width//2, y-width//2, width, width])
                #         value = hist.get_statistics().a_median()

                #         # check piont & 放开下方代码可检验数组大小，以确定阈值
                #         # img.draw_string(x, y+10, f'{value}', image.COLOR_WHITE)

                #         # 根据区间值盘点棋子颜色
                #         color_chess = ''
                #         if value < -110:
                #             color_chess = 'Black'
                #         elif value > -80:
                #             color_chess = 'White'

                #         if color_chess != '':
                #             img.draw_string(x, y+10, color_chess, image.COLOR_WHITE)
        return {"board":centers}
    
    def find_grid_centers(self, corners):
        """
        corners: List of 4 tuples representing the corners of the 3x3 grid in the order:
                top-left, top-right, bottom-right, bottom-left
        """
        # Define the 3x3 grid points in the normalized coordinate space
        grid_points = np.array([[0, 0], [1/3, 0], [2/3, 0], [1, 0],
                                [0, 1/3], [1/3, 1/3], [2/3, 1/3], [1, 1/3],
                                [0, 2/3], [1/3, 2/3], [2/3, 2/3], [1, 2/3],
                                [0, 1], [1/3, 1], [2/3, 1], [1, 1]])

        # Convert corners to numpy array
        src_points = np.array(corners, dtype=np.float32)

        # Define destination points for the perspective transform
        dst_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

        # Compute the perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

        # Transform grid points to image coordinate space
        transformed_points = cv2.perspectiveTransform(np.array([grid_points], dtype=np.float32), transform_matrix)[0]

        # Calculate the center points of the grid cells
        centers = []
        for i in range(0, 3):
            for j in range(0, 3):
                cx = (transformed_points[i * 4 + j][0] + transformed_points[i * 4 + j + 1][0] + 
                    transformed_points[(i + 1) * 4 + j][0] + transformed_points[(i + 1) * 4 + j + 1][0]) / 4
                cy = (transformed_points[i * 4 + j][1] + transformed_points[i * 4 + j + 1][1] + 
                    transformed_points[(i + 1) * 4 + j][1] + transformed_points[(i + 1) * 4 + j + 1][1]) / 4
                centers.append((int(cx), int(cy)))

        return centers

    
    # 指定位置显示指定字符串
    def show_info(self, img:image.Image, place, content: str, color = image.Color.from_rgb(0, 0, 255), myscale=1):
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
    print("开始解析")

    dir_data = {}   # 储存解析的数据

    try:
        # 将接收到的数据转化为字符串
        all_str_data = struct.unpack_from('B', bytes_data, offset=3)[0]
        print(f"收到信息{all_str_data=}，开始字节提取解析")
        if all_str_data == 1:
            return
            dir_data = {"scan":[1]}
        elif all_str_data == 5 or all_str_data == 2:
            dir_data = {"five":[1]}
        elif all_str_data == 4 or all_str_data == 3:
            dir_data = {"four":[1]}
        else:
            return

    except:
        print(f"字节提取解析错误")  
    else:
        # print(f"json解析成功")
        result = {key: values for key, values in dir_data.items() if values[0] != 0}
        
        if queue.qsize() < 1:
            queue.put(result)
            # queue.put({"one":1})
    
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

# 确定 .mud 文件的路径
model_dir = MODEL_DIR
mud_file = MUD_FILE
model_path = os.path.join(model_dir, mud_file)
# 初始化棋子模型
chess_detector = nn.YOLO11(model=model_path, dual_buff = True)

# 初始化串口及指挥线程
device = "/dev/ttyS0"
queue = Queue()
serial = uart.UART(device, 115200)

serial.set_received_callback(data_processing)

# 初始化相机和显示器
cam = camera.Camera(chess_detector.input_width(), chess_detector.input_height(), chess_detector.input_format())
disp = display.Display()        
cam.skip_frames(30)     # 跳过开头的30帧   
# 初始化触摸屏
ts = touchscreen.TouchScreen()

# 初始化题目逻辑
topic = topic_request.SerialNumber()

# 初始化发送数据字典
data_dir = {}

info = BaseInfo()   # 实例化基本信息类

# data = "hello 1\r\n".encode()
# serial.write(data)
# print("sent:", data)
print("wait data")

while not app.need_exit():
    img = cam.read()
    # data = "hello 1\r\n".encode()
    # serial0.write(data)
    # print("sent:", data)
    
    chess_info = {}
    board_info = {}
    try:
        # 寻找棋盘信息
        board_info = info.find_board(img)
        if len(board_info["board"]) != 3 :
            print("失败寻找棋盘")
            disp.show(img)
            fps = time.fps()
            print(f"time: {1000/fps:.02f}ms, fps: {fps:.02f}\n")
            continue
    except KeyError as e:
        print(f"寻找棋盘报错{e}")
        continue
    # else:
    #     print("棋盘成功")
    
    try:
        # 寻找棋子信息
        chess_info = info.find_chess_info(img)
        if (len(chess_info["black_chess"]) < 1) or (len( chess_info["white_chess"]) < 1):
            print("失败寻找棋子")
            disp.show(img)
            fps = time.fps()
            print(f"time: {1000/fps:.02f}ms, fps: {fps:.02f}\n")
            continue
    except KeyError as e:
        print(f"寻找棋子报错{e}")
        continue
    # else:
    #     print("寻找棋子成功")
    
    
    base_info = board_info | chess_info

    # base_info = {
    #         "board": board_centre_list,
    #         "black_chess": black_chess_list,
    #         "white_chess": white_chess_list
    #     }

    # 标注九宫格
    for row, values in enumerate(base_info["board"]):
        for cul, value in enumerate(values):
            info.show_info(img, value, f"{row*3 + cul+1}")
    
    # draw exit button
    exit_label = "< Exit"   # draw content
    size = image.string_size(exit_label)    # 确定框大小
    exit_btn_pos = [0, 0, 8*2 + size.width(), 12 * 2 + size.height()]   # rect 的 x, y, w, h
    img.draw_string(8, 12, exit_label, image.COLOR_WHITE)   # 显示在屏幕上
    img.draw_rect(exit_btn_pos[0], exit_btn_pos[1], exit_btn_pos[2], exit_btn_pos[3],  image.COLOR_WHITE, 2)

    # 判断是否点击
    x, y, pressed = ts.read()
    if is_in_button(x, y, exit_btn_pos):
        app.set_exit_flag(True)
    img.draw_circle(x, y, 1, image.Color.from_rgb(255, 255, 255), 2)

    if queue.qsize() < 1:
        # print(f"未收到指示，无操作")
        disp.show(img)
        fps = time.fps()
        # print(f"time: {1000/fps:.02f}ms, fps: {fps:.02f}\n")
        time.sleep_ms(200)
        continue
    else:       # 正常开始游戏
        order = queue.get()
        print(f"得到指令：{order=}")
    
        if "scan" in order:
            base_info.update(info.find_board(img))
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
            time.sleep(1)
            data_dir = {}
            continue

        # 图像批注from-to信息，后发送
        for key, value in data_dir.items():
            print(f"批注from, to：{key}: {value}")
            info.show_info(img, value, key, color=image.Color.from_rgb(255, 0, 255), myscale=2)
        else:
            print(f"启动进行串口发送")
            info.sending_data(data_dir)
            for key, value in data_dir.items():
                print(f"{key}:, {value}")

    disp.show(img)

    fps = time.fps()
    print(f"time: {1000/fps:.02f}ms, fps: {fps:.02f}\n")
    time.sleep(2)    # 正常接收，此时机械结构正在运行，可以休息
