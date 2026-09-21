# 导入函数库
from re import I
from MyUI import GUI
# from pid import PID  # 从pid模块中导入了PID类，用于实现PID控制算法。
from MySendThread import SendThread
from MyOpenCV import MyCV2 
import MyGetPixel, constants
from maix import camera, image, time, app, gpio, pinmap, nn, uart
import math, os.path
from json import loads
from threading import Lock
from queue import Queue


# 函数/变量初始化
rgb_to_lab = MyGetPixel.rgb_to_lab
rgb_to_hsv = MyGetPixel.rgb_to_hsv
set_configured_threshold = MyGetPixel.set_configured_threshold
get_configured_threshold = MyGetPixel.get_configured_threshold

# 初始化数字模型
model_dir = constants.MODEL_DIR
mud_file = constants.MUD_FILE
model_path = os.path.join(model_dir, mud_file)
num_detector = nn.YOLO11(model=model_path, dual_buff=True)

# 屏幕宽度和高度
_image_width = num_detector.input_width()
_image_height = num_detector.input_height()
_image_format = num_detector.input_format()
_btn_width = _image_width // 6
_btn_height = _image_height // 6

# 初始化照明引脚
pinmap.set_pin_function("B3", "GPIOB3")
led = gpio.GPIO("GPIOB3", gpio.Mode.OUT)
led.value(0)

# 初始化按钮框的id 和 flag
_btn_id_pixel = -1
_btn_id_binary = -1
_btn_id_RESET = -1
_btn_id_LED = -1

_to_show_binary_img = False
_to_get_pixel = False

# 主跟踪ROI区域初始化
TRA_ROIS = constants.TRA_ROIS
# 两边ROI区域初始化
SIDE_ROIS = constants.SIDE_ROIS

_scan_detected = False  # 是否处于扫描状态
_scan_start_time = 0    # 扫描开始时间
_scan_confidence = 0    # 扫描置信度
_scan_threshold = 5     # 需要连续5帧确认扫描结果
_goal_num = None       # 目标数字初始值

# 全局状态变量用于十字路口处理
_cross_detected = False  # 是否为十字路口
_cross_start_time = 0  # 遇到十字路口的开始时间
_cross_timeout = constants._cross_timeout  # 通过路口超时
_path_decision = None  # 存储路径决策

# # 提高决策可信度全局变量
_last_decision_time = 0  # 上次决策时间
_decision_confidence = 0  # 决策置信度
_decision_threshold = 5  # 需要连续3帧确认决策

# 添加串口数据发送相关的全局变量和锁
spd_dif_output = [0]  # 全局变量存储要发送的数据 -- 列表是为了线程共享
send_thread_active = True  # 控制发送线程是否运行
data_lock = Lock()  # 创建全局锁

# OBS_ROI = [(30 , 10, 100, 100)] #避障模式ROI区域


def Get_MaxIndex(blobs) -> int:
    """
        获得最大色块的位置索引函数
        # 输入:N个色块(blobs) 输出:N个色块中最大色块的索引(int i)
    """
    maxb_index = 0  # 最大色块索引初始化
    max_pixels = 0  # 最大像素值初始化
    for i in range(len(blobs)):  # 对N个色块进行N次遍历
        if blobs[i].pixels() > max_pixels:  # 当某个色块像素大于最大值
            max_pixels = blobs[i].pixels()  # 更新最大像素
            maxb_index = i  # 更新最大索引
            return maxb_index
    return -1

def btn_pressed(btn_id, state):
    '''
        界面上按键的装填改变回调函数
    '''
    global _to_show_binary_img, _to_get_pixel, _btn_id_binary, _btn_id_pixel
    global _btn_id_LED, _btn_id_EXIT

    if state == 0:  # 只响应触摸抬起的动作（抬起时响应）
        return

    if btn_id == _btn_id_binary:
        _to_show_binary_img = not _to_show_binary_img
        if _to_get_pixel:
            _to_get_pixel = False
    elif btn_id == _btn_id_pixel:
        _to_get_pixel = not _to_get_pixel
    elif btn_id == _btn_id_LED:
        led.toggle()
    elif btn_id == _btn_id_RESET:
        app.switch_app('roitrackline')

def create_button(gui: GUI, label: str, x: int, y: int, cbf) -> int:
    btn_id = gui.createButton(x, y, _btn_width, _btn_height)
    gui.setItemLabel(btn_id, label)  # 设置标签
    gui.setItemCallback(btn_id, btn_pressed)  # 设置回调
    return btn_id

def get_weight_for_side_point(x, y, tra_rois,scale=1.2):
    """
        获取tra_roisz中的权重值

        参数:
        x, y - 点在图像中的坐标
        tra_rois - TRA_ROIS列表，格式为(x, y, width, height, weight)
        scale 对匹配到的权重值的缩放
        返回:
        本点的权重值 和 质心
    """

    # 匹配TRA_ROI的权重值
    matched_weight = 0.0
    for roi in tra_rois:
        _, tra_y, _, tra_h, tra_weight = roi
        # 检查y坐标是否在当前TRA_ROI的y范围内
        if tra_y <= y < tra_y + tra_h:
            matched_weight = tra_weight*scale
            break

    # 返回得此点的权重值 和 质心
    return [matched_weight, x * matched_weight]

def detect_cross_intersection(img: image.Image, thresholds):
    """检测十字路口"""
    global _cross_detected, _cross_start_time

    # 检测左右两侧ROI是否同时有线
    left_blobs =  cv.find_hsv_blobs(img, thresholds, roi=SIDE_ROIS[0][0:4], pixels_threshold=50, merge=True)
    right_blobs =  cv.find_hsv_blobs(img, thresholds, roi=SIDE_ROIS[1][0:4], pixels_threshold=50, merge=True)

    # # 检测上方ROI是否有线（前方路口）
    # top_blobs =  cv.find_hsv_blobs(img, thresholds, roi=SIDE_ROIS[2][0:4], pixels_threshold=50, merge=True)

    # 十字路口判断条件：左右同时有线    且前方有线
    if left_blobs and right_blobs:  # and top_blobs
        if not _cross_detected:  # 新检测到路口初始化检测到的开始时间
            _cross_detected = True  
            _cross_start_time = time.ticks_ms()
            print("十字路口检测到!")
        return True
    return False

def filter_similar_thresholds(thresholds, tolerance=3):
    """
    过滤掉相似的颜色阈值列表
    比较每个列表中的所有六个数字，如果所有值都在容忍度范围内，则保留其中一个
    参数:
        thresholds: 阈值列表的列表 [[H_min, H_max, S_min, S_max, V_min, V_max], ...]
        tolerance: 允许的数值差异范围，默认为3
    返回:
        过滤后的阈值列表
    """
    # 存储过滤后的结果
    filtered = []
    
    # 用于记录已经保留的阈值
    kept_thresholds = []
    
    # 遍历所有阈值
    for th in thresholds:
        # 检查是否与已保留的阈值相似
        is_similar = False
        
        # 与每个已保留的阈值比较
        for kept in kept_thresholds:
            # 检查所有六个值是否都在容忍度范围内
            all_within_range = True
            for i in range(6):
                # 对于每个值，检查是否在容忍度范围内
                if abs(th[i] - kept[i]) > tolerance:
                    all_within_range = False
                    break
            
            # 如果所有值都在范围内，则标记为相似
            if all_within_range:
                is_similar = True
                break
        
        # 如果不相似，则保留当前阈值
        if not is_similar:
            filtered.append(th)
            kept_thresholds.append(th)
    
    return filtered

# 是否通过路口检测
def check_exit_intersection(img, thresholds):
    """检查是否真正离开路口"""
    # 1. 检查左右ROI无线
    left_clear = not  cv.find_hsv_blobs(img, thresholds, roi=SIDE_ROIS[0][0:4])
    right_clear = not  cv.find_hsv_blobs(img, thresholds, roi=SIDE_ROIS[1][0:4])

    # 2. 中央ROI检测到强信号
    center_strong = False
    for r in TRA_ROIS[2:4]:  # 只检查中央区域
        blobs =  cv.find_hsv_blobs(img, thresholds, roi=r[0:4], area_threshold=500)
        if blobs:
            center_strong = True
            break

    return left_clear and right_clear and center_strong

# 指挥函数
def data_processing(serial : uart.UART,bytes_data : bytes):
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
        left_start = all_str_data.find('{') if all_str_data.find('{') > -1 else 0
        right_end = (all_str_data.find('}') + 1)  if all_str_data.find('}') > -1 else 0
        str_data = all_str_data[left_start: right_end]

        dir_data = loads(str_data)  # json解析数据
    except:
        print(f"json解析错误, 非json格式")  
    else:
        # print(f"json解析成功")

        result = {key: values for key, values in dir_data.items()}
        if queue.qsize() < 1:
            queue.put(result)
        

# 初始化串口及指挥线程
queue = Queue()
# 处死话opencv处理
cv = MyCV2()
  
def main():
    # 初始化 二值化 和 取阈值 功能全局控制变量
    global _to_show_binary_img, _to_get_pixel, _btn_id_binary, _btn_id_pixel

    # 初始化 重启 和 点亮LED 功能全局变量
    global _btn_id_LED, _btn_id_RESET
    
    # 初始化 十字路口决策 全局变量
    global _cross_detected, _cross_start_time, _path_decision
    
    # 初始化 决策可信控制 全局变量
    global _last_decision_time, _decision_confidence  # 新增全局变量

    # 添加对串口发送相关全局变量的引用
    global spd_dif_output, send_thread_active

    # 使用全局变量决策开始识别的数字
    global _scan_detected, _scan_start_time, _scan_confidence, _goal_num

    cam = camera.Camera(_image_width, _image_height, _image_format)  # 初始化摄像头分辨率
    cam.skip_frames(30)  # 跳过开头的30帧
    cam.constrast(70)  # 设置对比度
    gui = GUI()  # 初始化ui显示

    # 创建并启动串口收发线程
    send_thread = SendThread(data_lock, spd_dif_output, send_thread_active, data_processing)
    send_thread.start()

    _debug_mode = False # 初始化debug模式标志

    print(f"{app.get_app_config_path()=}")
    _gray_flag = True

    # 初始化记录上一次的线质心值
    LastCenter_Pos = _image_width // 2

    # 初始化PID计算
    # saved_kp = app.get_app_config_kv('pid_params', 'kp', '0', True)
    # saved_ki = app.get_app_config_kv('pid_params', 'ki', '0', True)
    # saved_kd = app.get_app_config_kv('pid_params', 'kd', '0', True)

    # p = float(saved_kp) if saved_kp != '0' else constants._Kp
    # i = float(saved_ki) if saved_ki != '0' else constants._ki
    # d = float(saved_kd) if saved_kd != '0' else constants._kd
    
    # imax = constants._imax
    # spd_dif_pid = PID(p, i, d, imax)  # 创建一个PID对象spd_dif_pid，用于控制线的角度偏差。
    # print(f"初始化的参数为：{p,i,d,imax}")
    
    # 设置按钮框
    # _image_height - _btn_height 是为了将按钮表示在下边栏
    _btn_id_pixel = create_button(gui, '取阈值', 0, _image_height - _btn_height, btn_pressed)  # 左下
    _btn_id_binary = create_button(gui, '二值化', _image_width - _btn_width, _image_height - _btn_height, btn_pressed)  # 右下
    _btn_id_RESET = create_button(gui, '重启', 0, 0, btn_pressed)  # 左上0
    _btn_id_LED = create_button(gui, '照明', 0, _btn_height, btn_pressed)  # 左上1

    last_x = -1
    last_y = -1

    # 初始化阈值信息
    thresholds = get_configured_threshold()
    # print(thresholds)

    # 主循环
    while not app.need_exit():
        img = cam.read()  # 截一帧图像
        img_show = img.copy()

        # objs = num_detector.detect(img, conf_th=0.5, iou_th=0.45)
        # for obj in objs:
        #     img_show.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
        #     _goal_num = f"{num_detector.labels[obj.class_id]}"
        #     msg = f'{num_detector.labels[obj.class_id]}: {obj.score:.2f}'
        #     img_show.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED)
        

        # 收到扫描信号，开始识别数字
        if queue.qsize() > 0:
            order = queue.get()
            print(f"得到指令：{order=}")
            if 'scan' in order:
                # 开始扫描
                _scan_detected = True
                _scan_start_time = time.ticks_ms()
                _scan_confidence = 0
                print("开始数字扫描...")

        # ============== 数字扫描逻辑 ==============
        if _scan_detected:
            # 1. 显示扫描状态
            img_show.draw_string(0, _btn_height*2, "SCANNING...", color=image.COLOR_RED)
            
            # 2. 尝试识别数字
            current_num = None
            objs = num_detector.detect(img, conf_th=0.6, iou_th=0.45)
            for obj in objs:
                img_show.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_RED)
                current_num  = f"{num_detector.labels[obj.class_id]}"
                msg = f'{num_detector.labels[obj.class_id]}: {obj.score:.2f}'
                img_show.draw_string(obj.x, obj.y, msg, color=image.COLOR_RED)
            
            # 3. 更新置信度
            if current_num is not None:
                # 如果连续识别到相同的数字
                if current_num == _goal_num:
                    _scan_confidence += 1
                else:
                    # 识别到不同的数字，重置目标数字和置信度
                    _goal_num = current_num
                    _scan_confidence = 1
                
                # 显示当前识别状态
                img_show.draw_string(
                    0, _btn_height*3,
                    f"Num: {current_num} Conf: {_scan_confidence}/{_scan_threshold}", 
                    color=image.COLOR_GREEN
                )
            else:
                # 未识别到数字
                img_show.draw_string(
                    0, _btn_height*3,
                    "No number detected", 
                    color=image.COLOR_RED
                )
            
            # 4. 检查是否达到置信度阈值
            if _scan_confidence >= _scan_threshold:
                print(f"数字扫描完成! 确认数字: {_goal_num}")
                _scan_detected = False
            
            # 5. 检查扫描超时（3秒）
            if time.ticks_diff(time.ticks_ms(), _scan_start_time) > 3000:
                print("扫描超时")
                _scan_detected = False
                
                # 显示扫描失败
                img_show.draw_string(
                    0, _btn_height*2,
                    "Scan Timeout", 
                    color=image.COLOR_RED
                )

            # 显示ui
            gui.run(img_show)
            continue

        # opencv 进行图像处理
        img = cv.image_process(img)
        # # 为了调试
        # gui.show(img)

        # 得到自适应阈值的二值化图像
        # img_adapt_binary = apply(img) # 需要导入

        # 点击后根据回调将全局的 flag 被修改
        if _to_show_binary_img:  # 二值化显示
            img_show = img.binary(thresholds, False)  # 根据色域二值化图像

        if _to_get_pixel:  # 获取阈值
            x, y = gui.get_touch()
            if last_x != x or last_y != y:  # 判断是否与上次点击位置相同
                last_x = x
                last_y = y  # 重置 last
                
                # 重置全局阈值列表
                thresholds = []
                
                # 定义采样区域参数
                REGION_WIDTH = 6      # 区域宽度
                REGION_HEIGHT = 15    # 区域高度
                REGION_SPACING = 50   # 区域间距
                NUM_REGIONS = 3       # 总共采样3个区域
                
                # 在图像上绘制中心点
                img_show.draw_cross(x, y, image.COLOR_YELLOW, 8, 2)
                img_show.draw_string(x+5, y-10, "采样中心", image.COLOR_YELLOW)
                
                # 遍历三个采样区域
                for i in range(NUM_REGIONS):
                    # 计算当前区域的Y位置（从点击位置开始，向上偏移）
                    center_y = y - i * REGION_SPACING
                    
                    # 确保采样区域在图像范围内
                    start_x = max(0, x - REGION_WIDTH // 2)
                    end_x = min(_image_width - 1, x + REGION_WIDTH // 2)
                    
                    start_y = max(0, center_y - REGION_HEIGHT // 2)
                    end_y = min(_image_height - 1, center_y + REGION_HEIGHT // 2)
                    
                    # 调整实际尺寸（边界保护）
                    actual_width = end_x - start_x + 1
                    actual_height = end_y - start_y + 1
                    
                    # 在图像上绘制采样区域
                    img_show.draw_rect(start_x, start_y, actual_width, actual_height, image.COLOR_YELLOW, 1)
                    img_show.draw_string(
                        end_x + 5, 
                        center_y, 
                        f"区域{i+1}", 
                        image.COLOR_YELLOW
                    )
                    
                    # 存储当前区域的像素
                    region_pixels = []
                    
                    # 遍历当前区域的所有像素
                    for py in range(start_y, end_y + 1):
                        for px in range(start_x, end_x + 1):
                            pixel = img.get_pixel(px, py, True)
                            if pixel:
                                region_pixels.append(pixel)
                    
                    # 计算当前区域的平均RGB
                    if region_pixels:
                        avg_rgb = [0, 0, 0]
                        for p in region_pixels:
                            # 确保像素有至少3个值（R,G,B）
                            if len(p) >= 3:
                                for c in range(3):
                                    avg_rgb[c] += p[c]
                        
                        # 计算RGB平均值
                        for c in range(3):
                            avg_rgb[c] = int(avg_rgb[c] / len(region_pixels))
                        
                        # 转换为HSV阈值范围,并添加
                        threshold_ranges = rgb_to_hsv(avg_rgb)

                        # thresholds.append(threshold_ranges)   # 每次只返回一个列表
                        for threshold_range in range(len(threshold_ranges)):
                            thresholds.append(threshold_ranges[threshold_range])
                        
                        # 过滤相似信息，减小运算量
                        thresholds = filter_similar_thresholds(thresholds)

                        # 打印调试信息
                        print(f"区域 {i+1} (Y={center_y}) 的RGB: {avg_rgb}")
                        print(f"返回的阈值范围: {threshold_ranges}")
                        
                    else:
                        print(f"区域 {i+1} 没有有效像素")
                
                # 保存所有阈值到配置文件
                if len(thresholds) > 0:
                    # set_configured_threshold(thresholds)
                    
                    # 显示统计信息
                    stats_text = f"已保存 {len(thresholds)} 个阈值范围"
                    img_show.draw_string(5, _image_height - 20, stats_text, image.COLOR_GREEN)
                    print(f"保存的阈值列表: {thresholds}")
                else:
                    img_show.draw_string(5, _image_height - 20, "未找到有效像素", image.COLOR_RED)
            # 标记点击中心点
            img_show.draw_cross(x, y, image.COLOR_YELLOW, 8, 2)
            img_show.draw_string(x+5, y-10, "采样中心", image.COLOR_YELLOW)
        
        # print(f"{thresholds=}")

        if len(thresholds) < 1:
            thresholds = get_configured_threshold()

        # ==============  循迹实现  ==============
        # 初始化质心计算变量
        Centroid_Sum = 0  # 初始化质心和
        Weight_Sum = 0  # 权值和初始化

        # 初始化质心计算变量
        Center_Pos = 0

        # 判断是否有十字路口
        have_cross = detect_cross_intersection(img, thresholds)

        # ============== 十字路口处理逻辑 ==============
        if _cross_detected:     # 如果遇到了十字路口
            # 1. 显示此时路口状态
            img_show.draw_string(_btn_width, 0, "CROSS DETECTED", color=image.COLOR_RED)

            # 2. 更新决策置信度（如果连续几帧决策不一致，则重置决策）
            current_time = time.ticks_ms()
            if current_time - _last_decision_time > 3000:  # 通过路口的时间内隔一段时间每进行重新决策
                _decision_confidence = 0

            # 3. 如果决策次数未到，准备持续进行数字识别
            if _decision_confidence < _decision_threshold and time.ticks_ms() % 3 == 0: # 每3帧检测一次
                
                # 初始化本次决策
                current_decision = None

                # 使用神经网络识别数字
                if not _to_show_binary_img:
                    objs = num_detector.detect(img_show, conf_th=0.6, iou_th=0.45)
  
                    # 进行本次识别的判断
                    for obj in objs:
                        num = num_detector.labels[obj.class_id]
                        # img_show.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_YELLOW)
                        # 根据数字决定行动
                        if num == _goal_num:
                            if obj.x < _image_width // 2 - 30:  # 左转
                                current_decision = 'TURN_LEFT'
                                img_show.draw_string(_btn_width, _btn_height, f"LEFT？{_decision_confidence}TO: {num}", color=image.COLOR_GREEN)
                            elif obj.x > _image_width // 2 + 30:  # 右转
                                current_decision = 'TURN_RIGHT'
                                img_show.draw_string(_btn_width, _btn_height, f"RIGHT？{_decision_confidence}TO: {num}", color=image.COLOR_GREEN)
                            break  # 已寻找到，退出                                 
                    else:  # 如果没有识别到目标数字，默认直行
                        current_decision = 'STRAIGHT'
                        img_show.draw_string(_btn_width, _btn_height, f"GO STRAIGHT?{_decision_confidence}", color=image.COLOR_GREEN)
                    
                # 更新决策置信度
                if (_path_decision == current_decision) and current_decision is not None:
                    _decision_confidence += 1   # 与历史决策记录相同，可信度加一
                else:
                    _path_decision = current_decision
                    _decision_confidence = 1  # 重置为1，因为当前帧已经确认
                
                _last_decision_time = current_time
            else:
                # 本段时间内已经连续多次得到相同的决策，显示最终决策
                img_show.draw_string(_btn_width, _btn_height, f"{_path_decision} (Confirmed)", color=image.COLOR_PURPLE)
                pass

            # 4. 检查是否通过路口
            time_elapsed = time.ticks_diff(time.ticks_ms(), _cross_start_time)
            # 条件1：超时（规定时间必须通过）或条件2：不再检测到路口特征
            if time_elapsed > _cross_timeout or (not have_cross and check_exit_intersection(img, thresholds)):
                print(f"通过路口! 用时: {time_elapsed}ms, 决策: {_path_decision}")
                _cross_detected = False
                _path_decision = None
                _decision_confidence = 0

        # ============== 巡线逻辑 ==============
        # 获得主区域每个区域的线质心，并进行相加
        for r in TRA_ROIS:  # ROI的元组
            # 找到视野中ROI区域的色块,merge=true,将找到的图像区域合并
            blobs =  cv.find_hsv_blobs(img, thresholds, roi=r[0:4], pixels_threshold=100, area_threshold=100, merge=True)
            # print(f"寻找{thresholds}")

            if blobs:  # 如果找到了色块 计算质心和
                # print(f"找到色块")
                maxb_index = Get_MaxIndex(blobs)  # 找到多个色块中的最大色块返回索引值

                # 最大色块的中心位置标记十字
                cx = blobs[maxb_index].cx()
                cy = blobs[maxb_index].cy()
                if _debug_mode:
                    # 返回最大色块外框元组(x,y,w,h) 绘制线宽为2的矩形框
                    blob_rect = blobs[maxb_index].rect()
                    img_show.draw_rect(blob_rect[0], blob_rect[1], blob_rect[2], blob_rect[3],
                                        color=image.Color.from_rgb(0, 0, 255), thickness=2)

                    img_show.draw_cross(cx, cy, color=image.Color.from_rgb(255, 0, 0))
                    pass

                # 计算质心和 = (ROI中最大颜色块的中心点横坐标)cx * (ROI权值)w
                Centroid_Sum += cx * r[4]
                Weight_Sum += r[4]

        # # 直行通过路口逻辑
        # if _path_decision != None and _path_decision == 'STRAIGHT':
        #     # 只使用中间三个ROI区域（权重较高）
        #     for r in TRA_ROIS[1:4]:  # 使用中间三个ROI
        #         blobs =  cv.find_hsv_blobs(img, thresholds, roi=r[0:4], merge=True)
        #         if blobs:
        #             maxb_index = Get_MaxIndex(blobs)
        #             cx = blobs[maxb_index].cx()
        #             Centroid_Sum += cx * r[4]
        #             Weight_Sum += r[4]

        # 转向通过路口逻辑
        if _path_decision != None and _path_decision[0:4] == 'TURN':
            # 需要转向，根据左右转向指令将左右两边roi区域内的色块并入
            print(f"准备转弯")
            # 定义转向字典
            _decision_dict = {
                'TURN_LEFT': 0,
                'TURN_RIGHT': 1
            }

            r = SIDE_ROIS[_decision_dict[_path_decision]]  # 要转向的ROI
            
            # 找到视野中ROI区域的色块,merge=true,将找到的图像区域合并
            blobs =  cv.find_hsv_blobs(img, thresholds, roi=r[0:4], pixels_threshold=100, area_threshold=100,merge=True)
            # print(f"寻找{thresholds}")

            if blobs:   # 如果找到了色块 计算质心和
                maxb_index = Get_MaxIndex(blobs)    # 找到多个色块中的最大色块返回索引值

                # 返回最大色块外框元组(x,y,w,h) 绘制线宽为2的矩形框
                
                # 最大色块的中心位置标记十字
                cx = blobs[maxb_index].cx()
                cy = blobs[maxb_index].cy()
                if _debug_mode:
                    blob_rect = blobs[maxb_index].rect()
                    img_show.draw_rect(blob_rect[0], blob_rect[1], blob_rect[2], blob_rect[3], color=image.COLOR_YELLOW, thickness=2)

                    img_show.draw_cross(cx, cy, color=image.COLOR_YELLOW, thickness=2)
                    pass
                # 根据 y 值获得此次的 权重值 和 质心
                weight_side, centroid_side = get_weight_for_side_point(cx, cy, TRA_ROIS)
                # 计算质心和 = (ROI中最大颜色块的中心点横坐标)cx * (ROI权值)w
                Centroid_Sum += centroid_side
                Weight_Sum += weight_side
                print(f"转弯添加权重{centroid_side=}, {weight_side=}")

        # 确定线心位置 = 质心和 / 权值和
        Center_Pos = (Centroid_Sum / Weight_Sum) if Weight_Sum != 0 else LastCenter_Pos

        # ============== 通用处理逻辑 - 计算线心、PID ==============
        # 将线心 Center_Pos 转换为偏角
        Deflection_Angle = 0  # 偏角初始化为0
        Deflection_Angle = -math.atan((Center_Pos - _image_width / 2) / (_image_height / 2))  # 计算偏角
        Deflection_Angle = math.degrees(Deflection_Angle)  # 弧度值转换为角度

        # # 计算根据角度的
        angle_err = Deflection_Angle
        
        # 获取锁后修改out值
        with data_lock:
            # print(f"修改{spd_dif_output=}")
            # spd_dif_output[0] = spd_dif_pid.get_position_pid(angle_err, 1)
            spd_dif_output[0] = int(angle_err)
            # print(f"{spd_dif_output=}\n")
        
        # if abs(angle_err) > 10:   # 误差太大时添加PID自适应调整
        #     # 进行自适应 PID 算法，无误后保存到配置文件
        #     if spd_dif_pid.adapt_pid(angle_err, mode='position'):
        #         # 保存调整后的PID参数
        #         kp, ki, kd = spd_dif_pid.get_params()
        #         app.set_app_config_kv('pid_params', 'kp', str(kp), True)
        #         app.set_app_config_kv('pid_params', 'ki', str(ki), True)
        #         app.set_app_config_kv('pid_params', 'kd', str(kd), True)
        #         print(f"Saved PID params: Kp={kp}, Ki={ki}, Kd={kd}")


        if _debug_mode:  # 每5帧更新一次
            # 对图像进行必要批注
            # img_show.draw_line(
            #     160, 0, 160, 224,
            #     image.Color.from_rgb(255, 0, 0)
            # )  # 图像中心线

            # img_show.draw_line(
            #     int(Center_Pos), 0, _image_width // 2, _image_height // 2,
            #     color=image.Color.from_rgb(255, 0, 0),
            #     thickness=2
            # )  # 线心标注

            # p, i, d = spd_dif_pid.get_params()
            # img_show.draw_string(
            #     _btn_width + 30, _image_height - _btn_height + 15,
            #     f'Kp: {round(p, 2)}, Ki: {round(i, 2)}, Kd: {round(d, 2)}'
            # )  # 数值显示
            pass

        if time.ticks_ms() % 5 == 0:
            img_show.draw_string(
                    _image_width - 2 * _btn_width, 0,
                    f'angle: {round(angle_err, 2)}\ngoal: {_goal_num}'
                )  # 数值显示
            
        img_show.draw_arrow(
                160, _image_height - _btn_height, int(Center_Pos), _image_height - _btn_height,
                image.Color.from_rgb(0, 255, 0),
                thickness=4
            )  # 偏转箭头

        # 修改上一次的记录
        LastCenter_Pos = Center_Pos

        # 显示ui
        gui.run(img_show)
    
    # 退出循环后，设置标志停止发送线程
    send_thread_active = False


if __name__ == '__main__':
    main()
