from maix import image, camera, display, app, time, uart, pinmap
import math
# 初始化串口通信，设置设备路径和波特率
device = "/dev/ttyS0"
serial = uart.UART(device, 115200)


# 初始化摄像头和显示设备
cam = camera.Camera(320, 240)  # 创建320x240分辨率的摄像头对象
disp = display.Display()      # 创建显示对象用于输出图像
cam.skip_frames(30)           # 跳过开头的30帧

cam.gain(0)
# cam.luma(60)		    # 设置亮度，范围[0, 100]
# # cam.awb_mode(1)
# cam.constrast(70)		# 设置对比度，范围[0, 100]
# cam.gain(100)
# cam.skip_frames(30)           # 跳过开头的30帧
# cam.saturation(0)		# 设置饱和度，范围[0, 100]
# 全局变量：存储激光点位置和检测状态
x_red = 0                     # 红色激光点的x坐标
y_red = 0                     # 红色激光点的y坐标
has_laser = False             # 激光点检测标志
big_roi = None                # 存储检测到的最大矩形ROI
small_roi = None              # 存储缩小后的ROI区域
prev_rect = None              # 存储上一帧的矩形信息用于稳定性判断
prev_small_rect = None        # 存储上一帧small_roi内的矩形信息
stable_counter = 0            # 稳定计数器
small_stable_counter = 0      # small_roi内矩形的稳定计数器
MAX_STABLE_COUNT = 3          # 达到稳定所需的连续帧数
rect_history = []             # 矩形历史记录用于平滑
MAX_HISTORY = 5               # 最大历史记录数

# 固定参考点（图像中心）：用于判断物体位置
FIXED_CENTER_X = cam.width() // 2  # 图像水平中心位置(160)
FIXED_CENTER_Y = cam.height() // 2 # 图像垂直中心位置(120)

# 颜色阈值：用于颜色识别
black_threshold = (0, 15, -7, 13, -14, 6)  # 黑色阈值(HSL色彩空间)
jiguang_black_thresholds = [
    [46, 66, 9, 40, -11, 9],  # 激光颜色阈值1 - 调整为更宽的范围
    [0, 25, 15, 44, -4, 16],   # 激光颜色阈值2
    [4, 24, 25, 45, 5, 25]
]   # [4, 24, 25, 45, 5, 25]

# 卡尔曼滤波器类
class KalmanFilter:
    def __init__(self, q=0.1, r=1, p=1):
        self.Q = [[q, 0], [0, q]]  # 过程噪声协方差
        self.R = [[r, 0], [0, r]]  # 测量噪声协方差
        self.P = [[p, 0], [0, p]]  # 估计误差协方差
        self.A = [[1, 0], [0, 1]]  # 状态转移矩阵
        self.H = [[1, 0], [0, 1]]  # 观测矩阵
        self.x_est = [0, 0]  # 状态估计 [位置, 速度]
        self.last_time = time.ticks_ms()
        
    def update(self, z):
        """更新滤波器状态
        Args:
            z: 测量值 [位置, 速度]
        Returns:
            更新后的估计值 [位置, 速度]
        """
        current_time = time.ticks_ms()
        dt = (current_time - self.last_time) / 1000.0
        self.last_time = current_time
        
        # 更新状态转移矩阵
        self.A = [[1, dt], [0, 1]]
        
        # 预测步骤
        x_pred = [
            self.A[0][0] * self.x_est[0] + self.A[0][1] * self.x_est[1],
            self.A[1][0] * self.x_est[0] + self.A[1][1] * self.x_est[1]
        ]
        
        P_pred = [
            [
                self.A[0][0] * self.P[0][0] * self.A[0][0] + self.A[0][1] * self.P[1][1] * self.A[0][1] + self.Q[0][0],
                self.A[0][0] * self.P[0][1] * self.A[1][1] + self.A[0][1] * self.P[1][1] * self.A[1][1]
            ],
            [
                self.A[1][0] * self.P[0][0] * self.A[0][0] + self.A[1][1] * self.P[1][0] * self.A[0][1],
                self.A[1][0] * self.P[0][1] * self.A[1][0] + self.A[1][1] * self.P[1][1] * self.A[1][1] + self.Q[1][1]
            ]
        ]
        
        # 计算卡尔曼增益
        S = [
            [P_pred[0][0] + self.R[0][0], P_pred[0][1]],
            [P_pred[1][0], P_pred[1][1] + self.R[1][1]]
        ]
        
        det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
        if det == 0:
            K = [[0, 0], [0, 0]]
        else:
            inv_S = [
                [S[1][1]/det, -S[0][1]/det],
                [-S[1][0]/det, S[0][0]/det]
            ]
            K = [
                [
                    P_pred[0][0] * inv_S[0][0] + P_pred[0][1] * inv_S[1][0],
                    P_pred[0][0] * inv_S[0][1] + P_pred[0][1] * inv_S[1][1]
                ],
                [
                    P_pred[1][0] * inv_S[0][0] + P_pred[1][1] * inv_S[1][0],
                    P_pred[1][0] * inv_S[0][1] + P_pred[1][1] * inv_S[1][1]
                ]
            ]
        
        # 更新估计
        y = [z[0] - x_pred[0], z[1] - x_pred[1]]
        self.x_est = [
            x_pred[0] + K[0][0] * y[0] + K[0][1] * y[1],
            x_pred[1] + K[1][0] * y[0] + K[1][1] * y[1]
        ]
        
        # 更新误差协方差
        self.P = [
            [
                (1 - K[0][0]) * P_pred[0][0] - K[0][1] * P_pred[1][0],
                (1 - K[0][0]) * P_pred[0][1] - K[0][1] * P_pred[1][1]
            ],
            [
                -K[1][0] * P_pred[0][0] + (1 - K[1][1]) * P_pred[1][0],
                -K[1][0] * P_pred[0][1] + (1 - K[1][1]) * P_pred[1][1]
            ]
        ]
        
        return self.x_est

def red_jiguang(img, kf_x=None, kf_y=None, draw=True):
    """检测红色激光点并返回其坐标
    
    参数:
        img: 输入图像
        kf_x: x方向卡尔曼滤波器(可选)
        kf_y: y方向卡尔曼滤波器(可选)
        draw: 是否在图像上绘制标记
        
    返回:
        (x, y): 激光点的中心坐标，如果未检测到则返回(0, 0)
        (x_est, y_est): 经过卡尔曼滤波后的坐标(如果使用滤波器)
    """
    # 设置相机曝光以更好捕捉激光点
    cam.exposure(12000)
    
    # 查找激光点
    blobs = img.find_blobs(jiguang_black_thresholds, 
                         area_threshold=1,
                         pixels_threshold=1,
                         merge=True)
    
    x, y = 0, 0
    x_est, y_est = 0, 0
    has_laser = False
    
    if blobs:
        # 找到最大的色块作为激光点
        largest_blob = max(blobs, key=lambda b: b.pixels())
        x, y = largest_blob.cx(), largest_blob.cy()
        has_laser = True
        
        # 计算速度(如果使用卡尔曼滤波)
        if kf_x is not None and kf_y is not None:
            current_time = time.ticks_ms()
            dt = (current_time - kf_x.last_time) / 1000.0
            
            if dt > 0:
                vx = (x - kf_x.x_est[0]) / dt
                vy = (y - kf_y.x_est[0]) / dt
            else:
                vx, vy = 0, 0
                
            # 更新卡尔曼滤波器
            x_est, _ = kf_x.update([x, vx])
            y_est, _ = kf_y.update([y, vy])
            
            if draw:
                # 绘制原始点(绿色)和估计点(红色)
                img.draw_cross(x, y, color=image.COLOR_GREEN, size=3)
                img.draw_cross(int(x_est), int(y_est), color=image.COLOR_RED, size=3)
        elif draw:
            # 只绘制检测到的点
            img.draw_cross(x, y, color=image.COLOR_GREEN, size=3)
    
    if kf_x is not None and kf_y is not None:
        return x, y
    else:
        return x, y

def get_roi_from_largest_black_blob(img):
    """获取最大黑色色块边界的ROI，并返回放大120%后的大ROI
    
    参数:
        img: 输入图像
        
    返回:
        large_roi: 最大黑色色块边界的ROI [x, y, w, h]
        expanded_roi: 放大120%后的ROI [x, y, w, h]
        black_blob_vertices: 最大黑色色块的四个顶点坐标
    """
    # 检测黑色色块时关闭曝光设置
    cam.exp_mode(0)
    # cam.saturation(0)
    # 检测最大黑色色块
    black_blobs = img.find_blobs([black_threshold], 
                                pixels_threshold=1000,  # 最小像素数
                                area_threshold=1000,    # 最小面积
                                merge=True)            # 合并相邻色块
    
    large_roi = None
    expanded_roi = None
    black_blob_vertices = None
    
    if black_blobs:
        # 获取最大黑色色块
        largest_blob = max(black_blobs, key=lambda b: b.pixels())
        
        # 获取最大黑色色块的边界作为大ROI
        x, y, w, h = largest_blob.x(), largest_blob.y(), largest_blob.w(), largest_blob.h()
        large_roi = [x, y, w, h]
        
        # 计算放大120%后的ROI尺寸（宽度和高度各增加20%）
        expanded_w = int(w * 1.2)
        expanded_h = int(h * 1.2)
        
        # 计算放大后ROI的中心点（与原ROI中心点一致）
        center_x = x + w // 2
        center_y = y + h // 2
        
        # 计算放大后ROI的左上角坐标
        expanded_x = center_x - expanded_w // 2
        expanded_y = center_y - expanded_h // 2
        
        # 确保放大后的ROI在图像范围内
        expanded_x = max(0, expanded_x)
        expanded_y = max(0, expanded_y)
        expanded_w = min(img.width() - expanded_x, expanded_w)
        expanded_h = min(img.height() - expanded_y, expanded_h)
        
        expanded_roi = [expanded_x, expanded_y, expanded_w, expanded_h]
        
        # 获取并绘制最大黑色色块的四个顶点
        black_blob_vertices = get_black_blob_vertices(largest_blob)
        # 对黑色色块的顶点进行排序
        if black_blob_vertices:
            black_blob_vertices = sort_vertices(black_blob_vertices)
    
    return large_roi, expanded_roi, black_blob_vertices

def get_black_blob_vertices(blob):
    """获取黑色色块的四个顶点坐标
    
    参数:
        blob: 黑色色块对象
        
    返回:
        vertices: 四个顶点坐标的列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    # 获取黑色色块的边界
    x, y, w, h = blob.x(), blob.y(), blob.w(), blob.h()
    
    # 计算四个顶点的坐标
    top_left = (x, y)
    top_right = (x + w, y)
    bottom_left = (x, y + h)
    bottom_right = (x + w, y + h)
    
    return [top_left, top_right, bottom_left, bottom_right]

def draw_black_blob_vertices(img, vertices):
    """绘制黑色色块的四个顶点
    
    参数:
        img: 输入图像
        vertices: 四个顶点坐标的列表
    """
    if vertices:
        # 绘制四个顶点的圆圈
        for i, (x, y) in enumerate(vertices):
            img.draw_circle(x, y, 3, color=image.COLOR_BLUE, thickness=2)

def is_similar_rect(rect1, rect2, threshold=10):
    """判断两个矩形是否相似（位置和大小变化不超过阈值）
    
    参数:
        rect1: 第一个矩形 [x, y, w, h]
        rect2: 第二个矩形 [x, y, w, h]
        threshold: 相似度阈值
        
    返回:
        bool: 是否相似
    """
    if rect1 is None or rect2 is None:
        return False
    
    # 计算位置和尺寸的差异
    dx = abs(rect1[0] - rect2[0])
    dy = abs(rect1[1] - rect2[1])
    dw = abs(rect1[2] - rect2[2])
    dh = abs(rect1[3] - rect2[3])
    
    # 如果所有差异都在阈值内，则认为相似
    return dx <= threshold and dy <= threshold and dw <= threshold and dh <= threshold

def average_rects(rect_history):
    """计算历史矩形的平均值
    
    参数:
        rect_history: 矩形历史记录列表
        
    返回:
        averaged_rect: 平均后的矩形 [x, y, w, h]
    """
    if not rect_history:
        return None
    
    # 初始化累加器
    sum_x = 0
    sum_y = 0
    sum_w = 0
    sum_h = 0
    
    # 累加所有矩形的属性
    for rect in rect_history:
        sum_x += rect[0]
        sum_y += rect[1]
        sum_w += rect[2]
        sum_h += rect[3]
    
    # 计算平均值
    avg_x = int(sum_x / len(rect_history))
    avg_y = int(sum_y / len(rect_history))
    avg_w = int(sum_w / len(rect_history))
    avg_h = int(sum_h / len(rect_history))
    
    return [avg_x, avg_y, avg_w, avg_h]

def detect_largest_rectangle(img, roi):
    """在指定ROI区域内检测最大的矩形
    
    参数:
        img: 输入图像
        roi: 感兴趣区域 [x, y, w, h]
        
    返回:
        largest_rect: 最大矩形对象，如果未检测到则返回None
        rectangle_vertices: 最大矩形的四个顶点坐标
        small_rect_vertices: small_roi内最大矩形的四个顶点坐标
    """
    global big_roi, prev_rect, stable_counter, small_roi, prev_small_rect, small_stable_counter, rect_history
    
    if not roi:
        stable_counter = 0
        small_stable_counter = 0
        prev_rect = None
        prev_small_rect = None
        rect_history = []
        return None, None, None
    
    # 检测矩形时关闭曝光设置
    cam.exp_mode(0)
    # cam.saturation(0)
    # 使用find_rects函数在ROI区域内查找矩形
    rects = img.find_rects(roi=roi, threshold=5000)  # 降低阈值以提高检测灵敏度
    
    largest_rect = None
    rectangle_vertices = None
    small_rect_vertices = None
    
    if rects:
        # 找到面积最大的矩形
        largest_rect = max(rects, key=lambda r: r.w() * r.h())
        
        # 提取当前帧的矩形信息
        current_rect = [largest_rect.x(), largest_rect.y(), largest_rect.w(), largest_rect.h()]
        
        # 检查是否与上一帧相似
        if prev_rect is not None and is_similar_rect(current_rect, prev_rect):
            stable_counter += 1
        else:
            stable_counter = 0
        
        # 只有当矩形稳定出现足够次数时才认为有效
        if stable_counter >= MAX_STABLE_COUNT:
            big_roi = current_rect  # 保存为全局变量
            
            # 添加到历史记录
            rect_history.append(current_rect)
            if len(rect_history) > MAX_HISTORY:
                rect_history.pop(0)
            
            # 计算平均矩形
            avg_rect = average_rects(rect_history)
            
            # 绘制平均矩形（更稳定）
            if avg_rect:
                # img.draw_rect(avg_rect[0], avg_rect[1], avg_rect[2], avg_rect[3], 
                #              color=image.COLOR_RED, thickness=1)
                
                # 计算缩小95%的small_roi（基于平均矩形）
                x, y, w, h = avg_rect
                small_w = int(w * 0.95)
                small_h = int(h * 0.95)
                center_x = x + w // 2
                center_y = y + h // 2
                small_x = center_x - small_w // 2
                small_y = center_y - small_h // 2
                small_x = max(0, small_x)
                small_y = max(0, small_y)
                small_w = min(img.width() - small_x, small_w)
                small_h = min(img.height() - small_y, small_h)
                small_roi = [small_x, small_y, small_w, small_h]
                
                # 在small_roi区域内检测最大矩形
                small_rects = img.find_rects(roi=small_roi, threshold=5000)
                if small_rects:
                    # 找到small_roi区域内面积最大的矩形
                    largest_small_rect = max(small_rects, key=lambda r: r.w() * r.h())
                    current_small_rect = [largest_small_rect.x(), largest_small_rect.y(), 
                                        largest_small_rect.w(), largest_small_rect.h()]
                    
                    # 检查small_roi内矩形是否稳定
                    if prev_small_rect is not None and is_similar_rect(current_small_rect, prev_small_rect):
                        small_stable_counter += 1
                    else:
                        small_stable_counter = 0
                    
                    # 只有当small_roi内矩形也稳定时才绘制
                    if small_stable_counter >= MAX_STABLE_COUNT:
                        # 绘制small_roi区域内的最大矩形（蓝色）
                        # img.draw_rect(current_small_rect[0], current_small_rect[1], 
                        #             current_small_rect[2], current_small_rect[3],
                        #             color=image.COLOR_BLUE, thickness=1)
                        
                        # 获取small_roi区域内的矩形的四个顶点
                        small_rect_vertices = largest_small_rect.corners()
                        if small_rect_vertices:
                            small_rect_vertices = sort_vertices(small_rect_vertices)
                            # for i, (px, py) in enumerate(small_rect_vertices):
                            #     img.draw_circle(px, py, 3, color=image.COLOR_BLUE, thickness=2)
                        
                        # 更新上一帧small_rect信息
                        prev_small_rect = current_small_rect
                    
                    # 更新上一帧small_rect信息
                    prev_small_rect = current_small_rect
            
            # 获取并绘制矩形的四个顶点
            rectangle_vertices = largest_rect.corners()
            if rectangle_vertices:
                rectangle_vertices = sort_vertices(rectangle_vertices)
                
                # for i, (px, py) in enumerate(rectangle_vertices):
                #     img.draw_circle(px, py, 3, color=image.COLOR_BLUE, thickness=2)
            
            # 计算矩形的中心点
            center_x = current_rect[0] + current_rect[2] // 2
            center_y = current_rect[1] + current_rect[3] // 2
            img.draw_cross(center_x, center_y, color=image.COLOR_YELLOW, size=1)
        else:
            # 如果不够稳定，使用上一帧的结果
            if prev_rect is not None and stable_counter > 0:
                largest_rect = None  # 标记为无效，但保留prev_rect
                rectangle_vertices = None
        
        # 更新上一帧矩形信息
        prev_rect = current_rect
    
    return largest_rect, rectangle_vertices, small_rect_vertices

def sort_vertices(vertices):
    """
    对四个顶点进行排序，返回顺序为：左上、右上、右下、左下
    
    参数:
        vertices: 顶点坐标列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    
    返回:
        sorted_vertices: 排序后的顶点列表
    """
    if not vertices or len(vertices) != 4:
        return vertices
    
    # 计算四个顶点的中心点
    center_x = sum(v[0] for v in vertices) / 4
    center_y = sum(v[1] for v in vertices) / 4
    
    # 计算每个顶点相对于中心点的角度
    angles = []
    for x, y in vertices:
        # 计算相对于中心点的偏移
        dx = x - center_x
        dy = y - center_y
        # 计算角度（atan2返回弧度，转换为角度）
        angle = math.degrees(math.atan2(dy, dx))
        angles.append((x, y, angle))
    
    # 按照角度排序（从-180度到180度，顺时针方向）
    # 左上(-135度)、右上(-45度)、右下(45度)、左下(135度)
    sorted_angles = sorted(angles, key=lambda a: a[2])
    
    # 提取排序后的顶点坐标
    sorted_vertices = [(x, y) for x, y, _ in sorted_angles]
    
    return sorted_vertices

def calculate_midpoints(outer_vertices, inner_vertices):
    """
    计算外矩形和内矩形顶点之间的中点
    
    参数:
        outer_vertices: 外矩形四个顶点坐标
        inner_vertices: 内矩形四个顶点坐标
    
    返回:
        midpoints: 四个中点坐标的列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    if not outer_vertices or not inner_vertices or len(outer_vertices) != 4 or len(inner_vertices) != 4:
        return None
    
    midpoints = []
    for i in range(4):
        # 计算对应顶点的中点坐标
        x_mid = (outer_vertices[i][0] + inner_vertices[i][0]) // 2
        y_mid = (outer_vertices[i][1] + inner_vertices[i][1]) // 2
        midpoints.append((x_mid, y_mid))
    
    return midpoints

def draw_midpoints(img, midpoints):
    """
    绘制中点并打印中点坐标
    
    参数:
        img: 输入图像
        midpoints: 四个中点坐标的列表
    """
    if midpoints:
        for i, (x, y) in enumerate(midpoints):
            # 绘制黄色圆圈表示中点
            img.draw_circle(x, y, 3, color=image.COLOR_YELLOW, thickness=1)
            # 打印中点坐标
            # img.draw_string(x+5, y+5, f"{i+1}:({x},{y})", color=image.COLOR_YELLOW)

# 在函数外部定义全局状态标志
# 全局变量标记是否首次发送
is_first_send = True

def send_serial_data(x_red, y_red, midpoints, current_rect):
    """
    格式化并发送串口数据
    
    参数:
        x_red: 红色激光x坐标
        y_red: 红色激光y坐标
        midpoints: 四个中点坐标列表
        current_rect: 当前矩形信息 [x, y, w, h]
    """
    global is_first_send  # 声明使用全局变量
    
    # 检查激光坐标是否为0
    if x_red == 0 and y_red == 0:
        print("激光坐标为(0,0)，不发送数据")
        return
    
    # 格式化激光坐标（3位数字，不足补零）
    laser_x = f"{x_red:03d}"
    laser_y = f"{y_red:03d}"
    
    if is_first_send:
        # 首次发送需要完整数据
        if not midpoints or len(midpoints) != 4 or not current_rect:
            print("首次发送但缺少必要数据，等待完整数据")
            return
            
        # 计算矩形的中心点
        center_x = current_rect[0] + current_rect[2] // 2
        center_y = current_rect[1] + current_rect[3] // 2
        
        # 格式化矩形中心坐标
        rect_center_x = f"{center_x:03d}"
        rect_center_y = f"{center_y:03d}"

        # 格式化四个中点坐标
        midpoint_data = []
        for (x, y) in midpoints:
            midpoint_data.append(f"{x:03d}")
            midpoint_data.append(f"{y:03d}")

        # 构建完整数据
        data = f"@@{laser_x},{laser_y},{rect_center_x},{rect_center_y},{midpoint_data[0]},{midpoint_data[1]},{midpoint_data[2]},{midpoint_data[3]},{midpoint_data[4]},{midpoint_data[5]},{midpoint_data[6]},{midpoint_data[7]}#"
        is_first_send = False   # 标记已发送首次数据
        
        # 打印完整调试信息
        print(f"首次发送完整数据 - 激光({laser_x},{laser_y}) | 矩形中心({rect_center_x},{rect_center_y}) | 中点1({midpoint_data[0]},{midpoint_data[1]}) 中点2({midpoint_data[2]},{midpoint_data[3]}) 中点3({midpoint_data[4]},{midpoint_data[5]}) 中点4({midpoint_data[6]},{midpoint_data[7]})")
    else:
        # 后续发送简化数据（其他数据置零）
        data = f"@@{laser_x},{laser_y},000,000,000,000,000,000,000,000,000,000#"
        print(f"后续发送简化数据 - 激光({laser_x},{laser_y})")

    try:
        # 发送数据到串口
        serial.write(data.encode('ascii'))
        print(f"发送数据：{data}")
    except Exception as e:
        print(f"串口发送错误: {e}")

# 重置首次发送标志（可选）
def reset_first_send():
    global is_first_send
    is_first_send = True

# 主循环：持续捕获图像并进行处理
while True:
    try:
        t = time.ticks_ms()  # 记录开始时间
        img = cam.read()     # 读取一帧图像
        
        # 获取最大黑色色块边界的ROI和放大后的ROI以及黑色色块顶点
        large_roi, expanded_roi, black_blob_vertices = get_roi_from_largest_black_blob(img)
        
        midpoints = None
        current_rect = None
        if large_roi and expanded_roi:
            # 在expanded_roi区域内检测最大矩形
            largest_rect, rectangle_vertices, small_rect_vertices = detect_largest_rectangle(img, expanded_roi)
            
            if largest_rect:
                current_rect = [largest_rect.x(), largest_rect.y(), largest_rect.w(), largest_rect.h()]
            
            # 计算中点坐标（仅当两个顶点列表都存在时）
            if rectangle_vertices and small_rect_vertices:
                midpoints = calculate_midpoints(rectangle_vertices, small_rect_vertices)
                draw_midpoints(img, midpoints)

        x_red, y_red = red_jiguang(img)  # 检测激光点
        
        # 只有在激光坐标不为0时才发送数据
        if x_red != 0 or y_red != 0:
            send_serial_data(x_red, y_red, midpoints, current_rect)

        disp.show(img)                   # 显示处理后的图像
        
        # 控制帧率为约30FPS
        elapsed = time.ticks_ms() - t    # 计算处理耗时
        if elapsed < 30:
            time.sleep_ms(30 - elapsed)  # 延时补足30ms
            
    except Exception as e:
        print(f"主循环错误: {e}")








