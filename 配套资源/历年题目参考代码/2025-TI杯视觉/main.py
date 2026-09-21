from webbrowser import get
from maix import uart, app, time, camera, app, time, image
from MyUI import GUI
from MyUART import MyUtils
import constants    
import numpy as np
import cv2
from queue import Queue
from json import loads
# 添加标定管理器导入
from calibration_manager import CalibrationManager
from OverRectLapDetector import RectLapDetector
from SimpleShapeDetector import SimpleShapeDetector

#屏幕宽度和高度
_image_width  = 320
_image_height = 224
_btn_width  = _image_width//5
_btn_height = _image_height//5

class InforSeek:
    # 构造函数：初始化图像处理参数和内核
    def __init__(self, skernel: int=3, mkernel: int=3) -> None:
        
        #  初始化串口
        self.uart = MyUtils()  # 实例化自定义串口类

        self.gui = GUI() # 初始化ui显示


        self.last_machined_img = None       # 初始化上处理后的图片
        self._processed_img = None  # 用于存储处理后的图像
        self.SmoothKernel = skernel
        self.MorphologyKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mkernel, mkernel))

        # 初始化重叠检测检测器
        self.rect_lap_detector = RectLapDetector()

        # 初始化简单形状检测器
        self.simple_shape_detector = SimpleShapeDetector()

    def detect_overlap(self, _RawEdgeImg:image.Image, getDist: bool=False):
        """
        检测重叠部分并返回最小边长
        :param img: 输入的maix图像
        :return: 最小边长(mm)
        """
        # 转换为OpenCV格式
        cv_img = image.image2cv(_RawEdgeImg, copy=False)
        
        # 调用重叠检测器
        min_side_mm, squares, dist = self.rect_lap_detector.detect_overlap_squares(cv_img)
        
        if getDist:
            # 如果需要获取距离，返回最小边长和距离
            return (min_side_mm, dist)
        return min_side_mm

    def detect_simple_shape(self, _RawEdgeImg:image.Image):
        """
        检测简单几何形状（圆形、正方形、正三角形）
        :param _RawEdgeImg: 输入的maix图像
        :return: 检测结果字典，包含形状信息
        """
        # 转换为OpenCV格式
        cv_img = image.image2cv(_RawEdgeImg, copy=False)
        
        # 调用简单形状检测器
        result= self.simple_shape_detector.detect_simple_shape(cv_img)
        return result

    # 主函数：检测边界框并返回面积和角点坐标
    def make_sure_boundary(self, _RawEdgeImg:image.Image) -> float:
        
        # 对图像进行 opencv 预处理
        img_raw = image.image2cv(_RawEdgeImg, copy=False)   # 转换为 OpenCV 格式
        
        # 转换为灰度 为了方便滤波去除噪点
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

        # 进行中值滤波去除椒盐噪声
        img_medianBlur = cv2.medianBlur(img_gray, self.SmoothKernel)

        # 高斯滤波保留边缘
        img_gaussian = cv2.GaussianBlur(img_medianBlur, (self.SmoothKernel + 2 , self.SmoothKernel + 2), 0)

        # 闭运算 -- 白色底面，细小黑色为噪声
        img_close = cv2.morphologyEx(img_gaussian, cv2.MORPH_CLOSE, self.MorphologyKernel)

        # 保存此次预处理完成的图片
        self._processed_img = img_close  
        # self._processed_img = image.cv2image(img_close , copy=False)  # 转换回 maix 的 image 格式

        # 对图像进行角点检测
        median = np.median(self._processed_img)
        threshold1 = int(max(0, 0.95 * median))
        threshold2 = int(min(255, 1.35 * median))

        # 边缘检测
        img_edged = cv2.Canny(self._processed_img, threshold1, threshold2)

        # 查找轮廓角点
        contours, _ = cv2.findContours(img_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"一共寻找到{len(contours)}个轮廓\n")

        # 初始化用于记录四边形轮廓的列表，包含轮廓和面积信息
        valid_contours = []

        # 开始逐层记录，筛选出所有四边形轮廓并计算面积
        for contour in contours:    # 计算参数，进行多边形逼近
            # 获得近似值（与周长相关）：
            epsilon = 0.02 * cv2.arcLength(contour, True)

            """
            cv2.approxPolyDP(curve, epsilon, closed) -- 返回近似后的多边形点集
            # # curve: 输入的轮廓点集
            # # epsilon: 近似精度，值越小，近似越精确，一般与周长有关
            """
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 记录轮廓角点信息
            if len(approx) == 4:    # 如果有四个角点
                # 按照顺序对角点排序（左上，右上，右下，左下）
                corners = approx.reshape((4, 2))
                rect = np.zeros((4, 2), dtype="int")
                points_sum = corners.sum(axis=1)
                rect[0] = corners[np.argmin(points_sum)]
                rect[2] = corners[np.argmax(points_sum)]
                points_diff = np.diff(corners, axis=1)
                rect[1] = corners[np.argmin(points_diff)]
                rect[3] = corners[np.argmax(points_diff)]
                corners = rect
                
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                
                # 存储轮廓信息：(面积, 角点坐标)
                valid_contours.append((area, corners.tolist()))

        # 如果找到四边形轮廓，按面积排序
        if len(valid_contours) >= 4:  # 至少需要个边缘才能形成环形（外框和内框边缘各2个）
            # 按面积从大到小排序
            valid_contours.sort(key=lambda x: x[0], reverse=True)
            
            # 获取面积最大的轮廓（外环）和面积次大的轮廓（内环）
            outer_area = valid_contours[0][0]  # 外环面积
            outer_corners = valid_contours[0][1]  # 外环角点
            inner_corners = valid_contours[1][1]  # 内环角点

            # 得到内轮廓的内环角点

            inner_edge_corners = valid_contours[3][1]  # 内轮廓内环角点 -- 用于进行内部区域定界

            # 使用 average_points 函数计算外轮廓内外环角点的平均值
            averaged_corners = self.average_points(outer_corners, inner_corners)
            
            # 以外环面积作为返回值 -- 理论上为a4纸张的像素值
            averaged_area = outer_area

            print(f"外环面积: {outer_area:.2f}")

            # 返回 外轮廓面积、 外轮廓角点信息 和 内轮廓内环角点信息
            return (averaged_area, averaged_corners, inner_edge_corners)
            
        return (0, [], [])

    # 计算两个列表中对应坐标点的平均值
    def average_points(self, list1, list2) -> list:
        """
        计算两个列表中对应坐标点的平均值
        
        参数:
            list1: 第一个点列表，格式为 [[x1,y1], [x2,y2], ...]
            list2: 第二个点列表，格式为 [[x1,y1], [x2,y2], ...]
            
        返回:
            包含平均坐标的新列表，格式为 [[avg_x1, avg_y1], [avg_x2, avg_y2], ...]
        """
        # 检查两个列表长度是否一致
        if len(list1) != len(list2):
            raise ValueError(f"两个列表长度不一致: list1长度={len(list1)}, list2长度={len(list2)}")
        
        # 如果列表为空，直接返回空列表
        if len(list1) == 0:
            return []
        
        result = []
        for point1, point2 in zip(list1, list2):
            # 计算x坐标平均值,返回整数
            avg_x = round((point1[0] + point2[0]) / 2.0)
            # 计算y坐标平均值,返回整数
            avg_y = round((point1[1] + point2[1]) / 2.0)
            # 添加到结果列表
            result.append([avg_x, avg_y])
        
        return result

    # 距离计算：基于面积反比关系估算物体距离
    def calculate_distance(self, center_area, reference_area, reference_distance):
        """
        基于环形中心四边形面积计算距离
        :param center_area: 检测到的环形面积
        :param reference_area: 参考距离下的中心四边形面积 px（实际标定值）
        :param reference_distance: 参考距离，实际标定值（单位：mm）
        :return: 估算的距离
        """
        if center_area <= 0:
            return -1  # 无效面积
        
        # 面积与距离成反比关系（距离越远，面积越小）
        distance = reference_distance * np.sqrt(reference_area / center_area)
        
        return distance

    # 几何测量：检测ROI内的几何形状并计算尺寸
    def measure_inner_geometry_rect(self, boundary_corners: list, boundary_real_width: float, boundary_real_height: float) -> list:
        """
        测量环形边界内的几何形状尺寸（基于矩形内边界框标定）
        :param boundary_corners: 边界框的四个角点坐标 [[x,y], [x,y], [x,y], [x,y]]
        :param boundary_real_width: 内边界框的实际宽度（mm）
        :param boundary_real_height: 内边界框的实际高度（mm）
        :return: 包含检测到的几何形状列表，每个元素是一个字典
        """
        
        # 计算边界框的像素长宽 -- 使用两个长、宽的均值作为最终结果
        boundary_pixel_width, boundary_pixel_height = self._calculate_boundary_pixel_dimensions(boundary_corners)
        
        # 计算像素到实际尺寸的转换比例（mm/pixel）
        pixel_to_mm_ratio_x = boundary_real_width / boundary_pixel_width if boundary_pixel_width > 0 else 0
        pixel_to_mm_ratio_y = boundary_real_height / boundary_pixel_height if boundary_pixel_height > 0 else 0
        
        # 提取边界框的边界坐标
        x_coords = [point[0] for point in boundary_corners]
        y_coords = [point[1] for point in boundary_corners]
        
        # 获取边界框的范围，并留出一定的边距避免边界检测
        margin = 1  # 边距像素
        x_min = max(0, min(x_coords) + margin)
        y_min = max(0, min(y_coords) + margin)
        x_max = min(_image_width, max(x_coords) - margin)
        y_max = min(_image_height, max(y_coords) - margin)

        # 确保ROI区域的宽度和高度都是偶数
        roi_width = x_max - x_min
        roi_height = y_max - y_min
        
        # 如果宽度是奇数，减少1像素使其变为偶数
        if roi_width % 2 == 1:
            x_max -= 1
            roi_width = x_max - x_min
        
        # 如果高度是奇数，减少1像素使其变为偶数
        if roi_height % 2 == 1:
            y_max -= 1
            roi_height = y_max - y_min

        # 确保ROI区域有效
        if x_max <= x_min or y_max <= y_min:
            print("无效的ROI区域")
            return []
        
        # 提取ROI区域 (使用OpenCV的数组切片方式裁剪)
        if self._processed_img is not None:
            roi_img = self._processed_img[y_min:y_max, x_min:x_max]  # OpenCV格式: [y1:y2, x1:x2]
        else:
            print("错误图像区域")
            return []

        # 在ROI区域内检测轮廓
        median = np.median(roi_img)
        threshold1 = int(max(0, 0.95 * median))
        threshold2 = int(min(255, 1.35 * median))

        # 边缘检测
        roi_edged = cv2.Canny(roi_img, threshold1, threshold2)
        
        # 调试理论 -- 装换为 OpenCV 格式 变换为maixcam
        # img_show = image.cv2image(roi_edged, copy=False)
        # self.gui.show(img_show)  # 显示ROI边缘检测结果

        # time.sleep_ms(200)  # 等待显示更新

        # 查找轮廓角点
        contours, _ = cv2.findContours(roi_edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 存储检测到的几何形状信息列表
        detected_shapes = []
        
        print(f"ROI区域: 左上({x_min}, {y_min}) -> 右下({x_max}, {y_max})")
        # print(f"边界框像素尺寸: {boundary_pixel_width:.2f} x {boundary_pixel_height:.2f} px")
        # print(f"像素到毫米比例: X={pixel_to_mm_ratio_x:.4f}, Y={pixel_to_mm_ratio_y:.4f} mm/px")
        
        for contour in contours:
            # 计算轮廓面积（过滤太小的轮廓）
            area = cv2.contourArea(contour)
            if area < 15:  # 过滤掉较小的轮廓
                continue
                
            # 多边形逼近
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 根据角点数量判断几何形状
            if len(approx) == 3:  # 三角形
                shape_result = self._analyze_triangle_rect(approx, pixel_to_mm_ratio_x, pixel_to_mm_ratio_y, x_min, y_min)
                if shape_result:
                    detected_shapes.append(shape_result)
            elif len(approx) == 4:  # 矩形
                shape_result = self._analyze_rectangle_rect(approx, pixel_to_mm_ratio_x, pixel_to_mm_ratio_y, x_min, y_min)
                if shape_result:
                    detected_shapes.append(shape_result)
            else:  # 可能是圆形或其他曲线形状
                shape_result = self._analyze_circle_rect(contour, pixel_to_mm_ratio_x, pixel_to_mm_ratio_y, x_min, y_min)
                if shape_result:
                    detected_shapes.append(shape_result)
        
        return detected_shapes
    
    # 尺寸计算：均值计算边界框的像素长宽
    def _calculate_boundary_pixel_dimensions(self, boundary_corners):
        """计算边界框的像素长宽"""
        corners = np.array(boundary_corners)
        
        # 计算宽度（上边和下边的平均值）
        top_width = np.sqrt((corners[1][0] - corners[0][0])**2 + (corners[1][1] - corners[0][1])**2)
        bottom_width = np.sqrt((corners[2][0] - corners[3][0])**2 + (corners[2][1] - corners[3][1])**2)
        avg_width = (top_width + bottom_width) / 2
        
        # 计算高度（左边和右边的平均值）
        left_height = np.sqrt((corners[3][0] - corners[0][0])**2 + (corners[3][1] - corners[0][1])**2)
        right_height = np.sqrt((corners[2][0] - corners[1][0])**2 + (corners[2][1] - corners[1][1])**2)
        avg_height = (left_height + right_height) / 2
        
        return avg_width, avg_height
    
    # 三角形分析：计算三角形的边长和面积
    def _analyze_triangle_rect(self, triangle_points, pixel_to_mm_ratio_x, pixel_to_mm_ratio_y, offset_x, offset_y):
        """分析三角形的尺寸"""
        points = np.array(triangle_points).reshape((3, 2))
        
        # 转换ROI坐标到全局坐标
        points[:, 0] += offset_x  # X坐标偏移
        points[:, 1] += offset_y  # Y坐标偏移
        
        # 计算三边长度（像素），并分别使用X和Y方向的转换比例
        # 边长1: 点0到点1
        dx1 = points[1][0] - points[0][0]
        dy1 = points[1][1] - points[0][1]
        side1 = np.sqrt((dx1 * pixel_to_mm_ratio_x)**2 + (dy1 * pixel_to_mm_ratio_y)**2)
        
        # 边长2: 点1到点2
        dx2 = points[2][0] - points[1][0]
        dy2 = points[2][1] - points[1][1]
        side2 = np.sqrt((dx2 * pixel_to_mm_ratio_x)**2 + (dy2 * pixel_to_mm_ratio_y)**2)
        
        # 边长3: 点2到点0
        dx3 = points[0][0] - points[2][0]
        dy3 = points[0][1] - points[2][1]
        side3 = np.sqrt((dx3 * pixel_to_mm_ratio_x)**2 + (dy3 * pixel_to_mm_ratio_y)**2)
        
        # 计算面积（mm²）- 需要同时考虑X和Y方向的缩放
        area_pixels = cv2.contourArea(points)
        area_mm2 = area_pixels * pixel_to_mm_ratio_x * pixel_to_mm_ratio_y
        
        return {
            '类型': '三角形',
            '边长1 (mm)': side1,
            '边长2 (mm)': side2,
            '边长3 (mm)': side3,
            '面积 (mm²)': area_mm2,
            '全局坐标': points.tolist()
        }
    
    # 矩形分析：计算矩形的长宽和面积
    def _analyze_rectangle_rect(self, rectangle_points, pixel_to_mm_ratio_x, pixel_to_mm_ratio_y, offset_x, offset_y):
        """分析矩形的尺寸"""
        points = np.array(rectangle_points).reshape((4, 2))
        
        # 转换ROI坐标到全局坐标
        points[:, 0] += offset_x  # X坐标偏移
        points[:, 1] += offset_y  # Y坐标偏移
        
        # 计算四边长度（mm），分别使用X和Y方向的转换比例
        # 边1: 点0到点1
        dx1 = points[1][0] - points[0][0]
        dy1 = points[1][1] - points[0][1]
        side1 = np.sqrt((dx1 * pixel_to_mm_ratio_x)**2 + (dy1 * pixel_to_mm_ratio_y)**2)
        
        # 边2: 点1到点2
        dx2 = points[2][0] - points[1][0]
        dy2 = points[2][1] - points[1][1]
        side2 = np.sqrt((dx2 * pixel_to_mm_ratio_x)**2 + (dy2 * pixel_to_mm_ratio_y)**2)
        
        # 边3: 点2到点3
        dx3 = points[3][0] - points[2][0]
        dy3 = points[3][1] - points[2][1]
        side3 = np.sqrt((dx3 * pixel_to_mm_ratio_x)**2 + (dy3 * pixel_to_mm_ratio_y)**2)
        
        # 边4: 点3到点0
        dx4 = points[0][0] - points[3][0]
        dy4 = points[0][1] - points[3][1]
        side4 = np.sqrt((dx4 * pixel_to_mm_ratio_x)**2 + (dy4 * pixel_to_mm_ratio_y)**2)
        
        # 获取长和宽
        width = (side1 + side3) / 2
        height = (side2 + side4) / 2
        
        # 计算面积（mm²）
        area_pixels = cv2.contourArea(points)
        area_mm2 = area_pixels * pixel_to_mm_ratio_x * pixel_to_mm_ratio_y
        
        return {
            '类型': '矩形',
            '宽度 (mm)': width,
            '高度 (mm)': height,
            '面积 (mm²)': area_mm2,
            '全局坐标': points.tolist()
        }
    
    # 圆形分析：计算圆形的半径、直径和面积
    def _analyze_circle_rect(self, circle_contour, pixel_to_mm_ratio_x, pixel_to_mm_ratio_y, offset_x, offset_y):
        """分析圆形的尺寸"""
        # 计算外接圆
        (x, y), radius_pixels = cv2.minEnclosingCircle(circle_contour)
        
        # 转换ROI坐标到全局坐标
        center_global = [int(x + offset_x), int(y + offset_y)]
        
        # 对于圆形，使用X和Y方向转换比例的平均值
        avg_pixel_to_mm_ratio = (pixel_to_mm_ratio_x + pixel_to_mm_ratio_y) / 2
        radius_mm = radius_pixels * avg_pixel_to_mm_ratio
        
        # 计算面积（mm²）
        area_pixels = cv2.contourArea(circle_contour)
        area_mm2 = area_pixels * pixel_to_mm_ratio_x * pixel_to_mm_ratio_y
        
        return {
            '类型': '圆形',
            '半径 (mm)': radius_mm,
            '直径 (mm)': 2 * radius_mm,
            '面积 (mm²)': area_mm2,
            '全局坐标': center_global,
            '半径 (像素)': radius_pixels
        }

    # 发送数据
    def send_data(self, data_dict: dict) -> bool:
        """
            串口传输数据函数 -- json分装
        """
        if not self.uart.send_data(data_dict):
            return False
        return True

    # 设置串口数据接收回调函数
    def set_cbf(self, cbf):
        """
            设置串口数据接收回调函数
        """
        self.uart.received_callback(cbf)


# 指挥函数
def data_processing(serial: uart.UART,bytes_data : bytes):
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

        result = {key: values for key, values in dir_data.items() if values != '-1'}
        if queue.qsize() < 1:   # 只保留一条指令
            queue.put(result)

info = InforSeek()   # 实例化基本信息类
info.set_cbf(data_processing)

# 初始化串口及指挥线程
queue = Queue()

# 初始化发送数据集
send_data_dict = {}
# print = info.uart._print

# 主程序：摄像头采集、图像处理、几何检测和显示
def main():

    cam = camera.Camera(_image_width, _image_height)    # 初始化摄像头分辨率
    cam.skip_frames(30)     # 跳过开头的30帧
    
    # 初始化标定状态
    toCalibrate = [False, 1000.0]  # 标定标志位和标定距离，默认为1000mm

    # 创建标定管理器实例
    calibrator = CalibrationManager()

    order = {}
    
    while not app.need_exit():
        time.fps_start()

        img = cam.read()
        img_show = img.copy()
        
        # 初始化题目指令
        title_sort = None
        # 初始化目标数字
        findNumber = None

        if queue.qsize() > 0:  # 如果队列中有数据
            # 获取队列中的数据
            order = queue.get()
            
            if len(order.keys()) > 0:
                print(f"收到指令: {order}")

                # 处理指令
                if 'scan' in order.keys():  # 正常扫描
                    if 'ID' in order.keys() and order['scan'] == '3':    # 寻找数字的扫描
                        findNumber = order['ID']   # 寻找数字扫描的题目
                    elif order['scan'] == '2':  # 只需要寻找重叠部分
                        title_sort = order['scan']  # 准备完成重叠部分
                    elif order['scan'] == '1':  # 完成基础题
                        title_sort = order['scan']  # 准备完成基础题
                elif 'calibratio' in order.keys():   # 进行重新的标定
                    print("收到标定指令，开始标定\n")
                    # 进行标定操作 -- 抬起标定标志位，并记录本次标定的距离
                    toCalibrate = [True, int(order['calibratio'])]
                else:
                    print("未收到有效指令，跳过本次循环\n")
                    info.gui.run(img_show)
                    continue
        else:
            # print("未收到指令，跳过本次循环\n")
            info.gui.run(img_show)
            continue

        if title_sort == '1':
            # 初始化发送数据字典
        
            send_data_dict.clear()
            dist = 0
            x = 0
            
            # 得到边界框的像素面积 和 角点坐标
            Boundary = info.make_sure_boundary(img)
            
            # 获取边界框的像素面积
            Boundary_area = Boundary[0]
            print(f"用于计算环形的像素面积: {Boundary_area:.2f} px²")

            if Boundary_area == 0:  # 如果没有检测到环形
                print("未检测到环形，跳过本次循环")
                info.gui.run(img_show)
                continue

            # 获取角点坐标进行批注 [(x,y), (x,y), (x,y), (x,y)]
            oute_edge_corners = Boundary[1]
            
            # 对图像进行外边框的批注
            img_show.draw_line(oute_edge_corners[0][0], oute_edge_corners[0][1], oute_edge_corners[1][0], oute_edge_corners[1][1], image.COLOR_WHITE, 1)
            img_show.draw_line(oute_edge_corners[1][0], oute_edge_corners[1][1], oute_edge_corners[2][0], oute_edge_corners[2][1], image.COLOR_WHITE, 1)
            img_show.draw_line(oute_edge_corners[2][0], oute_edge_corners[2][1], oute_edge_corners[3][0], oute_edge_corners[3][1], image.COLOR_WHITE, 1)
            img_show.draw_line(oute_edge_corners[3][0], oute_edge_corners[3][1], oute_edge_corners[0][0], oute_edge_corners[0][1], image.COLOR_WHITE, 1)

            if Boundary_area > 0:   # 面积大于0，表示检测到外边框，进行测距及其他检测
                # 计算距离
                reference_area = constants.REFERENCE_AREA  # 外接框的标定像素面积 (px^2)
                reference_distance = constants.REFERENCE_DISTANCE  # 外接框的标定参考距离 (mm)
                distance = info.calculate_distance(Boundary_area, reference_area, reference_distance)
                print(f"检测到外边框，估算距离: {distance:.2f} cm")

                dist = int(distance * 10)  # 转换为mm单位

                # 测量内部几何形状（使用外边框进行统一的标定）
                boundary_real_width = constants.A4_REAL_WIDTH_MM   # 边界框实际宽度 (mm)
                boundary_real_height = constants.A4_REAL_LENGTH_MM  # 边界框实际高度 (mm)

                # 显示距离信息
                img_show.draw_string(10, 10, f"Dist: {distance:.1f}cm", color=image.COLOR_YELLOW)

                # 对图像进行内部几何形状检测 -- 使用内轮廓内环进行统一的标定
                geometry_results = info.measure_inner_geometry_rect(Boundary[2], boundary_real_width, boundary_real_height)
                
                # 绘制几何形状检测结果
                if geometry_results:
                    for i, result in enumerate(geometry_results):
                        if result:
                            shape_type = result['类型']
                            y_pos = 70 + i * 20
                            
                            # 根据形状类型显示对应的尺寸信息（cm单位）
                            if shape_type == '三角形':
                                side1_mm = result['边长1 (mm)']
                                side1_cm = side1_mm / 10
                                display_text = f"三角形: {side1_cm:.1f}cm"
                                print(f"检测到三角形，边长1: {side1_cm:.2f} cm")

                                # 发送三角形边长数据
                                x = int(side1_mm)
                            elif shape_type == '矩形':
                                width_mm = result['宽度 (mm)']
                                height_mm = result['高度 (mm)']
                                width_cm = width_mm / 10
                                height_cm = height_mm / 10
                                display_text = f"矩形: {width_cm:.1f}x{height_cm:.1f}cm"
                                print(f"检测到矩形，尺寸: {width_cm:.2f} x {height_cm:.2f} cm")

                                # 发送矩形宽度和高度数据
                                x = int(width_mm)
                            elif shape_type == '圆形':
                                diameter_mm = result['直径 (mm)']
                                diameter_cm = diameter_mm / 10
                                display_text = f"圆形: D={diameter_cm:.1f}cm"
                                print(f"检测到圆形，直径: {diameter_cm:.2f} cm")

                                # 发送圆形直径数据
                                x = int(diameter_mm)
                            else:
                                display_text = f"{shape_type}"
                            
                            img_show.draw_string(10, y_pos, display_text, color=image.COLOR_GREEN)

            # # 检测形状
            # result= info.detect_simple_shape(img.copy())

            # if result['type']:
            #     print(f"检测到 {result['type']}, 尺寸: {result['size_mm']:.2f} mm, 距离: {result['distance_cm']:.2f} cm")
            #     # 装换为全局坐标输出
            #     print(f"全局坐标: {result['center'][0]}, {result['center'][1]}")
            # 将几何图形添加到发送数据字典
            send_data_dict['x'] = f"{x}" if x > 0 else "0"
            send_data_dict['dist'] = f"{dist}" if dist > 0 else "0"
            
            
            # 发送数据
            if send_data_dict['x'] != "0" and send_data_dict['dist'] != "0":
                info.send_data(send_data_dict)
                print(f"发送数据: {send_data_dict}")
            else:
                print(f"未检测到，重新识别")
                queue.put({'scan': '1'})  # 重新添加指令到队列

            info.gui.run(img_show)
            time.sleep_ms(200)

            # # 调试使用， 一直收到此指令集进行基础题的检测 -- 存放在队列中
            # queue.put({'scan': '1'})  # 重新添加指令到队列
        
        if title_sort == '2':   # 准备完成重叠部分的区分扫描 -- 返回重叠部分的最小边长
            # 初始化发送数据
            send_data_dict.clear()
            img_lap = img.copy()  # 复制图像用于重叠检测
            
            # 检测重叠情况下的距离和最小边长
            _data = info.detect_overlap(img_lap, True)

            # 添加到发送数据
            min_side_mm = round(_data[0])  # 重叠部分最小边长 (cm)
            dist = round(_data[1])  # 现在的实际距离
            send_data_dict['x'] = f"{min_side_mm}" if min_side_mm > 0 else "0"
            send_data_dict['dist'] = f"{dist*10}" if dist > 0 else "0"

            # 发送数据
            if send_data_dict['x'] != "0" and send_data_dict['dist'] != "0":
                info.send_data(send_data_dict)
                print(f"发送重叠检测数据: {send_data_dict}")
            else:
                print(f"未检测到，重新识别")
                queue.put({'scan': '2'})  # 重新添加指令到队列

            # 显示结果
            img_show.draw_string(10, 10, f"Min Overlap: {min_side_mm:.2f}mm", color=image.COLOR_YELLOW)
            info.gui.run(img_show)
            time.sleep_ms(200)
            

            # # 调试使用， 一直收到此指令集进行重叠部分的检测 -- 存放在队列中
            # queue.put({'scan': '2'})  # 重新添加指令到队列

            pass

        if findNumber is not None:  # 准备寻找指定的数字目标
            # 初始化发送数据
            send_data_dict.clear()
            findNumber = None
            pass

        if toCalibrate[0]:  # 如果需要进行标定
            # 进行标定操作 -- 抬起标定标志位，并记录本次标定的距离
            toCalibrate[0] = False

            # 开始计算A4纸的像素面积、长度和宽度像素
            calibration_distance =float(toCalibrate[1])
            print(f"开始标定，标定距离: {calibration_distance} mm")

            # 计算像素面积
            frame = image.image2cv(img.copy(), copy=False)
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_img, constants.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            # 二值化后进行形态学操作 先膨胀后腐蚀（闭运算），消除噪声和填补空洞
            binary = cv2.dilate(binary, None, iterations=constants.MORPH_ITERATIONS)
            binary = cv2.erode(binary, None, iterations=constants.MORPH_ITERATIONS)
            
            # 对图像进行轮廓层次对分析
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
            candidate_pairs = []        # 存储潜在的父子轮廓对
            min_side_mm = 0
            
            if hierarchy is not None:
                for i, h_info in enumerate(hierarchy[0]):
                    if h_info[2] != -1: # 有孩子的是潜在的a4纸轮廓
                        p_cnt = contours[i] # # 获取当前父轮廓的点集
                        p_peri = cv2.arcLength(p_cnt, True)  # 计算父轮廓的周长（用于后续多边形逼近 True表示轮廓是闭合的）
                        p_approx = cv2.approxPolyDP(p_cnt, 0.02 * p_peri, True)  # 使用Douglas-Peucker算法逼近多边形，0.02*周长作为逼近精度阈值
                        if len(p_approx) == 4 and cv2.contourArea(p_cnt) > constants.PARENT_CONTOUR_MIN_AREA:
                            candidate_pairs.append((contours[i], contours[h_info[2]]))

            if candidate_pairs: 
                # 选择面积最大的A4纸轮廓
                selected_parent_contour, selected_child_contour = max(candidate_pairs, key=lambda p: cv2.contourArea(p[0]))
                rect_parent = cv2.minAreaRect(selected_parent_contour)      # 计算轮廓的最小外接旋转矩形    ((center_x, center_y), (width, height), angle )

                # 获取最大边界框的面积、长、宽
                boundary_area = cv2.contourArea(selected_parent_contour)  # 轮廓面积 (像素²)
                current_a_pixel_length = max(rect_parent[1])              # 长边像素宽度
                current_a_pixel_width = min(rect_parent[1])             # 短边像素宽度
                

                # 将标定得到的边界框进行处理
                if boundary_area > 0:
                    # 创建修改字典 -- 常量值为键、修改为值

                    calibration_data = {
                        'REFERENCE_DISTANCE': calibration_distance,  # 标定距离
                        'REFERENCE_AREA': boundary_area,        # 边界框面积
                        'PX_A4_WIDTH': round(current_a_pixel_width),  # 像素宽度
                        'PX_A4_LENGTH': round(current_a_pixel_length) # 像素高度
                    }
                    # 更新常量值
                    calibrator.update_constants_file(calibration_data)
                    print(f"标定成功，更新常量值: {calibration_data}")
                else:
                    print("标定失败,自添加指令")
                    # 如果标定失败，添加指令到队列
                    queue.put({'Calibratio': calibration_distance})
            else:
                print("未检测到有效的A4纸轮廓，标定失败")
                # 如果没有检测到有效的A4纸轮廓，添加指令到队列
                queue.put({'Calibratio': calibration_distance})
              

if __name__ == '__main__':
    main()
