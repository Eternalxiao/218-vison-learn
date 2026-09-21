import cv2
import numpy as np
import math
import constants

class SimpleShapeDetector:
    # --- 全局配置参数 ---
    # 物理参数
    KNOWN_A4_WIDTH_CM = 29.7  # A4纸横向宽度（物理参考值）
    MIN_SIDE_CM = 9.5         # 最小边长（厘米）
    MAX_SIDE_CM = 16.5        # 最大边长（厘米）
    ROI_SHRINK_PIXELS = constants.ROI_SHRINK_PIXELS
    ANGLE_TOLERANCE = constants.ANGLE_TOLERANCE
    DEDUPLICATION_TOLERANCE = constants.DEDUPLICATION_TOLERANCE
    SQUARE_ASPECT_RATIO_TOLERANCE = 1.35
    
    # 图像处理参数
    BINARY_THRESHOLD = constants.BINARY_THRESHOLD
    MORPH_ITERATIONS = constants.MORPH_ITERATIONS
    PARENT_CONTOUR_MIN_AREA = constants.PARENT_CONTOUR_MIN_AREA
    CHILD_CONTOUR_MIN_AREA = constants.CHILD_CONTOUR_MIN_AREA
    
    # 霍夫圆参数
    HOUGH_PARAM_1 = 200
    HOUGH_PARAM_2 = 4
    MIN_CIRCLE_RADIUS = 8
    MAX_CIRCLE_RADIUS = 280
    
    # 核心校准参数 - 统一使用面积标定
    SYSTEM_OFFSET_CM = constants.SYSTEM_OFFSET_CM
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_angle(p1, p2, p3):
        """计算三点之间的夹角"""
        v1 = p1 - p2
        v2 = p3 - p2
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        det = v1[0] * v2[1] - v1[1] * v2[0]
        angle = np.degrees(np.arctan2(det, dot))
        return angle + 360 if angle < 0 else angle

    @staticmethod
    def get_dist(p1, p2):
        """计算两点之间的欧氏距离"""
        return np.linalg.norm(p1 - p2)
    
    # 距离计算：基于面积反比关系估算物体距离
    @staticmethod
    def calculate_distance(center_area, reference_area=constants.REFERENCE_AREA, reference_distance=constants.REFERENCE_DISTANCE):
        """
        基于A4纸面积计算距离 - 统一面积标定方法
        :param center_area: 检测到的A4纸面积 (px²)
        :param reference_area: 参考距离下的A4纸面积 (px²)（实际标定值）
        :param reference_distance: 参考距离，实际标定值（单位：cm）
        :return: 估算的距离 (cm)
        """
        if center_area <= 0:
            return -1  # 无效面积
        
        # 面积与距离成反比关系（距离越远，面积越小）
        distance = reference_distance * np.sqrt(reference_area / center_area)
        
        return distance
    
    def detect_simple_shape(self, frame):
        """
        检测简单几何形状（圆形、正方形、正三角形）
        :param frame: 输入的BGR图像
        :return: 检测结果字典，包含形状信息
        """
        # 初始化结果
        result = {
            'type': None,
            'size_mm': 0,
            'center': None,
            'corners': None,
            'distance_cm': 0
        }
        
        # 图像预处理
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_img, self.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.dilate(binary, None, iterations=self.MORPH_ITERATIONS)
        binary = cv2.erode(binary, None, iterations=self.MORPH_ITERATIONS)
        
        # 查找轮廓层次
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidate_pairs = []
        
        if hierarchy is not None:
            for i, h_info in enumerate(hierarchy[0]):
                if h_info[2] != -1:  # 有子轮廓的轮廓
                    p_cnt = contours[i]
                    p_peri = cv2.arcLength(p_cnt, True)
                    p_approx = cv2.approxPolyDP(p_cnt, 0.02 * p_peri, True)
                    if len(p_approx) == 4 and cv2.contourArea(p_cnt) > self.PARENT_CONTOUR_MIN_AREA:
                        candidate_pairs.append((contours[i], contours[h_info[2]]))
        
        if not candidate_pairs:
            return result
        
        # 选择最大的A4纸轮廓
        selected_parent_contour, selected_child_contour = max(candidate_pairs, key=lambda p: cv2.contourArea(p[0]))
        
        # 使用面积标定计算距离
        current_area = cv2.contourArea(selected_parent_contour)
        print(f"当前A4纸面积: {current_area} px², 参考面积: {constants.REFERENCE_AREA} px²")

        # 计算距离 - 统一使用面积标定
        final_distance_cm = self.calculate_distance(current_area)
        
        # 计算像素到实际尺寸的转换比例
        # 距离比例：实际距离/参考距离
        distance_ratio = final_distance_cm / constants.REFERENCE_DISTANCE
        # 面积比例
        area_ratio = current_area / constants.REFERENCE_AREA
        # 像素到厘米的转换比例 = (A4实际宽度cm / 参考距离cm) * 距离比例 / sqrt(面积比例)
        pixel_to_cm_ratio = (constants.A4_REAL_WIDTH_MM / 10 / constants.REFERENCE_DISTANCE) * distance_ratio / np.sqrt(area_ratio)

        # 获取ROI区域
        x, y, w, h = cv2.boundingRect(selected_child_contour)
        x_shrunk, y_shrunk = x + self.ROI_SHRINK_PIXELS, y + self.ROI_SHRINK_PIXELS
        w_shrunk, h_shrunk = w - 2 * self.ROI_SHRINK_PIXELS, h - 2 * self.ROI_SHRINK_PIXELS
        
        if w_shrunk <= 0 or h_shrunk <= 0:
            return result
        
        roi_gray = gray_img[y_shrunk:y_shrunk+h_shrunk, x_shrunk:x_shrunk+w_shrunk]
        
        # 在roi内进行形状检测
        # 检测圆形
        circle = self._detect_circle(roi_gray)
        if circle:
            (cx, cy), radius = circle
            # 转换到全局坐标
            cx_global = int(cx + x_shrunk)
            cy_global = int(cy + y_shrunk)
            
            # 计算实际尺寸 - 使用统一的像素到厘米转换比例
            pixel_diameter = radius * 2
            diameter_mm = pixel_diameter * pixel_to_cm_ratio * 10  # 转换为毫米
            
            result = {
                'type': 'circle',
                'size_mm': diameter_mm,
                'center': (cx_global, cy_global),
                'radius': radius,
                'distance_cm': final_distance_cm
            }
            return result
        
        # 检测正方形或三角形
        shape = self._detect_polygon(roi_gray, pixel_to_cm_ratio)
        if shape:
            shape_type, side_mm, corners = shape
            
            # 转换到全局坐标
            global_corners = [point + [x_shrunk, y_shrunk] for point in corners]
            
            # 计算中心点
            center = np.mean(global_corners, axis=0).astype(int)
            
            result = {
                'type': shape_type,
                'size_mm': side_mm,
                'center': (center[0], center[1]),
                'corners': global_corners,
                'distance_cm': final_distance_cm
            }
            return result
        
        return result

    def _detect_circle(self, roi_gray):
        """在ROI区域内检测圆形"""
        roi_blurred = cv2.GaussianBlur(roi_gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            roi_blurred, cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=roi_gray.shape[1] / 4,
            param1=self.HOUGH_PARAM_1,
            param2=self.HOUGH_PARAM_2,
            minRadius=self.MIN_CIRCLE_RADIUS,
            maxRadius=self.MAX_CIRCLE_RADIUS
        )
        
        if circles is not None:
            print(f"检测到圆形, 数量: {len(circles[0])},")
            circles = np.uint16(np.around(circles))
            c = circles[0, 0]
            # print(f"圆心坐标: ({c[0]}, {c[1]}), 半径: {c[2]} px")
            return (c[0], c[1]), c[2]
        
        return None
    
    def _detect_polygon(self, roi_gray, pixel_to_cm_ratio):
        """在ROI区域内检测多边形（正方形或三角形）"""
        _, roi_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < self.CHILD_CONTOUR_MIN_AREA:
                continue
                
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            vertices = len(approx)
            
            # 检测三角形
            if vertices == 3:
                return self._analyze_triangle(approx, pixel_to_cm_ratio)
            
            # 检测正方形
            elif vertices == 4:
                square = self._analyze_square(approx, pixel_to_cm_ratio)
                if square:
                    return square
        
        return None
    
    def _analyze_triangle(self, triangle_points, pixel_to_cm_ratio):
        """分析三角形尺寸"""
        points = np.array([point[0] for point in triangle_points])
        
        # 计算边长
        side1 = self.get_dist(points[0], points[1]) * pixel_to_cm_ratio * 10  # 转换为毫米
        side2 = self.get_dist(points[1], points[2]) * pixel_to_cm_ratio * 10
        side3 = self.get_dist(points[2], points[0]) * pixel_to_cm_ratio * 10
        
        # 取平均边长作为尺寸
        avg_side = (side1 + side2 + side3) / 3.0
        
        return 'triangle', avg_side, points.tolist()
    
    def _analyze_square(self, square_points, pixel_to_cm_ratio):
        """分析正方形尺寸"""
        points = np.array([point[0] for point in square_points])
        
        # 计算四条边的长度
        sides = [
            self.get_dist(points[i], points[(i+1)%4]) * pixel_to_cm_ratio * 10  # 转换为毫米
            for i in range(4)
        ]
        
        # 检查是否为正方形（边长比例容差）
        max_side = max(sides)
        min_side = min(sides)
        if min_side == 0 or max_side / min_side > self.SQUARE_ASPECT_RATIO_TOLERANCE:
            return None
        
        # 检查角度是否接近90度
        for i in range(4):
            angle = self.get_angle(points[i-1], points[i], points[(i+1)%4])
            if not (90 - self.ANGLE_TOLERANCE < angle < 90 + self.ANGLE_TOLERANCE):
                return None
        
        # 取平均边长作为尺寸
        avg_side = sum(sides) / 4.0
        
        return 'square', avg_side, points.tolist()