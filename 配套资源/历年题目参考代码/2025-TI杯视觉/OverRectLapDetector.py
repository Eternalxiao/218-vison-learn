import cv2
import numpy as np
import math
import constants

class RectLapDetector:
    # --- 0. 全局常量和配置 (单线程/固定焦距/720x480 最终版) ---
    # ======================================================================
    

    # 1. 物理与几何参数
    KNOWN_A4_WIDTH_CM = 29.7  # A4纸横向宽度（物理参考值）
                              # 用于单目视觉测距的已知尺寸基准
                              # 测距公式：距离 = (物理宽度 × 焦距) / 像素宽度

    ANGLE_TOLERANCE = constants.ANGLE_TOLERANCE    # 角度容差（度）

    MIN_SIDE_CM = 5.5        # 正方形最小边长（厘米）
                             # 物理约束：过滤掉过小的噪声区域

    MAX_SIDE_CM = 12.0       # 正方形最大边长（厘米）
                             # 物理约束：过滤掉过大的非目标区域

    ROI_SHRINK_PIXELS = constants.ROI_SHRINK_PIXELS    # ROI区域收缩像素数

    APPROX_EPSILON_MULTIPLIER = 0.02  # 多边形逼近精度系数

    DEDUPLICATION_TOLERANCE = constants.DEDUPLICATION_TOLERANCE    # 去重容差（像素）

    SQUARE_ASPECT_RATIO_TOLERANCE = 1.35  # 正方形长宽比容差
                                          # 四边形四边长度比例限制：max_side/min_side < 1.25
                                          # 确保识别的是接近正方形的图形，而非长方形

    # 2. 图像处理与轮廓筛选参数
    BINARY_THRESHOLD = constants.BINARY_THRESHOLD   # 二值化阈值

    MORPH_ITERATIONS = constants.MORPH_ITERATIONS    # 形态学操作迭代次数

    PARENT_CONTOUR_MIN_AREA = constants.REFERENCE_DISTANCE - 200  # A4纸轮廓最小面积阈值（像素²）

    CHILD_CONTOUR_MIN_AREA = constants.CHILD_CONTOUR_MIN_AREA  # 子轮廓（正方形）最小面积阈值（像素²）

    # 3. 核心校准参数 - 统一使用面积标定
    SYSTEM_OFFSET_CM = constants.SYSTEM_OFFSET_CM

    # ======================================================================

    def __init__(self):
        pass

    # --- 辅助函数 ---
    @staticmethod
    def get_angle(p1, p2, p3):
        """向量叉积和点积计算三点之间的夹角，用于验证是否为直角（90°±容差）"""
        v1 = p1 - p2;
        v2 = p3 - p2
        dot = v1[0] * v2[0] + v1[1] * v2[1];
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


    def detect_overlap_squares(self, frame):
        """
        检测重叠的正方形并返回最小边长(mm)和测量长度(cm)
        :param frame: 输入的BGR图像 (OpenCV格式)
        :return: 最小边长(mm), 检测到的正方形列表, 距离(cm)
        """
        # 图像预处理
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_img, self.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        # 二值化后进行形态学操作 先膨胀后腐蚀（闭运算），消除噪声和填补空洞
        binary = cv2.dilate(binary, None, iterations=self.MORPH_ITERATIONS)
        binary = cv2.erode(binary, None, iterations=self.MORPH_ITERATIONS)
        
        # 对图像进行轮廓层次对分析
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
        candidate_pairs = []        # 存储潜在的父子轮廓对
        min_side_mm = 0
        detected_squares = []
        
        if hierarchy is not None:
            for i, h_info in enumerate(hierarchy[0]):
                if h_info[2] != -1: # 有孩子的是潜在的a4纸轮廓
                    p_cnt = contours[i] # # 获取当前父轮廓的点集
                    p_peri = cv2.arcLength(p_cnt, True)  # 计算父轮廓的周长（用于后续多边形逼近 True表示轮廓是闭合的）
                    p_approx = cv2.approxPolyDP(p_cnt, 0.02 * p_peri, True)  # 使用Douglas-Peucker算法逼近多边形，0.02*周长作为逼近精度阈值
                    if len(p_approx) == 4 and cv2.contourArea(p_cnt) > self.PARENT_CONTOUR_MIN_AREA:
                        candidate_pairs.append((contours[i], contours[h_info[2]]))

        if candidate_pairs: 
            # 选择面积最大的A4纸轮廓
            selected_parent_contour, selected_child_contour = max(candidate_pairs, key=lambda p: cv2.contourArea(p[0]))
            
            # 使用面积标定计算距离
            current_area = cv2.contourArea(selected_parent_contour)
            print(f"当前A4纸面积: {current_area} px², 参考面积: {constants.REFERENCE_AREA} px²")
            
            if current_area > 0:
                # 计算距离 - 统一使用面积标定
                final_distance_cm = self.calculate_distance(current_area)
                
                # 计算像素到实际尺寸的转换比例
                # 距离比例：实际距离/参考距离
                distance_ratio = final_distance_cm / constants.REFERENCE_DISTANCE
                # 面积比例
                area_ratio = current_area / constants.REFERENCE_AREA
                # 像素到厘米的转换比例 = (A4实际宽度cm / 参考距离cm) * 距离比例 / sqrt(面积比例)
                pixel_to_cm_ratio = (constants.A4_REAL_WIDTH_MM / 10 / constants.REFERENCE_DISTANCE) * distance_ratio / np.sqrt(area_ratio)

                
                # 获取子轮廓的最小正外接矩形 -- 内环（非旋转矩形）
                x_orig, y_orig, w_orig, h_orig = cv2.boundingRect(selected_child_contour)
                # 计算收缩后的ROI区域
                x_shrunk, y_shrunk = x_orig + self.ROI_SHRINK_PIXELS, y_orig + self.ROI_SHRINK_PIXELS
                w_shrunk, h_shrunk = w_orig - 2 * self.ROI_SHRINK_PIXELS, h_orig - 2 * self.ROI_SHRINK_PIXELS

                if w_shrunk > 0 and h_shrunk > 0:
                    # 提取收缩后的ROI区域
                    roi_gray_shrunk = gray_img[y_shrunk: y_shrunk + h_shrunk, x_shrunk: x_shrunk + w_shrunk]
                    # 对缩小的ROI区域进行算法过滤二值化
                    _, roi_binary_shrunk = cv2.threshold(roi_gray_shrunk, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    # 寻找内部轮廓
                    contours_in_roi, _ = cv2.findContours(roi_binary_shrunk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # 初始化假设列表 -- 三个策略
                    hypotheses_direct, hypotheses_A, hypotheses_B = [], [], []

                    for contour in contours_in_roi:
                        # 过滤掉小于最小阈值的轮廓
                        if cv2.contourArea(contour) < self.CHILD_CONTOUR_MIN_AREA: continue
                        # 计算轮廓周长
                        peri = cv2.arcLength(contour, True)
                        # 多边形逼近，减少顶点数
                        vertices = cv2.approxPolyDP(contour, self.APPROX_EPSILON_MULTIPLIER * peri, True)
                        # 获取角点数量
                        num_vertices = len(vertices)

                        # 情况1: 直接识别正方形
                        if num_vertices == 4:
                            points = np.array([v[0] for v in vertices])
                            # 计算四条边的长度
                            sides = [self.get_dist(points[i], points[(i + 1) % 4]) for i in range(4)]
                            # 有边长并且满足长宽比容差
                            if sides and min(sides) > 0 and max(sides) / min(sides) < self.SQUARE_ASPECT_RATIO_TOLERANCE:
                                # 验证四个角都是直角
                                angles_ok = True
                                for i in range(4):
                                    angle = self.get_angle(points[i - 1], points[i], points[(i + 1) % 4])
                                    if not (90 - self.ANGLE_TOLERANCE < angle < 90 + self.ANGLE_TOLERANCE):
                                        angles_ok = False
                                        break
                                
                                if angles_ok:
                                    hypotheses_direct.append({
                                        'side': np.mean(sides), 
                                        'corners': points, 
                                        'type': 'Direct-4'
                                    })
                                    continue

                        # 情况2: 识别直角三角形或直角梯形
                        if num_vertices > 4:
                            # 创建角点列表，标记是否为直角
                            corner_list = [
                                {
                                    'point': np.array(vertices[i][0]), 
                                    # 判断是否直角（90°±容差）
                                    'is_right': (90 - self.ANGLE_TOLERANCE < 
                                               self.get_angle(vertices[i - 1][0], vertices[i][0], vertices[(i + 1) % num_vertices][0]) < 
                                               90 + self.ANGLE_TOLERANCE),
                                    'processed': False  # 标记该点是否已被使用
                                }
                                for i in range(num_vertices)
                            ]

                            # 第一轮处理：寻找连续直角点组合（规则A）
                            for i in range(num_vertices):
                                c_i = corner_list[i]  # 当前顶点
                                
                                # 跳过非直角或已处理的点
                                if not c_i['is_right'] or c_i['processed']: 
                                    continue
                                    
                                c_next = corner_list[(i + 1) % num_vertices]
                                c_prev = corner_list[(i - 1 + num_vertices) % num_vertices]
                                
                                # === 情况A1：左-中-右三个连续直角（L型推导）===
                                if (c_prev['is_right'] and not c_prev['processed'] and 
                                    c_next['is_right'] and not c_next['processed']):
                                    
                                    # 几何推导第四个点：p4 = p_prev + (p_next - p_i)
                                    p_prev, p_i, p_next = c_prev['point'], c_i['point'], c_next['point']
                                    p4 = p_prev + (p_next - p_i)
                                    
                                    # 记录假设（L3模式：三个连续直角点推导）
                                    hypotheses_A.append({
                                        'side': self.get_dist(p_i, p_next),  # 使用相邻点距离作为边长
                                        'corners': [p_prev, p_i, p_next, p4],
                                        'type': 'RuleA-L3'  # L型三点推导
                                    })
                                    # 标记这三个点已处理
                                    c_prev['processed'] = c_i['processed'] = c_next['processed'] = True
                                    continue

                                # === 情况A2：处理中间点开始的连续直角 ===
                                if c_next['is_right'] and not c_next['processed']:
                                    c_next_next = corner_list[(i + 2) % num_vertices]

                                    # 子情况A2a：中间-右1-右2三个连续直角（I型直线推导）
                                    if (c_next_next['is_right'] and 
                                        not c_next_next['processed']):

                                        # 向量运算：p4 = p_i + (p_next_next - p_next)
                                        p_i, p_next, p_next_next = c_i['point'], c_next['point'], c_next_next['point']
                                        p4 = p_i + (p_next_next - p_next)

                                        hypotheses_A.append({
                                            'side': self.get_dist(p_next, p_next_next),
                                            'corners': [p_i, p_next, p_next_next, p4],
                                            'type': 'RuleA-I3'  # 直线三点推导
                                        })
                                        # 标记已处理
                                        c_i['processed'] = c_next['processed'] = c_next_next['processed'] = True
                                    else:
                                        # 子情况A2b：中间-右1两个连续直角
                                        p_prev_for_geom, p_i, p_next = c_prev['point'], c_i['point'], c_next['point']
                                        p4 = p_prev_for_geom + (p_next - p_i)
                                        hypotheses_A.append({
                                            'side': self.get_dist(p_i, p_next),
                                            'corners': [p_prev_for_geom, p_i, p_next, p4],
                                            'type': 'RuleA-I2'  # 两点推导
                                        })
                                        c_i['processed'] = c_next['processed'] = True
                                    continue  # 跳过后续处理

                            # 第二轮处理：孤立直角点组合（规则B）
                            # 收集所有未处理的孤立直角点
                            isolated_candidates = [c for c in corner_list if c['is_right'] and not c['processed']]
                            
                            # 需要至少2个孤立点用于几何推导
                            if len(isolated_candidates) >= 2:
                                # 使用统一的像素到厘米转换比例计算对角线范围
                                min_diag_px = (self.MIN_SIDE_CM * math.sqrt(2)) / pixel_to_cm_ratio
                                max_diag_px = (self.MAX_SIDE_CM * math.sqrt(2)) / pixel_to_cm_ratio
                                
                                # 遍历所有可能的点对组合
                                for i in range(len(isolated_candidates)):
                                    for j in range(i + 1, len(isolated_candidates)):
                                        c1, c2 = isolated_candidates[i], isolated_candidates[j]
                                        if c1['processed'] or c2['processed']: 
                                            continue

                                        # 获取两点坐标
                                        p1, p3 = c1['point'], c2['point']
                                        diag_dist = self.get_dist(p1, p3)
                                        # 对角线长度在允许范围内
                                        if min_diag_px < diag_dist < max_diag_px:
                                            # 1. 边长 = 对角线/sqrt(2)
                                            pixel_side = diag_dist / math.sqrt(2)

                                            # 2. 计算中心点
                                            center = (p1 + p3) / 2.0
                                            
                                            # 3. 生成正交向量（旋转原对角线向量90度）
                                            diag_vec = p1 - center
                                            p2_vec = np.array([-diag_vec[1], diag_vec[0]])

                                            # 4. 计算缺失的两个角点
                                            p2, p4 = center + p2_vec, center - p2_vec
                                            
                                            # 记录假设（规则B推导）
                                            hypotheses_B.append({
                                                'side': pixel_side,
                                                'corners': [p1, p2, p3, p4],  # 完整四个角点
                                                'type': 'RuleB'  # 几何推导标记
                                            })
                                            # 标记这两个点已处理 
                                            c1['processed'] = c2['processed'] = True

                    # 合并三种检测方式的结果（直接检测/规则A/规则B）
                    all_hypotheses = hypotheses_direct + hypotheses_A + hypotheses_B
                    unique_squares = []

                    if all_hypotheses:
                        """
                        去重逻辑：
                        计算每个假设的中心点坐标（四个角点的均值）
                        若新假设中心点与已存假设中心点距离 < DEDUPLICATION_TOLERANCE（13像素），则视为重复
                        保留最大边长的假设（因已排序）
                        """
                        # 按边长降序排序
                        all_hypotheses.sort(key=lambda s: s['side'], reverse=True)
                        for sq in all_hypotheses:
                            # 计算当前正方形的中心点
                            center = np.mean(sq['corners'], axis=0)
                            
                            # 检查是否与已存储的正方形重复
                            is_duplicate = False
                            for usq in unique_squares:
                                usq_center = np.mean(usq['corners'], axis=0)
                                if self.get_dist(center, usq_center) < self.DEDUPLICATION_TOLERANCE:
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                unique_squares.append(sq)

                    if unique_squares:
                        # 将ROI坐标转换回原图坐标系
                        for sq in unique_squares:
                            sq['corners'] = [point + [x_shrunk, y_shrunk] for point in sq['corners']]
                        
                        # 转换单位为毫米 - 使用统一的像素到厘米转换比例
                        valid_squares = []
                        for sq in unique_squares:
                            side_cm = sq['side'] * pixel_to_cm_ratio
                            side_mm = side_cm * 10  # 转换为毫米
                            
                            if self.MIN_SIDE_CM < side_cm < self.MAX_SIDE_CM:
                                sq['side_mm'] = side_mm
                                valid_squares.append(sq)
                        
                        if valid_squares:
                            # 找出最小正方形
                            min_square = min(valid_squares, key=lambda x: x['side_mm'])
                            min_side_mm = min_square['side_mm']
                            detected_squares = valid_squares    # 保存所有有效正方形 格式为 [{'side_mm': 10.0, 'corners': [[x1, y1], [x2, y2], ...], 'type': 'Direct-4'}, ...]

                            print(f"检测到正方形，最小边长: {min_side_mm:.2f} mm, 检测到 {len(detected_squares)} 个正方形")

                            

                return min_side_mm, detected_squares, final_distance_cm
            
        return 0, [], 0