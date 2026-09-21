import cv2
from maix import image
from maix._maix.image import cv2image, image2cv
import numpy as np
from typing import List, Optional

class MyBlob:
    def __init__(self, x: int, y: int, width: int, height: int, pixels: int, color: int):
        """
        色块对象（简化版，专注于位置和颜色信息）
        :param x: 左上角X坐标（全局坐标）
        :param y: 左上角Y坐标（全局坐标）
        :param width: 宽度
        :param height: 高度
        :param pixels: 像素数
        :param color: 颜色索引（原始阈值列表中的索引）
        """
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        self._pixels = pixels
        self.color = color  # 直接使用颜色索引
        self._cx = x + width // 2  # 中心X
        self._cy = y + height // 2  # 中心Y
    
    def rect(self) -> tuple:
        """返回矩形区域(x, y, w, h)"""
        return (self.x, self.y, self.w, self.h)
        
    def pixels(self) -> int:
        """
        获取色块的像素数
        
        返回:
            色块包含的像素数量
        """
        return self._pixels

    def cx(self) -> int:
        """
        获取色块中心的X坐标
        
        返回:
            色块中心的X坐标
        """
        return self._cx
    
    def cy(self) -> int:
        """
        获取色块中心的Y坐标
        
        返回:
            色块中心的Y坐标
        """
        return self._cy
    def __repr__(self) -> str:
        return f"Blob(x={self.x}, y={self.y}, w={self.w}, h={self.h}, pixels={self.pixels}, color={self.color})"
      

class MyGetPixel:
    def __init__(self) -> None:
        pass

    def rgb_to_lab(self, rgb):
        '''
        实现RGB值到LAB值的转换
        '''

        # RGB到XYZ的转换矩阵
        M = [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ]
        
        # 归一化RGB值
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        
        # 线性化RGB值
        r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
        
        # 计算XYZ值
        X = M[0][0] * r + M[0][1] * g + M[0][2] * b
        Y = M[1][0] * r + M[1][1] * g + M[1][2] * b
        Z = M[2][0] * r + M[2][1] * g + M[2][2] * b
        
        # XYZ到LAB的转换
        X /= 0.95047
        Y /= 1.0
        Z /= 1.08883
        
        def f(t):
            return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116
        
        L = 116 * f(Y) - 16
        a = 500 * (f(X) - f(Y))
        b = 200 * (f(Y) - f(Z))
        
        return [L, a, b]

    # 可靠的RGB转HSV函数
    def rgb_to_hsv(self, rgb, tolerance_h=10, tolerance_s=40, tolerance_v=40) -> list:
        """
        将RGB颜色转换为HSV阈值列表
        输入: 
            rgb: [R, G, B] 每个值范围0-255
            tolerance_h: 色相容差
            tolerance_s: 饱和度容差
            tolerance_v: 明度容差
        输出: 
            HSV阈值列表 [[H_min, H_max, S_min, S_max, V_min, V_max]]
        """
        # 将RGB转换为0-1范围
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        
        # 计算最大值和最小值
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        delta = max_val - min_val
        
        # 计算色调(H)
        if delta == 0:
            h = 0
        elif max_val == r:
            h = 60 * (((g - b) / delta) % 6)
        elif max_val == g:
            h = 60 * (((b - r) / delta) + 2)
        else:  # max_val == b
            h = 60 * (((r - g) / delta) + 4)
        
        # 确保H在0-360范围内
        h = h % 360
        if h < 0:
            h += 360
        
        # 计算饱和度和明度
        s = (delta / max_val) * 100 if max_val != 0 else 0
        v = max_val * 100
        
        # 转换为OpenCV HSV范围
        h_cv = h / 2          # 0-180
        s_cv = s * 2.55       # 0-255
        v_cv = v * 2.55       # 0-255
        
        # 创建基础阈值
        h_min = max(0, int(h_cv - tolerance_h))
        h_max = min(180, int(h_cv + tolerance_h))
        s_min = max(0, int(s_cv - tolerance_s))
        s_max = min(255, int(s_cv + tolerance_s))
        v_min = max(0, int(v_cv - tolerance_v))
        v_max = min(255, int(v_cv + tolerance_v))
        
        base_threshold = [h_min, h_max, s_min, s_max, v_min, v_max]
        
        # 如果是红色，添加双阈值
        if h_min <= 10 or h_max >= 170 :
            # 红色低区阈值 (0-10)
            red_low = [0, 10, s_min, s_max, v_min, v_max]
            # 红色高区阈值 (170-180)
            red_high = [170, 180, s_min, s_max, v_min, v_max]
            return [red_high, red_low]
        
        return [base_threshold]
  
    def filter_similar_thresholds(self, thresholds, tolerance=3):
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

    def set_configured_threshold(self, threshold):
        '''
        阈值参数信息存入配置文件
        '''
        if len(threshold) < 12:
            return 

        app.set_app_config_kv('demo_find_line', 'lmin', str(threshold[0]), False)
        app.set_app_config_kv('demo_find_line', 'lmax', str(threshold[1]), False)
        app.set_app_config_kv('demo_find_line', 'amin', str(threshold[2]), False)
        app.set_app_config_kv('demo_find_line', 'amax', str(threshold[3]), False)
        app.set_app_config_kv('demo_find_line', 'bmin', str(threshold[4]), False)
        app.set_app_config_kv('demo_find_line', 'bmax', str(threshold[5]), False)

    def get_configured_threshold(self):
        '''
            获取所存储配置文件中的阈值参数
        '''
        thresholds = [
            [0, 10, 43, 255, 46, 255],     #红色阈值1   
            [156, 180, 43, 255, 46, 255]  #红色阈值2
        ]
        # thresholds = [0, 10, 43, 255, 46, 255],     #红色阈值1   
        

        # value_str = app.get_app_config_kv('demo_find_line', 'lmin','', False)
        # if len(value_str) > 0:
        #     threshold[0] = int(value_str)
        # value_str = app.get_app_config_kv('demo_find_line', 'lmax','', True)
        # if len(value_str) > 0:
        #     threshold[1] = int(value_str)
        # value_str = app.get_app_config_kv('demo_find_line', 'amin','', False)
        # if len(value_str) > 0:
        #     threshold[2] = int(value_str)
        # value_str = app.get_app_config_kv('demo_find_line', 'amax','', False)
        # if len(value_str) > 0:
        #     threshold[3] = int(value_str)
        # value_str = app.get_app_config_kv('demo_find_line', 'bmin','', False)
        # if len(value_str) > 0:
        #     threshold[4] = int(value_str)
        # value_str = app.get_app_config_kv('demo_find_line', 'bmax','', False)
        # if len(value_str) > 0:
        #     threshold[5] = int(value_str)
        return thresholds


class MyCV2:
    def __init__(self, skernel: int=3, mkernel: int=11, method='entropy', block_size=15, c=5, max_value=255) -> None:

        # 初始化平滑处理和形态学处理的卷积核
        self.SmoothKernel = skernel
        self.MorphologyKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mkernel, mkernel))

        """
        自适应阈值处理器
        :param method: 阈值方法 ('otsu', 'adaptive', 'entropy')
        :param block_size: 自适应阈值块大小(奇数)   # 越小计算量越小
        :param c: 从均值/中值减去的常数
        :param max_value: 二值化最大值
        """
        self.method = method
        self.block_size = block_size
        self.c = c
        self.max_value = max_value
        
        # 历史阈值记录
        self.threshold_history = []
        self.history_size = 10

    def image_process(self, img):
        """
            对maix的图像进行opencv处理，参数说明：
            :param kernel_key: 卷积核大小
        """
        img_raw = image.image2cv(img, copy=False)   # 转换为 OpenCV 格式
        
        # 转换为灰度 为了方便滤波去除噪点
        # img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

        # 进行均值滤波
        img_blur = cv2.blur(img_raw, (self.SmoothKernel, self.SmoothKernel))
        
        # 闭运算 -- 白色底面，细小黑色为噪声
        img_close = cv2.morphologyEx(img_blur, cv2.MORPH_CLOSE, self.MorphologyKernel)

        img_return = img_close

        return image.cv2image(img_return , copy=True)

    def frame_dif_method(self, now_img: image.Image, last_img: image.Image):
        """
            帧差法实现动态目标的查找，返回轮廓的 x, y, w, h
        """
        
        img_now_cv = image.image2cv(now_img, copy=True)
        img_last_cv = image.image2cv(last_img, copy=True)

        img_now_cv = cv2.cvtColor(img_now_cv, cv2.COLOR_BGR2GRAY)
        img_last_cv = cv2.cvtColor(img_last_cv, cv2.COLOR_BGR2GRAY)

        # # 计算差值
        img_diff = cv2.absdiff(img_now_cv, img_last_cv)

        # 二值化处理
        _, img_dif_binary = cv2.threshold(img_diff, 25, 255, cv2.THRESH_BINARY)
        
        # 膨胀处理填充激光点空洞
        img_dilate_binary = cv2.dilate(img_dif_binary, self.MorphologyKernel, iterations=3)

        # return image.cv2image(img_dilate_binary, copy=True)
        
        # 查找轮廓
        contours, _ = cv2.findContours(img_dilate_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 初始化激光点区域记录列表
        all_region = list()

        for contour in contours:
            # 计算轮廓面积
            contour_area = cv2.contourArea(contour)

            # 过滤掉大范围的轮廓
            if contour_area < 350: 
                # 获取轮廓的外接矩形 
                x, y, w, h = cv2.boundingRect(contour)
                all_region.append([x,y,w,h])    # 记录roi区域
        
        return all_region

    # 使用 hsv 红绿区分
    def detect_red_green_regions(self, detect_RG_img:image.Image, region1, region2):
        """
            检测两个区域中哪个是红色区域，哪个是绿色区域
            
            参数:
            image - 输入图像(BGR格式)
            region1 - 第一个区域(x, y, w, h)
            region2 - 第二个区域(x, y, w, h)
            
            返回:
            (red_region, green_region) - 红色区域和绿色区域的坐标
        """
        
        detect_RG_img_cv = image.image2cv(detect_RG_img, copy=True)

        # 转换为HSV颜色空间（更容易检测颜色）
        hsv = cv2.cvtColor(detect_RG_img_cv, cv2.COLOR_BGR2HSV)
        
        # 提取两个ROI（感兴趣区域）
        x1, y1, w1, h1 = region1
        roi1 = hsv[y1:y1+h1, x1:x1+w1]
        
        x2, y2, w2, h2 = region2
        roi2 = hsv[y2:y2+h2, x2:x2+w2]
        
        # 定义红色和绿色的HSV范围
        # 红色在HSV中有两个范围（0°和180°附近）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # 绿色范围
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        # 计算每个区域中的红色像素比例
        red_mask1 = cv2.inRange(roi1, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(roi1, lower_red2, upper_red2)
        red_mask_roi1 = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels1 = np.sum(red_mask_roi1 > 0)
        total_pixels1 = roi1.shape[0] * roi1.shape[1]
        red_ratio1 = red_pixels1 / total_pixels1 if total_pixels1 > 0 else 0
        
        red_mask1 = cv2.inRange(roi2, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(roi2, lower_red2, upper_red2)
        red_mask_roi2 = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels2 = np.sum(red_mask_roi2 > 0)
        total_pixels2 = roi2.shape[0] * roi2.shape[1]
        red_ratio2 = red_pixels2 / total_pixels2 if total_pixels2 > 0 else 0
        
        # 计算每个区域中的绿色像素比例
        green_mask_roi1 = cv2.inRange(roi1, lower_green, upper_green)
        green_pixels1 = np.sum(green_mask_roi1 > 0)
        green_ratio1 = green_pixels1 / total_pixels1 if total_pixels1 > 0 else 0
        
        green_mask_roi2 = cv2.inRange(roi2, lower_green, upper_green)
        green_pixels2 = np.sum(green_mask_roi2 > 0)
        green_ratio2 = green_pixels2 / total_pixels2 if total_pixels2 > 0 else 0
        
        # 判断哪个区域是红色，哪个是绿色
        if red_ratio1 > green_ratio1 and green_ratio2 > red_ratio2:
            return region1, region2  # region1是红色，region2是绿色
        elif red_ratio2 > green_ratio2 and green_ratio1 > red_ratio1:
            return region2, region1  # region2是红色，region1是绿色
        else:
            # 如果无法明确区分，返回比例更高的那个
            if red_ratio1 + green_ratio2 > red_ratio2 + green_ratio1:
                return region1, region2
            else:
                return region2, region1

    def apply(self, gray_img):
        """
        应用自适应阈值
        :param gray_img: 输入灰度图像
        :return: 二值化图像
        """
        img_adapt_binary = None
        gray_cv_img = image2cv(gray_img,copy=True)

        if self.method == 'otsu':   # 光照稳定场景 -- 使用Otsu方法，计算量小
            # Otsu全局阈值
            _, img_adapt_binary = cv2.threshold(gray_cv_img, 0, self.max_value, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_adapt_binary = cv2.morphologyEx(img_adapt_binary, cv2.MORPH_OPEN, self.MorphologyKernel)
        elif self.method == 'adaptive': # 光照变化场景 -- 使用自适应阈值，处理光照不均
            # 自适应局部阈值
            img_adapt_binary = cv2.adaptiveThreshold(gray_cv_img, self.max_value,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 
                                          self.block_size, self.c)
            # img_adapt_binary = cv2.morphologyEx(img_adapt_binary, cv2.MORPH_OPEN, self.MorphologyKernel)
        
        elif self.method == 'entropy':  # 复杂背景场景 -- 使用熵阈值法，抗噪声干扰
            # 熵阈值法
            threshold = self._calculate_entropy_threshold(gray_cv_img)
            _, img_adapt_binary = cv2.threshold(gray_cv_img, threshold, self.max_value, cv2.THRESH_BINARY)
        
        # 更新阈值历史 -- 自适应阈值需要
        self._update_threshold_history(img_adapt_binary)
        
        img_return = cv2image(img_adapt_binary, copy=True)
        return img_return
    
    def _calculate_entropy_threshold(self, gray_img):
        """计算基于信息熵的阈值"""
        # 计算直方图
        hist = cv2.calcHist([gray_img], [0], None, [256], [0,256])
        hist = hist.ravel()/hist.sum()  # 归一化
        
        # 计算累积熵
        entropy = np.zeros(256)
        for t in range(1, 255):
            # 前景和背景概率
            p_back = hist[:t].sum()
            p_fore = hist[t:].sum()
            
            # 前景和背景熵
            entropy_back = -np.sum(hist[:t]/p_back * np.log(hist[:t]/p_back + 1e-10))
            entropy_fore = -np.sum(hist[t:]/p_fore * np.log(hist[t:]/p_fore + 1e-10))
            
            entropy[t] = entropy_back + entropy_fore
        
        return np.argmax(entropy)
    
    def _update_threshold_history(self, binary_img):
        """更新阈值历史并自动调整参数"""
        # 计算当前阈值效果
        line_area_ratio = np.sum(binary_img == 255) / binary_img.size
        
        # 理想线区域占比 (根据赛道经验值)
        ideal_ratio = 0.3
        
        # 记录历史
        self.threshold_history.append(line_area_ratio)
        if len(self.threshold_history) > self.history_size:
            self.threshold_history.pop(0)
        
        # 每10帧调整一次参数
        if len(self.threshold_history) == self.history_size:
            avg_ratio = np.mean(self.threshold_history)
            
            # 调整自适应阈值参数
            if avg_ratio < ideal_ratio * 0.7:  # 线区域太小
                self.c -= 1  # 降低阈值，保留更多区域
                print(f"自适应调整: 线区域过小({avg_ratio:.2f}), 降低阈值 c={self.c}")
            elif avg_ratio > ideal_ratio * 1.3:  # 线区域太大
                self.c += 1  # 提高阈值，减少噪声
                print(f"自适应调整: 线区域过大({avg_ratio:.2f}), 提高阈值 c={self.c}")
            
            # 重置历史
            self.threshold_history = []

    def find_hsv_blobs(self,
    img: image.Image,  # MaixPy图像对象
    thresholds: List[List[int]],
    roi: Optional[List[int]] = None,
    x_stride: int = 2,
    y_stride: int = 1,
    area_threshold: int = 10,
    pixels_threshold: int = 10,
    **kwargs
    ) -> List[MyBlob]:
        """
        在HSV色彩空间中查找所有满足阈值的色块
        
        参数:
            img: MaixPy图像对象
            thresholds: 阈值列表 [[H_min, H_max, S_min, S_max, V_min, V_max], ...]
            roi: 感兴趣区域 [x, y, w, h] (全局坐标)
            x_stride: X方向扫描步长
            y_stride: Y方向扫描步长
            area_threshold: 色块最小面积
            pixels_threshold: 色块最小像素数
        
        返回:
            Blob对象列表，包含全局坐标和颜色信息
        """
        # 将MaixPy图像转换为OpenCV格式 (RGB格式)
        img_cv = image.image2cv(img, ensure_bgr=False, copy=True)
        
        # 使用pixels_threshold和area_threshold中较大的值
        actual_threshold = max(area_threshold, pixels_threshold)
        
        # 获取图像尺寸
        img_h, img_w = img_cv.shape[:2]
        
        # 处理ROI区域
        roi_x, roi_y, roi_w, roi_h = 0, 0, img_w, img_h
        if roi is not None:
            roi_x, roi_y, roi_w, roi_h = roi
            # 确保ROI在图像范围内
            roi_x = max(0, min(roi_x, img_w - 1))
            roi_y = max(0, min(roi_y, img_h - 1))
            roi_w = max(1, min(roi_w, img_w - roi_x))
            roi_h = max(1, min(roi_h, img_h - roi_y))
            
            # 提取ROI区域
            img_roi = img_cv[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        else:
            img_roi = img_cv
            roi_x, roi_y, roi_w, roi_h = 0, 0, img_w, img_h
        
        # 转换为HSV
        hsv = cv2.cvtColor(img_roi, cv2.COLOR_RGB2HSV)

        # 对亮度通道进行直方图均衡化，补偿全局光照不均，提升光照鲁棒性
        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)  # 仅均衡化亮度通道
        hsv = cv2.merge([h, s, v_eq])
        
        # 创建合并掩膜
        merged_mask = np.zeros((img_roi.shape[0], img_roi.shape[1]), dtype=np.uint8)
        
        # 应用所有阈值
        for th in thresholds:
            h_min, h_max, s_min, s_max, v_min, v_max = th
            
            # 创建掩膜
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)
            
            # 合并到总掩膜
            merged_mask = cv2.bitwise_or(merged_mask, mask)
        
        # 形态学操作 - 消除噪声和连接区域
        kernel = np.ones((3, 3), np.uint8)
        merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, kernel)
        merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel)
        
        all_blobs = []
        
        # 处理步长参数
        if x_stride > 1 or y_stride > 1:
            # 创建缩小版的掩膜
            small_h = (merged_mask.shape[0] + y_stride - 1) // y_stride
            small_w = (merged_mask.shape[1] + x_stride - 1) // x_stride
            
            # 处理可能的0尺寸情况
            if small_h <= 0 or small_w <= 0:
                return all_blobs
            
            small_mask = cv2.resize(merged_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            
            # 在缩小掩膜上查找轮廓
            contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 缩放轮廓到原始尺寸
            for cnt in contours:
                # 将轮廓点转换为浮点数以便缩放
                cnt = cnt.astype(np.float32)
                cnt[:, :, 0] *= x_stride
                cnt[:, :, 1] *= y_stride
                cnt = cnt.astype(np.int32)
                
                # 计算面积 - 需要乘以步长面积因子
                area = cv2.contourArea(cnt) * (x_stride * y_stride)
                if area < actual_threshold:
                    continue
                
                # 获取边界框
                x_local, y_local, w_local, h_local = cv2.boundingRect(cnt)
                
                # 转换为全局坐标
                x_global = roi_x + x_local
                y_global = roi_y + y_local
                
                # 创建Blob对象
                blob = MyBlob(
                    x=x_global, 
                    y=y_global,
                    width=w_local,
                    height=h_local,
                    pixels=int(area),
                    color=0  # 使用单一颜色索引
                )
                all_blobs.append(blob)
        else:
            # 形态学操作
            kernel = np.ones((3, 3), np.uint8)
            merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, kernel)
            merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel)
            
            # 不使用步长 - 在原始掩膜上查找轮廓
            contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 处理轮廓
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < actual_threshold:
                    continue
                    
                # 获取边界框
                x_local, y_local, w_local, h_local = cv2.boundingRect(cnt)
                
                # 转换为全局坐标
                x_global = roi_x + x_local
                y_global = roi_y + y_local
                
                # 创建Blob对象
                blob = MyBlob(
                    x=x_global, 
                    y=y_global,
                    width=w_local,
                    height=h_local,
                    pixels=int(area),
                    color=0  # 使用单一颜色索引
                )
                all_blobs.append(blob)
        
        return all_blobs
    
  