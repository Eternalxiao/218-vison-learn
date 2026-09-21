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
            _, img_adapt_binary = cv2.threshold(gray_cv_img, threshold, self.max_value, 
                                     cv2.THRESH_BINARY)
        
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



if __name__ == '__main__':
    from maix import camera, image, display
    import time


    # 初始化摄像头
    cam = camera.Camera(320, 240)

    # 初始化显示
    disp = display.Display()

    # 初始化类
    cv = MyCV2()

    # 定义多种颜色的HSV阈值
    # 每个颜色独立阈值
    thresholds = [
        # 蓝色
        [100, 124, 43, 255, 46, 255],
        # 绿色
        [35, 77, 43, 255, 46, 255],
        # 红色1
        [0, 10, 43, 255, 46, 255],
        # 红色2 (独立处理，不自动合并)
        [156, 180, 43, 255, 46, 255],
        # 黄色
        [20, 34, 100, 255, 100, 255],
        # 紫色
        [125, 155, 50, 255, 50, 255]
    ]

    # 颜色名称映射
    color_names = {
        0: "蓝色",
        1: "绿色",
        2: "红色1",
        3: "红色2",
        4: "黄色",
        5: "紫色"
    }

    # 颜色绘制颜色
    draw_colors = {
        0: (255, 0, 0),    # 蓝色 - RGB
        1: (0, 255, 0),    # 绿色
        2: (255, 0, 0),    # 红色1
        3: (255, 0, 0),    # 红色2
        4: (255, 255, 0),  # 黄色
        5: (255, 0, 255)   # 紫色
    }

    while True:
        # 获取图像
        img = cam.read()
        
        # 定义ROI区域 [x, y, w, h]
        roi = [50, 30, 220, 180]  # 只检测图像中央区域
        
        # 查找所有色块
        blobs = cv.find_blobs(
            img=img,
            thresholds=thresholds,
            roi=roi,
            x_stride=2,
            y_stride=2,
            area_threshold=50
        )
        
        # 处理检测结果
        for blob in blobs:
            # 获取颜色信息
            color_idx = blob.color
            color_name = color_names.get(color_idx, f"颜色{color_idx}")
            draw_color = draw_colors.get(color_idx, (255, 255, 255))  # 默认白色
            
            # 在图像上绘制边界框
            img.draw_rect(
                int(blob.x), int(blob.y), int(blob.w), int(blob.h), 
                color=image.COLOR_BLUE,
                thickness=2
            )
            
            # 绘制中心点
            img.draw_circle(
                int(blob.cx()), int(blob.cy()), 3, 
                color=image.COLOR_BLUE, thickness=-1
            )
            
        
        # 显示图像
        disp.show(img)
        
        # 控制帧率
        time.sleep(0.05)