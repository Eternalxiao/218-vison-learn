import cv2
from maix import image
import numpy as np

class MyCV2:
    def __init__(self, skernel: int=3, mkernel: int=3) -> None:

        # 初始化平滑处理和形态学处理的卷积核
        self.SmoothKernel = skernel
        self.MorphologyKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mkernel, mkernel))

    def image_process(self, img: image.Image, whetherCopy:bool=True, whetherTogray: bool=True):
        """
            对maix的图像进行opencv处理，参数说明：
            :param kernel_key: 卷积核大小
        """
        img_raw = image.image2cv(img, copy=whetherCopy)   # 转换为 OpenCV 格式
        
        if whetherTogray:
            # 转换为灰度 为了方便滤波去除噪点
            img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_raw 

        # 进行中值滤波去除椒盐噪声
        img_medianBlur = cv2.medianBlur(img_gray, self.SmoothKernel)

        # 高斯滤波保留边缘
        img_gaussian = cv2.GaussianBlur(img_medianBlur, (self.SmoothKernel, self.SmoothKernel), 0)

        # 闭运算 -- 白色底面，细小黑色为噪声
        img_close = cv2.morphologyEx(img_gaussian, cv2.MORPH_CLOSE, self.MorphologyKernel)

        img_return = img_close

        return image.cv2image(img_return, copy=True)

    def find_corners(self, CornersMachineImg: image.Image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE):
        """
            通过边缘检测进行角点识别,默认全部检测，只保留角点
            cv2.findContours(img,mode,method)  -- 返回轮廓列表和层次结构信信息
            # # mode:   RETR_EXTERNAL 只检测外围轮廓
                        RETR_TREE     按照树形存储轮廓，从大大小，从左到右
            # # method: CHAIN_APPROX_NONE       保存轮廓上所有的点
                        CHAIN_APPROX_SIMPLE     只保存角点
        """ 
        contours = None
        CornersMachineImg_cv = image.image2cv(CornersMachineImg, copy=True)   # 转换为 OpenCV 格式
        
        median = np.median(CornersMachineImg_cv)
        threshold1 = int(max(0, 0.95 * median))
        threshold2 = int(min(255, 1.35 * median))

        # 边缘检测
        img_edged = cv2.Canny(CornersMachineImg_cv, threshold1, threshold2)

        # # 效果不好可以再次可以选择性再次进行闭运算，加强效果 -- 一般不用
        # edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)  

        # 查找轮廓角点
        # contours, _ = cv2.findContours(img_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(img_edged, mode, method)

        return contours

    def approach_polygon(self, contour):
        """
            进行多边形逼近
        """

        # 获得近似值（与周长相关）：
        epsilon = 0.02 * cv2.arcLength(contour, True)

        """
        cv2.approxPolyDP(curve, epsilon, closed) -- 返回近似后的多边形点集
        # # curve: 输入的轮廓点集
        # # epsilon: 近似精度，值越小，近似越精确，一般与周长有关
        """
        approx = cv2.approxPolyDP(contour, epsilon, True)

        return approx

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
