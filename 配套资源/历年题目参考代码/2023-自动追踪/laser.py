from maix import image, camera, display, app
import cv2, random, time
import numpy as np

def center_scale_rect(x, y, w, h, scale=0.5):
    """
    对矩形进行中心缩放，返回缩放后的位置和尺寸
    
    参数:
    x, y - 矩形左上角坐标
    w, h - 矩形的宽度和高度
    scale - 缩放比例 (1.0表示不变，0.5表示缩小一半，2.0表示放大一倍)
    
    返回:
    (new_x, new_y, new_w, new_h) - 缩放后矩形的左上角坐标和尺寸
    """
    # 计算矩形的中心点
    center_x = x + w / 2
    center_y = y + h / 2
    
    # 计算缩放后的尺寸
    new_w = w * scale
    new_h = h * scale
    
    # 计算缩放后的左上角坐标（基于中心点不变）
    new_x = center_x - new_w / 2
    new_y = center_y - new_h / 2
    
    # 返回缩放后的矩形参数（坐标转为整数）
    return (int(new_x), int(new_y), int(new_w), int(new_h))

def detect_red_green_regions(image, region1, region2):
    """
    检测两个区域中哪个是红色区域，哪个是绿色区域
    
    参数:
    image - 输入图像(BGR格式)
    region1 - 第一个区域(x, y, w, h)
    region2 - 第二个区域(x, y, w, h)
    
    返回:
    (red_region, green_region) - 红色区域和绿色区域的坐标
    """
    # 转换为HSV颜色空间（更容易检测颜色）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
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

cam  = camera.Camera(160, 160) 
disp = display.Display()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# 初始化激光点坐标位置变量
point_x = 0
point_y = 0

last_img_cv_gray = None
while not app.need_exit():
    img = cam.read()
    img_show = img.copy()

    # 转换为 cv2 格式
    img_raw = image.image2cv(img, False, False)
    
    # 转换为灰度 为了方便滤波去除噪点
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    img = cv2.medianBlur(img, 3)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    if last_img_cv_gray is None:
        last_img_cv_gray = img.copy()

    # 计算差值
    img_diff = cv2.absdiff(img, last_img_cv_gray)

    # 二值化处理
    _, img_binary = cv2.threshold(img_diff, 25, 255, cv2.THRESH_BINARY)
    
    # 膨胀处理填充激光点空洞
    img_binary = cv2.dilate(img_binary, kernel, iterations=2)
    
    # 更新上一帧图片
    last_img_cv_gray = img.copy()

    # check point 放开注释,应能黑色背景下激光点二值化后的白点
    img_show = image.cv2image(img_binary, False, False) 
    disp.show(img_show)

    # # 查找轮廓
    # contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    # # 初始化用于记录红色和绿色的颜色区域
    # all_region = []
    # for contour in contours:
    #     # 计算轮廓面积
    #     contour_area = cv2.contourArea(contour)

    #     # 激光点不大，过滤掉大范围的轮廓
    #     if contour_area < 350: 
    #         # 计算轮廓质心作为激光点位置坐标
    #         M = cv2.moments(contour)
    #         point_x = int(M["m10"] / M["m00"])
    #         point_y = int(M["m01"] / M["m00"])
    #         img_show.draw_cross(point_x, point_y, image.COLOR_BLUE, 5, 2)
    #         # 获取激光点轮廓的外接矩形 
    #         x, y, w, h = cv2.boundingRect(contour)
    #         x, y, w, h = center_scale_rect(x,y,w,h) # 对矩形进行缩放
    #         all_region.append([x,y,w,h])    # 记录区域

    # print(all_region)
    
    # if len(all_region) > 1:
    #     # 对两个 rio 进行区分
    #     red_region, green_region = detect_red_green_regions(img_cv, all_region[0], all_region[1])
    #     # 图中标注
    #     img_show.draw_rect(red_region[0],red_region[1],red_region[2],red_region[3], image.COLOR_GREEN, 2)
    #     img_show.draw_rect(green_region[0],green_region[1],green_region[2],green_region[3], image.COLOR_RED, 2)
        
    #     # img_show.draw_cross(point_x, point_y, image.COLOR_BLUE, 5, 2)

    #     # img.draw_cross(point_x, point_y, image.COLOR_BLUE, 5, 2)
    # disp.show(img_show)
    # print("\n")
    # time.sleep(0.1)