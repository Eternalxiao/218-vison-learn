from maix import camera, display, app, image, time
import cv2
import numpy as np


def average_points(list1, list2) -> dict:
    """
    计算两个列表中对应坐标点的平均值
    
    参数:
        list1: 第一个点列表，格式为 [[x1,y1], [x2,y2], ...]
        list2: 第二个点列表，格式为 [[x1,y1], [x2,y2], ...]
        
    返回:
        包含平均坐标的新列表，格式为 [[avg_x1, avg_y1], [avg_x2, avg_y2], ...]
    """

    index_dict = {
        0: 'first',
        1: 'second',
        2: 'third',
        3: 'forth'
    }
    i = 0
    result = {}
    for point1, point2 in zip(list1, list2):
        # 计算x坐标平均值
        avg_x = (point1[0] + point2[0]) / 2.0
        # 计算y坐标平均值
        avg_y = (point1[1] + point2[1]) / 2.0
        # 添加到结果列表
        result[index_dict[i]] = (avg_x, avg_y)
        i += 1
    return result

# 初始化相机和显示器
cam = camera.Camera(160, 160)
disp = display.Display()        
cam.skip_frames(30)     # 跳过开头的30帧

# 闭运算卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

while not app.need_exit():
    img = cam.read()
    img_safe = image.Image(img.width(), img.height(), img.format())
    img_show = img_safe.draw_image(0, 0, img)

    # 转换为 CV2 格式
    img_raw = image.image2cv(img, copy=False)

    # 转换为灰度 为了方便滤波去除噪点
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    img = cv2.medianBlur(img, 3)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    
    # 边缘检测
    # 自动计算
    median = np.median(img)
    threshold1 = int(max(0, 0.95 * median))
    threshold2 = int(min(255, 1.35 * median))
    img = cv2.Canny(img, threshold1, threshold2)

    # # 效果不好可以再次可以选择性再次进行闭运算，加强效果
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  
    
    # # 为了调试：
    # img_show1 = image.cv2image(img, copy=False)
    # disp.show(img_show1) 
    # time.sleep(0.2)
    # disp.show(img_show)
    # time.sleep(0.2)


    # ######################### 图像处理结束 #########################

    ######################### 边缘检测 #########################
    """
    cv2.findContours(img,mode,method)  -- 返回轮廓列表和层次结构信信息
    # # mode:   RETR_EXTERNAL 只检测外围轮廓
                RETR_TREE     按照树形存储轮廓，从大大小，从左到右
    # # method: CHAIN_APPROX_NONE       保存轮廓上所有的点
                CHAIN_APPROX_SIMPLE     只保存角点
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"一共寻找到{len(contours)}层\n")
    
    # 初始化用于记录内外层角点的列表
    all_corners = []
    contour_outer = []
    contour_inlayer = []

    # 开始逐层记录，正常应该为四层，我们取第一层和最后一层
    for contour in contours:
        # 计算参数，多边形逼近

        # 获得近似值（与周长相关）：
        epsilon = 0.02 * cv2.arcLength(contour, True)

        """
        cv2.approxPolyDP(curve, epsilon, closed) -- 返回近似后的多边形点集
        # # curve: 输入的轮廓点集
        # # epsilon: 近似精度，值越小，近似越精确，一般与周长有关
        """
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 记录轮廓角点信息
        if len(approx) == 4:
            # 按照顺序对角点排序（左上，右上，右下，左下）
            corners = approx.reshape((4, 2))
            rect = np.zeros((4, 2), dtype="int")
            points_sum = corners.sum(axis=1)
            rect[0] = corners[np.argmin(points_sum)]
            rect[2] = corners[np.argmax(points_sum)]
            points_diff = np.diff(corners,axis=1)
            rect[1] = corners[np.argmin(points_diff)]
            rect[3] = corners[np.argmax(points_diff)]
            corners = rect

            # 转换为列表存储轮廓角点信息
            all_corners.append(corners.tolist())

    if len(all_corners) != 4:
        continue

    # 获取最外侧和最内侧的角点坐标列表
    contour_outer = all_corners[0]
    contour_inlayer = all_corners[-1]

    # 获得最终的结果坐标
    edge_corners_dict = average_points(contour_outer, contour_inlayer)

    # img_show = image.cv2image(img, copy=False)
    print(edge_corners_dict)
    disp.show(img_show)         # Show image to screen
    fps = time.fps()            # Calculate FPS between last time fps() call and this time call.
    print(f"time: {1000/fps:.02f}ms, fps: {fps:.02f}") # print FPS in console 

