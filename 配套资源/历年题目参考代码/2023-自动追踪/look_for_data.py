from maix import image
from MyOpenCV import MyCV2
import numpy as np


class InforSeek:

    def __init__(self) -> None:
        self.cv = MyCV2()
        self.last_machined_img = None       # 初始化上处理后的图片

        # 初始化用于记录红色和绿色的颜色区域
        self.all_region = list()

    def look_for_edge(self, _RawEdgeImg:image.Image) -> dict:
        
        # 使用 opencv 进行图像处理
        img_machine_edge = self.cv.image_process(_RawEdgeImg)

        # 对处理后的图像进行角点检测
        contours = self.cv.find_corners(img_machine_edge)
        # print(f"一共寻找到{len(contours)}层\n")
        
        # 初始化用于记录内外层角点的列表
        all_corners = []
        contour_outer = []
        contour_inlayer = []

        # 开始逐层记录，正常应该为四层，我们取第一层和最后一层
        for contour in contours:
            # 计算参数，多边形逼近

            approx = self.cv.approach_polygon(contour)
            # 记录轮廓角点信息
            if len(approx) == 4:    # 如果有四个角点
                # print(f"得到四边形，进行排序")
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

        if len(all_corners) == 4:   # 如果是矩形
            # 获取最外侧和最内侧的角点坐标列表
            contour_outer = all_corners[0]
            contour_inlayer = all_corners[-1]

            # 获得最终的结果坐标
            edge_corners_dict = self.average_points(contour_outer, contour_inlayer)

            # 返回结果字典集
            return edge_corners_dict
        return {}
        
    def look_for_laser_spot(self, _RawLaserImg:image.Image):

        # 初始化激光点坐标位置变量
        point_x = 0
        point_y = 0

        # 使用 opencv 进行图像处理
        # img_machine_laser = self.cv.image_process(_RawLaserImg)
        img_machine_laser = _RawLaserImg
        
        # 判断上一次是否为空
        if self.last_machined_img is None:
            img_safe = image.Image(img_machine_laser.width(), img_machine_laser.height(), img_machine_laser.format())
            self.last_machined_img = img_safe.draw_image(0, 0, img_machine_laser)

        # 帧差法提取颜色区域
        self.all_region = self.cv.frame_dif_method(img_machine_laser, self.last_machined_img)
    
        # 更新上一帧的图片为下一次做准备
        img_safe = image.Image(img_machine_laser.width(), img_machine_laser.height(), img_machine_laser.format())
        self.last_machined_img = img_safe.draw_image(0, 0, img_machine_laser)
        
        # return self.all_region

        # 仅识别到两个区域时
        if len(self.all_region) == 2:
            # 对两个 rio 进行区分
            red_region, green_region = self.cv.detect_red_green_regions(_RawLaserImg, self.all_region[0], self.all_region[1])
            red_point = self.get_centroid(red_region)
            green_point = self.get_centroid(green_region) 
            # 获得质心坐标进行返回
            laser_spot_dict = {
                'red_dot': red_point,
                'green_dot': green_point
            }
            return laser_spot_dict
        elif len(self.all_region) == 1:
            red_point = self.get_centroid(self.all_region[0])
            # 获得质心坐标进行返回
            laser_spot_dict = {
                'red_dot': red_point
            }
            
            return laser_spot_dict
        return {}
        
    def average_points(self, list1, list2) -> dict:
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

    # 根据 x, y, w, h 得到质心坐标
    def get_centroid(self, region: list) -> list:
        """
            根据 x, y, w, h 得到质心坐标
        """
        return [(region[0]+region[2] / 2.0) , (region[1]+region[3] / 2.0) ]

    # 矩形的中心缩放缩放
    def center_scale_rect(self, x, y, w, h, scale=0.5):
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

