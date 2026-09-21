from constants import MAX_DISTANCE
from maix import image
import numpy as np


class GetBoard:
    """
        获得正确的棋盘坐标信息，并且实现棋盘的映射
    """

    # 初始化棋盘中心坐标列表
    _last_board_centre_list = []
    
    _flag = 1

    def __init__(self) -> None:
        self.MAX_DISTANCE = MAX_DISTANCE

    # 寻找棋盘中心坐标信息
    def find_board(self, img:image.Image, board_threshold: list[list[int]]):
        """
            二值化寻找棋盘中心坐标信息，返回记录有中心坐标的信息“列表”
        """
        board_binary_img = img.binary(thresholds=board_threshold, invert=False, zero=False, copy=True)      
        # 获取棋盘信息
        board_blobs = board_binary_img.find_blobs(
                thresholds = [[90, 100, 0, 0, 0, 0]],
                x_stride = 1, y_stride = 1,      # skip 的像素
                area_threshold = 15,        # 面积阈值
                pixels_threshold = 15,      # 像素阈值
                merge= False,               # 合并未过滤的色块
                margin = 1,                 # 合并矩形框边界的距离
                x_hist_bins_max = 0, y_hist_bins_max = 0
                )

        # 更新棋盘信息
        if len(board_blobs) == 9:
            # print(f"识别到九个棋盘色块,调整数据")
            new_board_centre_list = []
            for _ in board_blobs:
                # 记录九宫格的九个小格中心坐标
                new_board_centre_list.append((_.x() + _.w()/2, _.y() + _.h()/2))
            
            # 对中心坐标进行排序
            new_board_centre_list = self.sort_tilted_grid(new_board_centre_list)
            self._last_board_centre_list = new_board_centre_list    # 本来取上一次是为了判断本次错误的，但是没有这个数学逻辑，使用没用了
                    
        # print("更新棋盘坐标为：\n"+f"{board_centre_list}")  # 得到1-9的中心坐标
        # print("更新棋盘坐标")  # 得到1-9的中心坐标
        return self._last_board_centre_list

    # 将所有棋子映射到棋盘中，并更新棋盘数据
    def map_all_chess_to_board(self, chess_list, board, board_centre_list, chess_class: int) -> list:
        """
        :param chess_list: 棋子中心坐标列表
        :param board: 对弈棋盘 《==》 全0的棋盘
        :param board_centre_list: 棋盘中心坐标列表
        :param chess_class: 本次要映射的棋子类别, 白棋为1， 黑棋为2
        :return: 0 1 2表示的棋盘信息
        """
        # print(f"这是map_all中传入的：{board=}")
        for chess in chess_list:
            pos = self.map_chess_to_board((chess[0], chess[1]), board_centre_list)
            if pos is not None:
                # 处理重叠冲突（后检测的覆盖先检测的）
                board[pos[0]][pos[1]] = chess_class

        # 返回映射以后的棋盘
        # print(f"这是map_all中返回的：{board=}")
        return board

        # 对棋盘中心坐标进行先行后列的排序
    
    # 通过距离大小判断棋子位置
    def map_chess_to_board(self, chess_center, board_centre_list):
        """
        :param chess_center: 棋子中心坐标 (x, y)
        :param board_centre_list: 棋盘中心坐标列表
        :param max_distance: 有效匹配阈值(根据棋盘实际尺寸调整)
        :return: (row, col) 或 None
        """
        min_dist_sq = float('inf')
        best_pos = None
        for i in range(3):
            for j in range(3):
                # 计算平方距离避免开根号
                dx = chess_center[0] - board_centre_list[i][j][0]
                dy = chess_center[1] - board_centre_list[i][j][1]
                dist_sq = dx*dx + dy*dy
                
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_pos = (i, j)
        
        # 验证距离是否在有效范围内
        if min_dist_sq <= self.MAX_DISTANCE:
            return best_pos
        return None

    # 对坐标进行先行后列的排序
    def sort_tilted_grid(self, points):
        """
        对倾斜九宫格中心点排序，行方向为x轴右向，列方向为y轴下向
        :param points: 九个中心点坐标列表，格式 [(x1,y1), (x2,y2), ...]
        :return: 排序后的3x3矩阵，排列顺序：行（左→右）优先，列（上→下）次之
        """
        points = np.array(points)       # 九行俩列存储

        # === 阶段1：数据去中心化 ===
        centroid = np.mean(points, axis=0)  # 计算几何中心
        centered = points - centroid        # 平移坐标系到中心点 -->> 向下为y轴正方向的坐标记录

        # 判断本次的顺逆旋转
        x_list = [x for x, y in centered]
        y_list = [y for x, y in centered]
        max_x_index = x_list.index(max(x_list))     # 得到最大x的下标

        if y_list[max_x_index] < 0:     # 通过判断最大x的纵坐标来判断顺逆时针 --》》 向下为正所以y小于零为顺时针
            # 顺时针
            clockwise_flag = True
        else:
            # 逆时针
            clockwise_flag = False

        # === 阶段2：计算坐标轴方向 ===
        # 计算协方差矩阵的特征向量 -> 判断行和列的相关性
        cov_matrix = np.cov(centered.T)        # 协方差矩阵, centered.T 变为了2*9，0行为x 1行为y
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # 特征分解
        # print(f"{eigenvalues=}, {eigenvectors=}")
        # 提取主次方向（此处主方向=行方向，次方向=列方向）
        eigenvalues =  np.abs(eigenvectors[0])  # 重载特征值，适应顺逆时针的不同情况，每次x都是取较大值
        row_vector = eigenvectors[:, np.argmax(eigenvalues)]  # 行方向候选（x轴方向）
        col_vector = eigenvectors[:, np.argmin(eigenvalues)]  # 列方向候选（y轴方向）

        # === 阶段3：方向向量修正 ===
        # 修改 顺时针旋转时侯：row_vector(+, +),col_vector(-, +)
        #      逆时针旋转时侯：row_vector(+, -),col_vector(+, +)
        # 检查方向向量
        # print(f"未修改：{row_vector=}, {col_vector=}")

        if clockwise_flag:      # 如果是顺时针
            # row_vector(+, +),col_vector(-, +)
            row_vector = np.abs(row_vector)
            col_vector[0] = -abs(col_vector[0])
            col_vector[1] = abs(col_vector[1])
        else:
            # row_vector(+, -),col_vector(+, +)
            col_vector = np.abs(col_vector)
            row_vector[0] = abs(col_vector[0])
            row_vector[1] = -abs(col_vector[1])

        # print(f"修改后：{row_vector=}, {col_vector=}")

        # === 阶段4：坐标投影 ===
        # 数学原理：投影值决定行列顺序
        # print(f"{points=}")
        row_proj = np.dot(centered, row_vector)  # 行投影值（决定左→右顺序）
        col_proj = np.dot(centered, col_vector)  # 列投影值（决定上→下顺序）

        # === 阶段5：双重排序 ===
        # 第一步：按列投影（y轴）升序排序 → 确定行归属
        # 较小的列投影值对应更上方的行
        row_indices = np.argsort(col_proj)  # 获取排序索引
        
        # 第二步：在每行内按行投影（x轴）升序排序 → 确定列顺序
        sorted_points = []
        for i in range(0, 9, 3):  # 将数据分成3行
            # 提取当前行的三个点
            current_row = points[row_indices[i:i+3]]
            # 提取对应的行投影值
            current_proj = row_proj[row_indices[i:i+3]]
            # 按行投影值排序（左→右）
            col_order = np.argsort(current_proj)
            sorted_points.extend(current_row[col_order].tolist())
        # print(f"{clockwise_flag=}")
        return np.array(sorted_points).reshape(3,3,2)
