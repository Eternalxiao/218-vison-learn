import numpy as np

def sort_tilted_grid(points):
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

    print(f"修改后：{row_vector=}, {col_vector=}")

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
    print(f"{clockwise_flag=}")
    return np.array(sorted_points).reshape(3,3,2)
