

# 用于标定外界框的相关常量
REFERENCE_DISTANCE = 100.0  # 外接框的标定距离 (cm) - 统一使用厘米
REFERENCE_AREA = 6493  # 外接框的标定像素面积 (px^2) - 主要标定参数

# A4纸张的实际物理尺寸 (mm)
A4_REAL_WIDTH_MM = 210.0   # A4纸张实际宽度 (mm)
A4_REAL_LENGTH_MM = 297.0  # A4纸张实际长度 (mm)

# 边界框的实际尺寸 (mm) - 内边界框尺寸，用于内部几何形状标定
BOUNDARY_REAL_WIDTH_MM = 170.0   # 内边界框实际宽度 (mm) = 210-40
BOUNDARY_REAL_HEIGHT_MM = 257.0  # 内边界框实际高度 (mm) = 297-40


# 重叠检测相关常量
# 1. 核心校准参数 - 统一使用面积标定
SYSTEM_OFFSET_CM = 0.0  # 系统偏移，用于补偿固定误差，面积标定通常不需要偏移


# 2. 物理参数
ANGLE_TOLERANCE = 22.0    # 正方形直角识别的允许偏差范围：90° ± 22°（度）
DEDUPLICATION_TOLERANCE = 13.0    # 去重容差（像素） -- 中心像素间隔
SQUARE_ASPECT_RATIO_TOLERANCE = 1.25  # 正方形长宽比容差

# 3. 图像处理与轮廓筛选参数
BINARY_THRESHOLD = 100   # 二值化阈值
ROI_SHRINK_PIXELS = 8   # ROI区域收缩像素数
MORPH_ITERATIONS = 2    # 形态学操作迭代次数

PARENT_CONTOUR_MIN_AREA = 1800 - 100  # A4纸轮廓最小面积阈值（像素²）   -- 小于此的都将被过滤掉
CHILD_CONTOUR_MIN_AREA = 18  # 子轮廓（正方形）最小面积阈值（像素²）
