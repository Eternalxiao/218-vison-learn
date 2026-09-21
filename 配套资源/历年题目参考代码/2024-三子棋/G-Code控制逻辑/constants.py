


from re import M


X_ERROR = 0         # 设置x方向的坐标系误差
Y_ERROR = 0         # 设置y方向的坐标系误差
ERRORS = (X_ERROR, Y_ERROR)

MAX_DISTANCE = 60   # 俩中心格坐标的最大距离
MODEL_DIR = '/root/models/BlackAndWhiteChess'
MUD_FILE = 'BlackAndWhiteChess.mud'


BLACK_EDGE = 90     # 黑色棋子放置区的边界，黑棋子在左边
WHITE_EDGE = 250    # 白色棋子放置区的边界，白棋子在右边