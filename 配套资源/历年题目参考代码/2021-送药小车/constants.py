
# (x,y,w,h,weight)=(矩形左上顶点的坐标(x,y),矩形宽度和高度(w,h),权重)
TRA_ROIS = [ # [ROI, weight]
    (96, 0, 128, 32, 0.05),    # top
    (64, 32, 192, 32, 0.05),
    (64, 64, 192, 32, 0.10),
    (64, 96, 192, 32, 0.25),
    (64, 128, 192, 32, 0.3),
    (64, 160, 192, 32, 0.25)
]   # 屏幕6分

SIDE_ROIS = [ # [ROI, weight]
    (0, 0, 64, 192, -1),   # left
    (256, 0, 32, 192, -1),  # right
    (16, 0, 198, 32, -1)    # top
]

_cross_timeout = 5000   # 通过路口超时时间 -ms

_Kp = 0.3
_ki = 0.1
_kd = 0.1
_imax = 53



MODEL_DIR = '/root/models/'
MUD_FILE = 'DrugDeliveryCart2_int8.mud'