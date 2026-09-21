from maix import app

def rgb_to_lab(rgb):
    '''
    实现RGB值到LAB值的转换
    '''

    # RGB到XYZ的转换矩阵
    M = [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ]
    
    # 归一化RGB值
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    
    # 线性化RGB值
    r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
    
    # 计算XYZ值
    X = M[0][0] * r + M[0][1] * g + M[0][2] * b
    Y = M[1][0] * r + M[1][1] * g + M[1][2] * b
    Z = M[2][0] * r + M[2][1] * g + M[2][2] * b
    
    # XYZ到LAB的转换
    X /= 0.95047
    Y /= 1.0
    Z /= 1.08883
    
    def f(t):
        return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116
    
    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))
    
    return [L, a, b]

# 可靠的RGB转HSV函数
def rgb_to_hsv(rgb, tolerance_h=10, tolerance_s=40, tolerance_v=40) -> list:
    """
    将RGB颜色转换为HSV阈值列表
    输入: 
        rgb: [R, G, B] 每个值范围0-255
        tolerance_h: 色相容差
        tolerance_s: 饱和度容差
        tolerance_v: 明度容差
    输出: 
        HSV阈值列表 [[H_min, H_max, S_min, S_max, V_min, V_max]]
    """
    # 将RGB转换为0-1范围
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    
    # 计算最大值和最小值
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val
    
    # 计算色调(H)
    if delta == 0:
        h = 0
    elif max_val == r:
        h = 60 * (((g - b) / delta) % 6)
    elif max_val == g:
        h = 60 * (((b - r) / delta) + 2)
    else:  # max_val == b
        h = 60 * (((r - g) / delta) + 4)
    
    # 确保H在0-360范围内
    h = h % 360
    if h < 0:
        h += 360
    
    # 计算饱和度和明度
    s = (delta / max_val) * 100 if max_val != 0 else 0
    v = max_val * 100
    
    # 转换为OpenCV HSV范围
    h_cv = h / 2          # 0-180
    s_cv = s * 2.55       # 0-255
    v_cv = v * 2.55       # 0-255
    
    # 创建基础阈值
    h_min = max(0, int(h_cv - tolerance_h))
    h_max = min(180, int(h_cv + tolerance_h))
    s_min = max(0, int(s_cv - tolerance_s))
    s_max = min(255, int(s_cv + tolerance_s))
    v_min = max(0, int(v_cv - tolerance_v))
    v_max = min(255, int(v_cv + tolerance_v))
    
    base_threshold = [h_min, h_max, s_min, s_max, v_min, v_max]
    
    # 如果是红色，添加双阈值
    if h_min <= 10 or h_max >= 170 :
        # 红色低区阈值 (0-10)
        red_low = [0, 10, s_min, s_max, v_min, v_max]
        # 红色高区阈值 (170-180)
        red_high = [170, 180, s_min, s_max, v_min, v_max]
        return [red_high, red_low]
    
    return [base_threshold]

def set_configured_threshold(threshold):
    '''
    阈值参数信息存入配置文件
    '''
    if len(threshold) < 12:
        return 

    app.set_app_config_kv('demo_find_line', 'lmin', str(threshold[0]), True)
    app.set_app_config_kv('demo_find_line', 'lmax', str(threshold[1]), True)
    app.set_app_config_kv('demo_find_line', 'amin', str(threshold[2]), False)
    app.set_app_config_kv('demo_find_line', 'amax', str(threshold[3]), False)
    app.set_app_config_kv('demo_find_line', 'bmin', str(threshold[4]), False)
    app.set_app_config_kv('demo_find_line', 'bmax', str(threshold[5]), False)

def get_configured_threshold():
    '''
    获取所存储配置文件中的阈值参数
    '''
    thresholds = [
        [0, 10, 43, 255, 46, 255],     #红色阈值1   
        [156, 180, 43, 255, 46, 255]  #红色阈值2
    ]
    # thresholds = [0, 10, 43, 255, 46, 255],     #红色阈值1   
    

    # value_str = app.get_app_config_kv('demo_find_line', 'lmin','', False)
    # if len(value_str) > 0:
    #     threshold[0] = int(value_str)
    # value_str = app.get_app_config_kv('demo_find_line', 'lmax','', True)
    # if len(value_str) > 0:
    #     threshold[1] = int(value_str)
    # value_str = app.get_app_config_kv('demo_find_line', 'amin','', False)
    # if len(value_str) > 0:
    #     threshold[2] = int(value_str)
    # value_str = app.get_app_config_kv('demo_find_line', 'amax','', False)
    # if len(value_str) > 0:
    #     threshold[3] = int(value_str)
    # value_str = app.get_app_config_kv('demo_find_line', 'bmin','', False)
    # if len(value_str) > 0:
    #     threshold[4] = int(value_str)
    # value_str = app.get_app_config_kv('demo_find_line', 'bmax','', False)
    # if len(value_str) > 0:
    #     threshold[5] = int(value_str)
    return thresholds

