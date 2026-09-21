from maix import camera, display, image, nn, app
from maix import touchscreen, app, time
import os

# 初始化摄像头指定图像的宽度、高度、帧率
cam = camera.Camera(320, 224, fps=30)
# 初始化屏幕显示
disp = display.Display()


# # 设置为灰度图像
# cam = camera.Camera(640, 480, image.Format.FMT_GRAYSCALE)
# # 设置为 NV21 图像
# cam = camera.Camera(640, 480, image.Format.FMT_YVU420SP)

cam.skip_frames(30)     # 跳过开头的30帧
# cam.gain(100)         # 设置增益
cam.awb_mode(1)			# 0,开启白平衡;1,关闭白平衡
cam.exp_mode(0)         # 切换回自动曝光模式

cam.luma(50)		    # 设置亮度，范围[0, 100]
cam.constrast(50)		# 设置对比度，范围[0, 100]
cam.saturation(50)		# 设置饱和度，范围[0, 100]


"""
# 显示图片
while not app.need_exit():
    # 从文件系统中读取图片 注意是MaixCAM的文件系统
    # img = image.load("/root/image.jpg")
    # img = cam.read()
    # img=cam.read().binary(Threshold, invert=False, zero=True)     # 二值化
    # 对图片进行畸变矫正 --> 调整strength的值直到画面不再畸变
    # img = img.lens_corr(strength=1.5)	

    # 在图像上画框
    img.draw_rect(
        10, 10, 100, 100,   # x, y, w, h
        image.Color.from_rgb(255, 0, 0),    # image.Color.from_rgb方法创建颜色, 
        thickness=1         # thickness 指定线宽 -1 是实心框
    )

    # 在图像上写字符串
    img.draw_string(
        10, 10,         # x 和 y 是文字的左上角坐标
        "Hello MaixPy", # text 是要写的文字
        image.Color.from_rgb(255, 0, 0), # color 是文字的颜色
        scale=2     # 放大字体
    )

    # 获取字体的宽度和高度
    w, h = image.string_size("Hello MaixPy", scale=2)
    print(w, h)

    # 画线
    img.draw_line(10, 10, 100, 100, image.Color.from_rgb(255, 0, 0))

    # 画圆
    img.draw_circle(100, 100, 50, image.Color.from_rgb(255, 0, 0))

    # 画十字 -->>  十字的延长大小是5，所以线段长度为2 * size + thickness
    img.draw_cross(100, 100, image.Color.from_rgb(255, 0, 0), size=5, thickness=1)
    
    # 画箭头 -->>  (10, 10)画一个红色的箭头，箭头的终点是(100, 100)，线宽是1
    img.draw_arrow(10, 10, 100, 100, image.Color.from_rgb(255, 0, 0), thickness=1)

    # 画图
    img2 = image.Image(100, 100, image.Format.FMT_RGB888)
    img.draw_image(10, 10, img2)
    # resize 返回缩放后的新图像
    img_new = img.resize(160, 120)

    # crop 返回裁剪后的新图像
    img_new = img.crop(10, 10, 100, 100)
    
    # rotate 返回旋转后的新图像 
    img_new = img.rotate(90)

    # copy 拷贝一份独立的图像
    img_new = img.copy()

    # 转换格式返回新的图像对象
    img_jpg = img.to_format(image.Format.FMT_JPEG)

    # affine 进行仿射变换
    #
    # 即提供当前图中三个及以上的点坐标，以及目标图中对应的点坐标
    #可以自动进行图像的旋转、缩放、平移等操作变换到目标图像：

    # img_new = img.affine([(10, 10), (100, 10), (10, 100)], [(10, 10), (100, 20), (20, 100)])

    # draw_keypoints 在图像上画出关键点

    # 在坐标(10, 10)、(100, 10)、(10, 100)画三个红色的关键点，
    # 关键点的大小是10, 线宽是1, 不填充。

    keypoints = [10, 10, 100, 10, 10, 100]
    img.draw_keypoints(
        keypoints, 
        image.Color.from_rgb(255, 0, 0), 
        size=10, 
        thickness=1, 
        # fill=False
    )

    # 和 bytes 数据相互转换 得到新的bytes对象，使用 from_bytes 还原
    img_to_bytes = image.Image(320, 240, image.Format.FMT_RGB888)
    data = img_to_bytes.to_bytes()
    print(type(data), len(data), img.data_size())
    img_jpeg = image.from_bytes(320, 240, image.Format.FMT_RGB888, data)

    disp.show(img)

    # 可以对图片进行修改后保存到指定文件中
    # img.save("/root/image.jpg")

"""


# # 读取触摸
# ts = touchscreen.TouchScreen()

# pressed_already = False
# last_x = 0
# last_y = 0
# last_pressed = False
# while not app.need_exit():
#     x, y, pressed = ts.read()
#     if x != last_x or y != last_y or pressed != last_pressed:
#         print(x, y, pressed)
#         last_x = x
#         last_y = y
#         last_pressed = pressed
#     if pressed:
#         pressed_already = True
#     else:
#         if pressed_already:
#             print(f"clicked, x: {x}, y: {y}")
#             pressed_already = False
#     time.sleep_ms(1)  # sleep some time to free some CPU usage


# thresholds = [[0, 80, 40, 80, 10, 80]]      # red
# # thresholds = [[0, 80, -120, -10, 0, 30]]    # green
# # thresholds = [[0, 80, 30, 100, -120, -60]]  # blue

# 寻找色块
# while 1:
#     img = cam.read()
#     blobs = img.find_blobs(
#         thresholds, 
#         pixels_threshold=500,   # 有效像素点阈值
#         area_threshold=10   # 面积阈值
#     )
#     for blob in blobs:
#         img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
#     disp.show(img)


# # thresholds = [[0, 80, 40, 80, 10, 80]]      # red
# thresholds = [[0, 80, -120, -10, 0, 30]]    # green
# # thresholds = [[0, 80, 30, 100, -120, -60]]  # blue

# # 使用 image.get_regression 寻找直线 -->> 通常只返回一个直线，多个可用 find_line
# while 1:
#     img = cam.read()
#     # 转换为灰度图像更适合单一环境的巡线
#     gray_img = img.to_format(image.Format.FMT_GRAYSCALE)	

#     lines = img.get_regression(
#         thresholds, 
#         pixels_threshold = 10,   # 过滤掉有效像素小的点   
#         area_threshold = 100     # 过滤像素面积小于此值的点
#         )
# 
#     for a in lines:
#         img.draw_line(
#             a.x1(), a.y1(),     # 直线的一个端点坐标
#             a.x2(), a.y2(),     # 直线的另一个端点坐标
#             image.COLOR_GREEN, 
#             2
#         )

#         theta = a.theta()   # 直线与y轴的夹角

#         rho = a.rho()       # 原点与直线的垂线的长度

#         # 转换为与x轴的夹角
#         if theta > 90:      
#             theta = 270 - theta
#         else:
#             theta = 90 - theta  
#         img.draw_string(
#         0, 0, 
#         "theta: " + str(theta) + ", rho: " + str(rho), 
#         image.COLOR_BLUE
#         )

#     disp.show(img)


# # 识别二维码使用 find_qrcodes()
# while 1:
#     img = cam.read()

#     # 识别到二维码并将返回值存到 qrcodes
#     qrcodes = img.find_qrcodes()
#     # qrcodes = image.QRCodeDetector()      # 采用硬件加速的方法识别

#     for qr in qrcodes:  # 遍历二维码列表
#         # 将二维码 rect
#         corners = qr.corners()  # 获取已扫描到的二维码的四个顶点坐标 -->> 四行2列
#         for i in range(4):
#             # 循环画线
#             img.draw_line(
#                 corners[i][0], corners[i][1], 
#                 corners[(i + 1) % 4][0], corners[(i + 1) % 4][1], 
#                 image.COLOR_RED
#                 )

#         # 显示读取的内容
#         img.draw_string(
#             qr.x(), qr.y() - 15,   # 在二维码的上方15个像素开始显示
#             qr.payload(),       # 获取二维码的内容
#             image.COLOR_RED
#             )

#     disp.show(img)


# # 条形码的识别 find_barcodes
# while 1:
#     img = cam.read()
    
#     # 识别到条形码信息并将返回值存到 barcodes
#     barcodes = img.find_barcodes()

#     # 遍历条形码列表，将其 rect， 并返回内容
#     for b in barcodes:      # 遍历条形码列表
#         rect = b.rect()
#         img.draw_rect(
#             rect[0], rect[1], rect[2], rect[3], 
#             image.COLOR_BLUE, 
#             2
#         )

#         img.draw_string(
#             0, 0,   
#             "payload: " + b.payload(),      # 读取条形码的信息
#             image.COLOR_GREEN
#         )

#     disp.show(img)

# 串口通信：
# from maix import uart, pinmap
# device = "/dev/ttyS0"   # usb转接口 - 反接
# # pinmap.set_pin_function("A16", "UART0_TX")
# # pinmap.set_pin_function("A17", "UART0_RX")
# serial0 = uart.UART(device, 115200)
