from maix import camera, display, image, nn, app
import os


# 确定 .mud 文件的路径
model_dir = '/root/models/BlackAndWhiteChess'
mud_file = 'BlackAndWhiteChess.mud'

model_path = os.path.join(model_dir, mud_file)

# 初始化检测模型
detector = nn.YOLO11(model=model_path, dual_buff = True)

cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
cam.skip_frames(30)     # 跳过开头的30帧
disp = display.Display()


while not app.need_exit():
    img = cam.read()
    objs = detector.detect(img, conf_th = 0.5, iou_th = 0.45)
    for obj in objs:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_BLUE)
    disp.show(img)

