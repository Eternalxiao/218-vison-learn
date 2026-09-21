"""Load the bundled YOLO11 model pair on MaixCAM Pro."""

from maix import app, camera, display, image, nn, time


MODEL_PATH = "/root/models/TIbegin_int8.mud"
CONFIDENCE = 0.5
IOU_THRESHOLD = 0.45


def main():
    detector = nn.YOLO11(model=MODEL_PATH, dual_buff=True)
    cam = camera.Camera(detector.input_width(), detector.input_height())
    screen = display.Display()

    while not app.need_exit():
        time.fps_start()
        frame = cam.read()
        objects = detector.detect(frame, conf_th=CONFIDENCE, iou_th=IOU_THRESHOLD)

        for obj in objects:
            label = detector.labels[obj.class_id]
            frame.draw_rect(obj.x, obj.y, obj.w, obj.h, image.COLOR_RED, 2)
            frame.draw_string(obj.x, max(0, obj.y - 16), "%s %.2f" % (label, obj.score))

        frame.draw_string(4, 4, "FPS %.1f" % time.fps(), image.COLOR_GREEN)
        screen.show(frame)


if __name__ == "__main__":
    main()
