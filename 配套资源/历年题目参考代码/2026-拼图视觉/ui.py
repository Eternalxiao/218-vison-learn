"""
UI - 简单按钮工具 (大字体/粗边框, 提升小屏可点击性)
"""
from maix import touchscreen, display, image


class GUI:
    def __init__(self, img_w=640, img_h=448):
        self.img_w = img_w
        self.img_h = img_h
        self._ts = touchscreen.TouchScreen()
        self._disp = display.Display()
        self._last_pressed = 0

    def _to_img(self, tx, ty):
        x, y = image.resize_map_pos_reverse(
            self.img_w, self.img_h,
            self._disp.width(), self._disp.height(),
            image.Fit.FIT_CONTAIN, tx, ty)
        return max(x, 0), max(y, 0)

    def get_click(self):
        tx, ty, pressed = self._ts.read()
        if self._last_pressed != pressed:
            self._last_pressed = pressed
            if pressed:
                ix, iy = self._to_img(tx, ty)
                return ix, iy, True
        return 0, 0, False

    def get_touching(self):
        tx, ty, pressed = self._ts.read()
        if pressed:
            ix, iy = self._to_img(tx, ty)
            return ix, iy, True
        return 0, 0, False

    @staticmethod
    def hit(ix, iy, x, y, w, h):
        return x <= ix <= x + w and y <= iy <= y + h

    @staticmethod
    def draw_btn(img, x, y, w, h, label, active=False):
        # 粗边框(3px) + 自动放大字体: 优先scale=2, 放不下退回scale=1
        color = image.COLOR_GREEN if active else image.COLOR_WHITE
        img.draw_rect(x, y, w, h, color, 3)
        scale = 2
        try:
            s = image.string_size(label, scale=scale)
            if s.width() > w - 6 or s.height() > h - 4:
                scale = 1
                s = image.string_size(label, scale=1)
        except Exception:
            scale = 1
            s = image.string_size(label)
        sx = x + (w - s.width()) // 2
        sy = y + (h - s.height()) // 2
        # 防止文字画出图像边界(按钮贴边时标签被裁切)
        try:
            sx = max(0, min(sx, img.width() - s.width() - 1))
            sy = max(0, min(sy, img.height() - s.height() - 1))
        except Exception:
            pass
        img.draw_string(sx, sy, label, image.COLOR_WHITE, scale=scale)

    def show(self, img):
        self._disp.show(img)
