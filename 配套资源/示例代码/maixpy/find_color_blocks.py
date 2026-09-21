"""Minimal MaixCAM Pro color-blob example.

Input: live camera frames.
Output: bounding boxes, centers, and FPS on the device display.
"""

from maix import app, camera, display, image, time


WIDTH = 320
HEIGHT = 224
ROI = [0, 0, WIDTH, HEIGHT]

# LAB ranges are a two-dimensional list so one target may use several ranges.
THRESHOLDS = [
    [0, 80, -120, -10, 0, 30],
]


def main():
    cam = camera.Camera(WIDTH, HEIGHT, fps=30)
    screen = display.Display()
    cam.skip_frames(30)

    while not app.need_exit():
        time.fps_start()
        frame = cam.read()
        blobs = frame.find_blobs(
            THRESHOLDS,
            roi=ROI,
            pixels_threshold=100,
            area_threshold=100,
            merge=True,
        )

        for blob in blobs:
            x, y, width, height = blob.rect()
            frame.draw_rect(x, y, width, height, image.COLOR_GREEN, 2)
            frame.draw_cross(blob.cx(), blob.cy(), image.COLOR_RED, size=5)
            frame.draw_string(blob.cx() + 5, blob.cy(), "%d,%d" % (blob.cx(), blob.cy()))

        fps = time.fps()
        frame.draw_string(4, 4, "FPS %.1f" % fps, image.COLOR_GREEN)
        screen.show(frame)


if __name__ == "__main__":
    main()
