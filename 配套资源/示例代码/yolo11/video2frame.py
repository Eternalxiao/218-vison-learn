# ============================================================
# video2frame.py - 视频按间隔抽帧
# 从一段视频中每隔 INTERVAL 帧保存一张图片，并缩放到部署画面尺寸。
# 不会修改源视频；如果输出目录中已有同前缀帧，则停止，防止覆盖。
# 用法：修改下方调参区，然后直接运行 python video2frame.py
# MaixCam 中使用默认相机录制的视频保存路径通常为：maixapp/share/video
# ============================================================

from pathlib import Path

# ==================== 调参区 ====================
INPUT_VIDEO = Path("input.mp4")   # 输入视频路径（必须修改为实际视频文件路径）
OUTPUT_DIR  = Path("frames")      # 抽帧图片保存目录（建议使用新目录或空目录）
INTERVAL    = 9                   # 每 INTERVAL 帧保存一张（实际时间间隔还取决于视频 FPS）
WIDTH       = 320                 # 输出图片宽（对应本项目板端输入，可根据部署画面尺寸调整）
HEIGHT      = 224                 # 输出图片高
PREFIX      = "frame"             # 文件名前缀；不同视频用不同前缀，避免输出目录里已有同名帧时报错
# ================================================


def save_jpeg(path, frame):
    """使用 imencode 保存，避免 Windows 中文路径导致写入失败。"""
    import cv2

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError(f"无法编码该帧：{path}")
    encoded.tofile(path)


def main():
    # 检查参数是否合法
    if INTERVAL < 1 or WIDTH < 1 or HEIGHT < 1:
        raise ValueError("INTERVAL、WIDTH 和 HEIGHT 必须为正数")

    # 检查输入视频是否存在
    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"找不到输入视频：{INPUT_VIDEO}")

    # 检查文件名前缀是否合法
    if not PREFIX or any(character in PREFIX for character in '<>:"/\\|?*'):
        raise ValueError("PREFIX 必须是合法的 Windows 文件名前缀")

    import cv2

    # 创建输出目录；如果目录中已有同前缀 jpg，则停止，避免覆盖旧数据
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT_DIR.glob(f"{PREFIX}_*.jpg")):
        raise FileExistsError("输出目录中已存在同前缀的帧图片；请更换输出目录或 PREFIX")

    # 打开视频
    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{INPUT_VIDEO}")

    # 读取视频信息
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0.0
    print(f"源视频帧数={total_frames} FPS={fps:.2f} 时长={duration:.2f}秒")

    saved = 0
    frame_number = 0

    # 逐帧读取，按间隔抽帧并缩放保存
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_number += 1

        # 不是间隔点则跳过
        if frame_number % INTERVAL:
            continue

        # 缩放到目标尺寸
        resized = cv2.resize(frame, (WIDTH, HEIGHT))

        # 保存为 jpg，文件名从 00000 开始编号
        destination = OUTPUT_DIR / f"{PREFIX}_{saved:05d}.jpg"
        save_jpeg(destination, resized)
        saved += 1

    capture.release()
    print(f"已保存={saved} 输出目录={OUTPUT_DIR}")


if __name__ == "__main__":
    main()