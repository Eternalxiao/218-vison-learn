"""
2026拼图 - 主程序
状态机: menu / debug / calibrate / calib_offset / running
"""
from maix import camera, app, time, image
import cv2
import numpy as np
import json

import vision
import solver
from my_utils import MyUtils
from ui import GUI
import alert_io

# ==================== 读配置 ====================
with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

W = cfg["camera"]["width"]
H = cfg["camera"]["height"]
LENS_STRENGTH = cfg["camera"].get("lens_corr", 0)

# ==================== 初始化 ====================
cam = camera.Camera(W, H)
cam.skip_frames(cfg["camera"]["skip_frames"])


def apply_camera_settings(cam, cfg):
    """应用亮度/对比度/曝光/增益 (暗场手动拉曝光/增益)"""
    cs = cfg["camera"]
    try:
        cam.luma(cs.get("luma", 50))
        try:
            cam.constrast(cs.get("contrast", 50))  # MaixPy API 拼写为 constrast
        except Exception:
            try:
                cam.contrast(cs.get("contrast", 50))
            except Exception:
                pass
        cam.saturation(cs.get("saturation", 50))
        exp = cs.get("exposure", 0)
        gn = cs.get("gain", 0)
        if exp and exp > 0:
            cam.exposure(exp)
        if gn and gn > 0:
            cam.gain(gn)
        print("[cam] luma=%s exp=%s gain=%s" % (cs.get("luma"), exp, gn))
    except Exception as e:
        print("[cam] settings fail:", e)


apply_camera_settings(cam, cfg)
gui = GUI(W, H)
utils = MyUtils(cfg)
aio = alert_io.AlertIO(cfg)

show_binary = False
save_confirm = False
calib_msg = ""

# ==================== run-state globals (new flow) ====================
arm_cnt = 0
settle_cnt = 0
t0_ms = 0
run_ctx = None
pnt_idx = 0
px_manual = False


def save_config():
    import os
    tmp = "config.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, "config.json")
    print("[cfg] saved")
    try:
        aio.play_once()  # 保存成功提示音(防卡顿: 听到响=已保存)
    except Exception:
        pass


def draw_dashed_line(img, x1, y1, x2, y2, color, thickness=2, dash=8, gap=8):
    dx, dy = x2 - x1, y2 - y1
    length = max(int(max(abs(dx), abs(dy))), 1)
    seg = dash + gap
    for t in range(0, length, seg):
        t2 = min(t + dash, length)
        sx = x1 + dx * t // length
        sy = y1 + dy * t // length
        ex = x1 + dx * t2 // length
        ey = y1 + dy * t2 // length
        img.draw_line(sx, sy, ex, ey, color, thickness)


def draw_roi_overlay(img, roi_info):
    roi = roi_info["roi"]
    piece_rect = roi_info["piece_rect"]
    puzzle_rect = roi_info["puzzle_rect"]
    if roi is not None:
        rx, ry, rw, rh = roi
        img.draw_rect(rx, ry, rw, rh, image.COLOR_YELLOW, 2)
    if piece_rect is not None:
        px, py, pw, ph = piece_rect
        img.draw_rect(px, py, pw, ph, image.COLOR_GREEN, 2)
    if puzzle_rect is not None:
        ux, uy, uw, uh = puzzle_rect
        img.draw_rect(ux, uy, uw, uh, image.COLOR_RED, 2)


def full_frame_binary(binary, roi_info, W, H):
    """把碎片区域二值图贴回全图尺寸, 便于整屏查看(Bin键)"""
    canvas = np.zeros((H, W), np.uint8)
    pr = roi_info.get("piece_rect")
    if pr is not None:
        x, y, w, h = pr
        bh, bw = binary.shape[:2]
        cw = min(bw, W - x)
        ch = min(bh, H - y)
        if cw > 0 and ch > 0:
            canvas[y:y+ch, x:x+cw] = binary[:ch, :cw]
    return canvas

# ==================== 新流程: 计时/等待刷屏/声光提示/二次复查 ====================

def elapsed_s(t0):
    """自t0(ticks_ms)以来经过的整秒"""
    return (time.ticks_ms() - t0) // 1000


def deadline_hit(t0, cfg):
    """是否超过 alert.deadline_s (规则要求2分钟内必须声光提示)"""
    return elapsed_s(t0) >= cfg["alert"]["deadline_s"]


def wait_with_screen(sec, title, t0, cfg, gui, cam, scale=2, bar_h=34, overlay=False):
    """等待sec秒, 期间刷新屏幕显示倒计时(解决画面冻结)
    超过deadline或需要退出时返回False, 正常等完返回True"""
    t_end = time.ticks_ms() + sec * 1000
    dl = cfg["alert"]["deadline_s"]
    while time.ticks_ms() < t_end:
        if app.need_exit():
            return False
        if elapsed_s(t0) >= dl:
            return False
        remain = (t_end - time.ticks_ms() + 999) // 1000
        frame = cam.read()
        if LENS_STRENGTH > 0:
            frame = frame.lens_corr(strength=LENS_STRENGTH)
        if overlay:
            draw_run_overlay(frame)
        frame.draw_rect(0, 0, W, bar_h, image.COLOR_BLACK, -1)
        frame.draw_string(6, bar_h // 2 - 8, "%s next:%ds T:%d/%ds" % (
            title, remain, elapsed_s(t0), dl), image.COLOR_YELLOW, scale=scale)
        gui.show(frame)
        time.sleep_ms(200)
    return True


def run_alert(gui, cam, cfg, aio, ok=True):
    """完成声光提示: LED闪烁 + 喇叭WAV + 屏幕交替闪, 硬件失败全部容错
    ok=True 绿色闪(成功); ok=False 黄色闪(有疑似遗留/超时兜底)"""
    a = cfg["alert"]
    dur = a.get("duration_s", 8)
    t_end = time.ticks_ms() + dur * 1000
    last_beep = -10000
    phase = 0
    while time.ticks_ms() < t_end and not app.need_exit():
        now = time.ticks_ms()
        if now - last_beep >= 2600:
            aio.play_once()
            last_beep = now
        aio.led_toggle()
        phase = 1 - phase
        frame = cam.read()
        if ok:
            bg = image.COLOR_GREEN if phase == 0 else image.COLOR_WHITE
        else:
            bg = image.COLOR_YELLOW if phase == 0 else image.COLOR_BLACK
        frame.draw_rect(0, 0, W, H, bg, -1)
        txt = "DONE!" if ok else "DONE? CHECK!"
        if bg == image.COLOR_WHITE or bg == image.COLOR_YELLOW:
            tcol = image.COLOR_BLACK
        else:
            tcol = image.COLOR_WHITE
        frame.draw_string(max(4, W // 2 - len(txt) * 8), H // 2 - 16, txt, tcol, scale=2)
        gui.show(frame)
        time.sleep_ms(250)
    aio.led_off()


def find_leftover_results(cam, cfg, used_ids):
    """二次复查: 重新检测碎片区, 找出尚未使用模板的碎片(漏抓补捡)"""
    frame = cam.read()
    if LENS_STRENGTH > 0:
        frame = frame.lens_corr(strength=LENS_STRENGTH)
    pieces, _, _, _ = vision.detect_pieces(frame, cfg)
    extra = []
    used = set(used_ids)
    for p in pieces:
        pid, score = solver.identify_piece(p, cfg)
        if pid is None or pid in used:
            continue
        used.add(pid)
        dth = solver.compute_delta_theta(p, pid, cfg)
        t = cfg["templates"][pid]
        extra.append({
            "id": pid,
            "from_px": p["centroid"],
            "target_offset_mm": t["target_offset_mm"],
            "theta": dth,
            "score": score,
            "vertices": p["vertices"],
        })
    return extra

def draw_run_overlay(frame):
    """运行叠加层: 目标拼图(黄框+十字+红色目标轮廓+软连接线)+目标ID
    纯绘制不检测: 不影响帧率, 机械臂入镜也不会误识别"""
    ctx = run_ctx
    if ctx is None or not ctx.get("targets"):
        return
    cx0, cy0 = ctx.get("cx", 0), ctx.get("cy", 0)
    px_per_mm = cfg["calibration"]["px_per_mm"]
    tw_px = int(cfg["display"]["target_rect_mm"][0] * px_per_mm)
    th_px = int(cfg["display"]["target_rect_mm"][1] * px_per_mm)
    frame.draw_rect(cx0 - tw_px // 2, cy0 - th_px // 2, tw_px, th_px, image.COLOR_YELLOW, 1)
    frame.draw_cross(cx0, cy0, image.COLOR_YELLOW, size=6, thickness=1)
    for pid, fpx, tverts, tcenter in ctx["targets"]:
        frame.draw_line(int(fpx[0]), int(fpx[1]), tcenter[0], tcenter[1], image.COLOR_RED, 2)
        ntv = len(tverts)
        for i in range(ntv):
            x1, y1 = tverts[i]
            x2, y2 = tverts[(i + 1) % ntv]
            frame.draw_line(x1, y1, x2, y2, image.COLOR_RED, 2)
        s = image.string_size(pid, scale=1.5)
        frame.draw_rect(tcenter[0] - s.width() // 2 - 2, tcenter[1] - 27, s.width() + 4, s.height() + 2, image.COLOR_BLACK, -1)
        frame.draw_string(tcenter[0] - s.width() // 2, tcenter[1] - 26, pid, image.COLOR_YELLOW, scale=1.5)


def make_tick_cb(title, scale=2, bar_h=34, overlay=False):
    """发送滴灌回调: 发送期间刷新屏幕+查deadline, 返回False中止发送"""
    def cb(bi, nb):
        if app.need_exit():
            return False
        if deadline_hit(t0_ms, cfg):
            return False
        frame = cam.read()
        if LENS_STRENGTH > 0:
            frame = frame.lens_corr(strength=LENS_STRENGTH)
        if overlay:
            draw_run_overlay(frame)
        frame.draw_rect(0, 0, W, bar_h, image.COLOR_BLACK, -1)
        frame.draw_string(6, bar_h // 2 - 8, "%s blk:%d/%d T:%d/%ds" % (
            title, bi, nb, elapsed_s(t0_ms), cfg["alert"]["deadline_s"]),
            image.COLOR_YELLOW, scale=scale)
        gui.show(frame)
        return True
    return cb

def app_exit():
    """退出程序(调试重启用): 所有参数改动均已即时保存, 直接退出不丢数据"""
    try:
        app.set_need_exit()
    except Exception:
        try:
            import sys
            sys.exit()
        except Exception:
            pass


# ==================== 主循环 ====================
state = "menu"
frame_count = 0
quick_count = 0

while not app.need_exit():
    time.fps_start()
    img = cam.read()
    if LENS_STRENGTH > 0:
        img = img.lens_corr(strength=LENS_STRENGTH)

    frame_count += 1

    # ==================== 菜单 ====================
    if state == "menu":
        if frame_count % 30 == 1:
            quick_count = vision.count_contours_quick(img, cfg)

        gui.draw_btn(img, 170, 80, 300, 60, "Start")
        gui.draw_btn(img, 170, 145, 300, 60, "Debug")
        gui.draw_btn(img, 170, 210, 300, 60, "Calibrate")
        gui.draw_btn(img, 170, 275, 300, 60, "Light")
        gui.draw_btn(img, 170, 340, 300, 60, "Home")
        gui.draw_btn(img, 480, 15, 150, 45, "TEST")
        gui.draw_btn(img, 5, 15, 150, 45, "EXIT")
        color = image.COLOR_GREEN if quick_count >= 4 else image.COLOR_RED
        img.draw_string(10, 420, "N:%d FPS:%d" % (quick_count, int(time.fps())), color)
        gui.show(img)

        ix, iy, clicked = gui.get_click()
        if clicked:
            if gui.hit(ix, iy, 170, 80, 300, 60):
                # 新流程: 先遮挡打乱, 揭开自动跑 (config start.wait_uncover)
                arm_cnt = 0
                settle_cnt = 0
                if cfg["start"].get("wait_uncover", True):
                    state = "wait_uncover"
                else:
                    t0_ms = time.ticks_ms()
                    run_ctx = {"phase": "detect", "retry": 0, "used": set(),
                               "cx": 0, "cy": 0, "ok": True, "pass2": False}
                    state = "running"
            elif gui.hit(ix, iy, 170, 145, 300, 60):
                state = "debug"
            elif gui.hit(ix, iy, 170, 210, 300, 60):
                state = "calibrate"
                calib_msg = ""
                px_manual = False
            elif gui.hit(ix, iy, 170, 275, 300, 60):
                state = "light"
            elif gui.hit(ix, iy, 170, 340, 300, 60):
                utils.send_init()
            elif gui.hit(ix, iy, 480, 15, 150, 45):
                # 声光自检: 比赛前验证LED+喇叭
                run_alert(gui, cam, cfg, aio, True)
            elif gui.hit(ix, iy, 5, 15, 150, 45):
                # 退出程序(调试重启用): 改过的参数都已即时保存, 直接退出不丢数据
                app_exit()

    # ==================== 调试 ====================
    elif state == "debug":
        img_cv = image.image2cv(img, copy=False)
        pieces, binary, otsu_val, roi_info = vision.detect_pieces(img, cfg)
        center_gray = vision.sample_center_gray(img_cv)
        b = cfg["binary"]
        r = cfg["roi"]
        pp = cfg["preprocess"]

        if show_binary:
            fullbin = full_frame_binary(binary, roi_info, W, H)
            bg = cv2.cvtColor(fullbin, cv2.COLOR_GRAY2BGR)
            show = image.cv2image(bg, copy=False)
        else:
            show = img

        draw_roi_overlay(show, roi_info)

        for p in pieces:
            v = p["vertices"]
            n = len(v)
            for i in range(n):
                x1, y1 = int(v[i][0]), int(v[i][1])
                x2, y2 = int(v[(i+1) % n][0]), int(v[(i+1) % n][1])
                show.draw_line(x1, y1, x2, y2, image.COLOR_GREEN, 1)
            cx, cy = int(p["centroid"][0]), int(p["centroid"][1])
            show.draw_circle(cx, cy, 3, image.COLOR_RED, -1)

        show.draw_cross(W // 2, H // 2, image.COLOR_YELLOW, size=8, thickness=2)
        pol = r.get("paper_polarity", "dark")
        show.draw_rect(0, 0, W, 26, image.COLOR_BLACK, -1)
        show.draw_string(70, 3, "G:%d O:%d T:%d P:%s M:%s" % (center_gray, otsu_val, b["fixed_threshold"], pol, b["mode"]), image.COLOR_RED, scale=1.5)

        gui.draw_btn(show, 2,  28, 125, 50, "Otsu",  b["mode"] == "otsu")
        gui.draw_btn(show, 2,  82, 125, 50, "Fixed", b["mode"] == "fixed")
        gui.draw_btn(show, 2, 136, 125, 50, "Adapt", b["mode"] == "adaptive")
        gui.draw_btn(show, 2, 190, 125, 50, "HSV",   b["mode"] == "hsv")
        gui.draw_btn(show, 2, 244, 125, 50, "QS")
        gui.draw_btn(show, 2, 385, 125, 55, "Back")
        gui.draw_btn(show, 513,  28, 125, 50, "Bin",  show_binary)
        gui.draw_btn(show, 513,  82, 125, 50, "Inv",  b["invert"])
        gui.draw_btn(show, 513, 136, 125, 50, "CLA",  pp.get("clahe_enabled", False))
        gui.draw_btn(show, 513, 190, 125, 50, "P:%s" % pol)
        gui.draw_btn(show, 513, 244, 125, 50, "Sure?" if save_confirm else "Save")
        gui.draw_btn(show, 245, 385, 70, 55, "T-")
        gui.draw_btn(show, 325, 385, 70, 55, "T+")
        gui.show(show)

        ix, iy, clicked = gui.get_click()
        if clicked:
            if gui.hit(ix, iy, 2, 28, 125, 50):
                b["mode"] = "otsu"
            elif gui.hit(ix, iy, 2, 82, 125, 50):
                b["mode"] = "fixed"
            elif gui.hit(ix, iy, 2, 136, 125, 50):
                b["mode"] = "adaptive"
            elif gui.hit(ix, iy, 2, 190, 125, 50):
                b["mode"] = "hsv"
            elif gui.hit(ix, iy, 2, 244, 125, 50):
                b["mode"] = "fixed"
                b["fixed_threshold"] = max(30, min(220, center_gray - 20))
            elif gui.hit(ix, iy, 2, 385, 125, 55):
                state = "menu"
            elif gui.hit(ix, iy, 513, 28, 125, 50):
                show_binary = not show_binary
            elif gui.hit(ix, iy, 513, 82, 125, 50):
                b["invert"] = not b["invert"]
            elif gui.hit(ix, iy, 513, 136, 125, 50):
                pp["clahe_enabled"] = not pp.get("clahe_enabled", False)
            elif gui.hit(ix, iy, 513, 190, 125, 50):
                order = ["dark", "light", "auto"]
                cur = r.get("paper_polarity", "dark")
                r["paper_polarity"] = order[(order.index(cur) + 1) % 3]
            elif gui.hit(ix, iy, 513, 244, 125, 50):
                if save_confirm:
                    save_config()
                    save_confirm = False
                else:
                    save_confirm = True
            elif gui.hit(ix, iy, 245, 385, 70, 55):
                b["mode"] = "fixed"
                b["fixed_threshold"] = max(0, b["fixed_threshold"] - 5)
            elif gui.hit(ix, iy, 325, 385, 70, 55):
                b["mode"] = "fixed"
                b["fixed_threshold"] = min(255, b["fixed_threshold"] + 5)

    # ==================== 标定 Step2: px_per_mm + 模板 + 手动微调(手动优先) ====================
    elif state == "calibrate":
        import calib  # 延迟导入, 减少启动内存
        SQ_COL = getattr(image, "COLOR_CYAN", getattr(image, "COLOR_BLUE", image.COLOR_GREEN))
        px_mm = cfg["calibration"]["px_per_mm"]

        # 分区检测: 方块放左半区/拼好的4片放右半区, 两区都能检测到
        pieces, roi_info = vision.detect_pieces_both_regions(img, cfg)

        show = img
        draw_roi_overlay(show, roi_info)

        sq, sq_px_mm = calib.detect_calib_square(pieces)
        pieces_no_sq = [p for p in pieces if sq is None or id(p) != id(sq)]
        mapping = calib.map_pieces_to_templates(pieces_no_sq)
        piece2tpl = {}
        if mapping is not None:
            for tid, pc in mapping.items():
                piece2tpl[id(pc)] = tid

        for p in pieces:
            v = p["vertices"]
            n = len(v)
            is_sq = (sq is not None and id(p) == id(sq))
            col = SQ_COL if is_sq else image.COLOR_GREEN
            for i in range(n):
                x1, y1 = int(v[i][0]), int(v[i][1])
                x2, y2 = int(v[(i+1) % n][0]), int(v[(i+1) % n][1])
                show.draw_line(x1, y1, x2, y2, col, 2)
            cx, cy = int(p["centroid"][0]), int(p["centroid"][1])
            show.draw_circle(cx, cy, 3, image.COLOR_RED, -1)
            if is_sq:
                rect = cv2.minAreaRect(p["contour"])
                side_px = int(round((rect[1][0] + rect[1][1]) / 2))
                lab = "30mm=%dpx" % side_px
                s = image.string_size(lab, scale=1.5)
                lx, ly = cx - s.width() // 2, cy - 28
                show.draw_rect(lx - 2, ly - 1, s.width() + 4, s.height() + 2, image.COLOR_BLACK, -1)
                show.draw_string(lx, ly, lab, SQ_COL, scale=1.5)
            else:
                # 右半区碎片: 模板ID标签 + 每条边的mm数值(纯数字无单位, 黑底黄字高辨识度)
                tid = piece2tpl.get(id(p), "?")
                lab = "%s %dv" % (tid, p["n_vertices"])
                s = image.string_size(lab, scale=1.5)
                lx, ly = cx - s.width() // 2, cy - s.height() - 8
                show.draw_rect(lx - 2, ly - 1, s.width() + 4, s.height() + 2, image.COLOR_BLACK, -1)
                show.draw_string(lx, ly, lab, image.COLOR_YELLOW, scale=1.5)
                for i in range(n):
                    x1e, y1e = float(v[i][0]), float(v[i][1])
                    x2e, y2e = float(v[(i + 1) % n][0]), float(v[(i + 1) % n][1])
                    et = "%.1f" % (p["edge_lengths"][i] / px_mm)
                    es = image.string_size(et, scale=1.5)
                    mx = int((x1e + x2e) / 2) - es.width() // 2
                    my = int((y1e + y2e) / 2) - es.height() // 2
                    show.draw_rect(mx - 2, my - 1, es.width() + 4, es.height() + 2, image.COLOR_BLACK, -1)
                    show.draw_string(mx, my, et, image.COLOR_YELLOW, scale=1.5)

        # 顶部状态条: 黑底大字, 不再和纸框重叠
        show.draw_rect(0, 0, W, 32, image.COLOR_BLACK, -1)
        sq_side_now = None
        if sq is not None:
            rect = cv2.minAreaRect(sq["contour"])
            sq_side_now = (rect[1][0] + rect[1][1]) / 2.0
        if sq is not None and sq_px_mm is not None:
            show.draw_string(5, 5, "SQ %dpx->%.2fpx/mm  N=%d" % (int(round(sq_side_now)), sq_px_mm, len(pieces)), image.COLOR_YELLOW, scale=2)
        else:
            show.draw_string(5, 5, "NO SQUARE  px/mm=%.2f  N=%d" % (px_mm, len(pieces)), image.COLOR_RED, scale=2)

        # SAVE 结果横幅: 黑底彩字大字 (含JUMP跳变拒绝/手动模式提示)
        if calib_msg:
            if "OK" in calib_msg:
                mcol = image.COLOR_GREEN
            elif "FAIL" in calib_msg or "no square" in calib_msg or "JUMP" in calib_msg or "NOT" in calib_msg:
                mcol = image.COLOR_RED
            else:
                mcol = image.COLOR_YELLOW
            show.draw_rect(0, 34, W, 32, image.COLOR_BLACK, -1)
            show.draw_string(5, 39, calib_msg, mcol, scale=2)

        gui.draw_btn(show, 2, 72, 140, 60, "SAVE")
        gui.draw_btn(show, 2, 140, 140, 60, "Offset")
        gui.draw_btn(show, 2, 208, 140, 55, "P-01")
        gui.draw_btn(show, 2, 268, 140, 55, "P+01")
        # 人眼核对区: 方块边长折算mm(应≈30) + MANUAL标记(手动优先中, SAVE不会覆盖)
        if sq_side_now is not None:
            show.draw_string(6, 335, "sq=%.1fpx->%.1fmm%s" % (sq_side_now, sq_side_now / px_mm, " MANUAL" if px_manual else ""), SQ_COL, scale=1.5)
        else:
            show.draw_string(6, 335, "CFG px/mm=%.2f%s" % (px_mm, " MANUAL" if px_manual else ""), image.COLOR_WHITE, scale=1.5)
        gui.draw_btn(show, 2, 385, 140, 55, "Back")
        gui.show(show)

        ix, iy, clicked = gui.get_click()
        if clicked:
            if gui.hit(ix, iy, 2, 72, 140, 60):
                sq, new_px_mm = calib.detect_calib_square(pieces)
                pieces_no_sq = [p for p in pieces if sq is None or id(p) != id(sq)]
                mapping = calib.map_pieces_to_templates(pieces_no_sq)
                old_px_mm = cfg["calibration"]["px_per_mm"]

                if px_manual:
                    # 手动优先: 已用P±01微调过, SAVE只用手动值刷新模板, 绝不覆盖px/mm
                    if mapping is not None:
                        calib.apply_to_cfg(cfg, mapping, old_px_mm)
                        utils.reload_cfg(cfg)
                        save_config()
                        calib_msg = "OK tpl saved px/mm=%.2f MANUAL" % old_px_mm
                    else:
                        calib_msg = "MANUAL: need 4 pcs for tpl"
                elif new_px_mm is not None and abs(new_px_mm - old_px_mm) / old_px_mm > 0.40:
                    # 跳变保护: 方块破损/误检时px/mm会异常, 与当前值差>40%拒绝保存
                    calib_msg = "JUMP %.2f->%.2f NOT saved!" % (old_px_mm, new_px_mm)
                elif new_px_mm is not None and mapping is not None:
                    calib.apply_to_cfg(cfg, mapping, new_px_mm)
                    utils.reload_cfg(cfg)
                    save_config()
                    px_manual = False
                    calib_msg = "OK px/mm=%.2f +4tpl saved" % new_px_mm
                elif new_px_mm is not None:
                    cfg["calibration"]["px_per_mm"] = round(new_px_mm, 4)
                    utils.reload_cfg(cfg)
                    save_config()
                    px_manual = False
                    calib_msg = "px/mm=%.2f saved, tpl FAIL" % new_px_mm
                elif mapping is not None:
                    calib_msg = "no square! nothing saved"
                else:
                    calib_msg = "FAIL: need 30mm sq + 4 pcs"

            elif gui.hit(ix, iy, 2, 140, 140, 60):
                state = "calib_offset"
            elif gui.hit(ix, iy, 2, 208, 140, 55):
                # px/mm手动微调 -0.01: 手动优先, 之后SAVE只刷模板不覆盖此值
                px_manual = True
                pxv = max(0.5, min(5.0, round(cfg["calibration"]["px_per_mm"] - 0.01, 4)))
                cfg["calibration"]["px_per_mm"] = pxv
                utils.reload_cfg(cfg)
                save_config()
                calib_msg = "px/mm=%.2f MANUAL saved" % pxv
            elif gui.hit(ix, iy, 2, 268, 140, 55):
                # px/mm手动微调 +0.01
                px_manual = True
                pxv = max(0.5, min(5.0, round(cfg["calibration"]["px_per_mm"] + 0.01, 4)))
                cfg["calibration"]["px_per_mm"] = pxv
                utils.reload_cfg(cfg)
                save_config()
                calib_msg = "px/mm=%.2f MANUAL saved" % pxv
            elif gui.hit(ix, iy, 2, 385, 140, 55):
                state = "menu"
    # ==================== 标定 Step3: Robot Offset 微调 + 速度 + 发送节奏 + PNT ====================
    elif state == "calib_offset":
        off = cfg["calibration"]["robot_offset"]
        ox, oy = off[0], off[1]
        xs = cfg["calibration"].get("x_sign", 1)
        ys = cfg["calibration"].get("y_sign", 1)
        fr = cfg["mechanical"]["feed_rate"]
        blk = cfg["serial"]["block_delay_ms"]
        gap = cfg["start"]["piece_interval_s"]

        img.draw_rect(0, 0, W, H, image.COLOR_BLACK, -1)
        img.draw_string(180, 8, "ROBOT OFFSET CALIB", image.COLOR_YELLOW, scale=2)
        img.draw_string(120, 36, "!! PRESS HOME FIRST !!", image.COLOR_RED, scale=1.5)

        img.draw_string(30, 58, "X: %.1f mm (%s)" % (ox, "+" if xs > 0 else "-"), image.COLOR_GREEN, scale=2)
        img.draw_string(30, 118, "Y: %.1f mm (%s)" % (oy, "+" if ys > 0 else "-"), image.COLOR_GREEN, scale=2)

        bw, bh = 92, 48
        bx0 = 250
        by_x = 50
        by_y = 110

        gui.draw_btn(img, bx0,       by_x, bw, bh, "<<")
        gui.draw_btn(img, bx0+98,   by_x, bw, bh, "<")
        gui.draw_btn(img, bx0+196,   by_x, bw, bh, ">")
        gui.draw_btn(img, bx0+294,   by_x, bw, bh, ">>")

        gui.draw_btn(img, bx0,       by_y, bw, bh, "<<")
        gui.draw_btn(img, bx0+98,   by_y, bw, bh, "<")
        gui.draw_btn(img, bx0+196,   by_y, bw, bh, ">")
        gui.draw_btn(img, bx0+294,   by_y, bw, bh, ">>")

        gui.draw_btn(img, 50,  170, 160, 45, "MOVE")
        gui.draw_btn(img, 240, 170, 160, 45, "SAVE")
        gui.draw_btn(img, 430, 170, 160, 45, "BACK")

        img.draw_string(30, 233, "F: %d" % fr, image.COLOR_GREEN, scale=2)
        gui.draw_btn(img, 250, 225, 120, 45, "F-500")
        gui.draw_btn(img, 390, 225, 120, 45, "F+500")
        gui.draw_btn(img, 535, 225, 100, 45, "X+-")

        img.draw_string(30, 288, "BLK: %dms" % blk, image.COLOR_GREEN, scale=2)
        gui.draw_btn(img, 250, 280, 90, 45, "B-100")
        gui.draw_btn(img, 350, 280, 90, 45, "B+100")
        gui.draw_btn(img, 535, 280, 100, 45, "Y+-")

        img.draw_string(30, 343, "GAP: %ds" % gap, image.COLOR_GREEN, scale=2)
        gui.draw_btn(img, 250, 335, 90, 45, "G-1")
        gui.draw_btn(img, 350, 335, 90, 45, "G+1")
        gui.draw_btn(img, 455, 335, 130, 45, "PNT")

        img.draw_string(30, 392, "<<>>=5mm X+-Y+-=flip sign PNT=5pt", image.COLOR_WHITE, scale=1.5)

        if calib_msg:
            img.draw_string(30, 420, calib_msg, image.COLOR_GREEN, scale=1.5)

        gui.show(img)

        ix, iy, clicked = gui.get_click()
        if clicked:
            if gui.hit(ix, iy, bx0, by_x, bw, bh):
                off[0] = round(off[0] - 5, 1)
            elif gui.hit(ix, iy, bx0+98, by_x, bw, bh):
                off[0] = round(off[0] - 1, 1)
            elif gui.hit(ix, iy, bx0+196, by_x, bw, bh):
                off[0] = round(off[0] + 1, 1)
            elif gui.hit(ix, iy, bx0+294, by_x, bw, bh):
                off[0] = round(off[0] + 5, 1)
            elif gui.hit(ix, iy, bx0, by_y, bw, bh):
                off[1] = round(off[1] - 5, 1)
            elif gui.hit(ix, iy, bx0+98, by_y, bw, bh):
                off[1] = round(off[1] - 1, 1)
            elif gui.hit(ix, iy, bx0+196, by_y, bw, bh):
                off[1] = round(off[1] + 1, 1)
            elif gui.hit(ix, iy, bx0+294, by_y, bw, bh):
                off[1] = round(off[1] + 5, 1)
            elif gui.hit(ix, iy, 50, 170, 160, 45):
                utils.reload_cfg(cfg)
                utils.send_calib_move()
                calib_msg = "sent G1 X%.1f Y%.1f" % (off[0], off[1])
            elif gui.hit(ix, iy, 240, 170, 160, 45):
                save_config()
                utils.reload_cfg(cfg)
                calib_msg = "offset saved!"
            elif gui.hit(ix, iy, 430, 170, 160, 45):
                state = "calibrate"
                calib_msg = ""
            elif gui.hit(ix, iy, 250, 225, 120, 45):
                fr = max(1000, fr - 500)
                cfg["mechanical"]["feed_rate"] = fr
                utils.reload_cfg(cfg)
                save_config()
                calib_msg = "F=%d saved" % fr
            elif gui.hit(ix, iy, 390, 225, 120, 45):
                fr = min(6000, fr + 500)
                cfg["mechanical"]["feed_rate"] = fr
                utils.reload_cfg(cfg)
                save_config()
                calib_msg = "F=%d saved" % fr
            elif gui.hit(ix, iy, 535, 225, 100, 45):
                # X符号翻转: PNT上下两点镜像偏移时按 (相机v方向 vs 机械x方向)
                cfg["calibration"]["x_sign"] = -xs
                utils.reload_cfg(cfg)
                save_config()
                calib_msg = "X sign -> %s saved" % ("+" if -xs > 0 else "-")
            elif gui.hit(ix, iy, 250, 280, 90, 45):
                blk = max(100, blk - 100)
                cfg["serial"]["block_delay_ms"] = blk
                utils.reload_cfg(cfg)
                save_config()
                calib_msg = "BLK=%dms saved" % blk
            elif gui.hit(ix, iy, 350, 280, 90, 45):
                blk = min(3000, blk + 100)
                cfg["serial"]["block_delay_ms"] = blk
                utils.reload_cfg(cfg)
                save_config()
                calib_msg = "BLK=%dms saved" % blk
            elif gui.hit(ix, iy, 535, 280, 100, 45):
                # Y符号翻转: PNT左右两点镜像偏移时按 (相机u方向 vs 机械y方向)
                cfg["calibration"]["y_sign"] = -ys
                utils.reload_cfg(cfg)
                save_config()
                calib_msg = "Y sign -> %s saved" % ("+" if -ys > 0 else "-")
            elif gui.hit(ix, iy, 250, 335, 90, 45):
                gap = max(0, gap - 1)
                cfg["start"]["piece_interval_s"] = gap
                save_config()
                calib_msg = "GAP=%ds saved" % gap
            elif gui.hit(ix, iy, 350, 335, 90, 45):
                gap = min(30, gap + 1)
                cfg["start"]["piece_interval_s"] = gap
                save_config()
                calib_msg = "GAP=%ds saved" % gap
            elif gui.hit(ix, iy, 455, 335, 130, 45):
                # PNT五点校验: 每按切换下一个像素点, 臂移到对应机械位(先按HOME!)
                # 5点都压中红十字 = 像素->机械变换全场正确(含符号/比例/原点)
                pnt_idx = (pnt_idx + 1) % 5
                pts = [(W // 2, H // 2), (W // 4, H // 2), (W * 3 // 4, H // 2),
                       (W // 2, H // 4), (W // 2, H * 3 // 4)]
                pu, pv = pts[pnt_idx]
                img.draw_cross(pu, pv, image.COLOR_RED, size=15, thickness=2)
                img.draw_circle(pu, pv, 25, image.COLOR_RED, 2)
                img.draw_string(130, 8, "ARM->CROSS %d/5" % (pnt_idx + 1), image.COLOR_YELLOW, scale=2)
                gui.show(img)
                utils.reload_cfg(cfg)

                def pnt_tick(bi, nb):
                    frame = cam.read()
                    if LENS_STRENGTH > 0:
                        frame = frame.lens_corr(strength=LENS_STRENGTH)
                    frame.draw_cross(pu, pv, image.COLOR_RED, size=15, thickness=2)
                    frame.draw_circle(pu, pv, 25, image.COLOR_RED, 2)
                    frame.draw_string(130, 8, "ARM->CROSS %d/5" % (pnt_idx + 1), image.COLOR_YELLOW, scale=2)
                    gui.show(frame)
                    return True

                utils.send_calib_point(pu, pv, pnt_tick)
                calib_msg = "PNT %d/5 (%d,%d)" % (pnt_idx + 1, pu, pv)
    # ==================== 调光页: 暗场曝光/增益/CLAHE ====================
    # ==================== 调光页: 暗场曝光/增益/CLAHE ====================
    # ==================== 调光页: 暗场曝光/增益/CLAHE ====================
    elif state == "light":
        cs = cfg["camera"]
        pp = cfg["preprocess"]
        exp = cs.get("exposure", 0)
        gn = cs.get("gain", 0)

        img.draw_string(180, 5, "LIGHT CALIB", image.COLOR_YELLOW, scale=2)
        img.draw_string(10, 38, "exp=%s gain=%s CLA=%s" % (exp, gn, "ON" if pp.get("clahe_enabled") else "off"), image.COLOR_GREEN, scale=1.5)

        bw, bh = 100, 55
        img.draw_string(10, 78, "EXP", image.COLOR_WHITE, scale=2)
        gui.draw_btn(img, 90,  65, bw, bh, "<<")
        gui.draw_btn(img, 200, 65, bw, bh, "<")
        gui.draw_btn(img, 310, 65, bw, bh, ">")
        gui.draw_btn(img, 420, 65, bw, bh, ">>")
        img.draw_string(10, 148, "GAIN", image.COLOR_WHITE, scale=2)
        gui.draw_btn(img, 90,  135, bw, bh, "<<")
        gui.draw_btn(img, 200, 135, bw, bh, "<")
        gui.draw_btn(img, 310, 135, bw, bh, ">")
        gui.draw_btn(img, 420, 135, bw, bh, ">>")
        gui.draw_btn(img, 90, 205, 200, 55, "CLAHE:%s" % ("ON" if pp.get("clahe_enabled") else "off"), pp.get("clahe_enabled", False))
        gui.draw_btn(img, 320, 205, 200, 55, "AutoExp")
        gui.draw_btn(img, 90, 270, 200, 55, "SAVE")
        gui.draw_btn(img, 320, 270, 200, 55, "BACK")
        img.draw_string(10, 342, "LENS:%.1f" % cs.get("lens_corr", 0), image.COLOR_GREEN, scale=2)
        gui.draw_btn(img, 150, 335, 100, 50, "L-1")
        gui.draw_btn(img, 260, 335, 100, 50, "L+1")
        img.draw_string(10, 395, "dark? raise EXP, then GAIN", image.COLOR_WHITE, scale=1.5)
        img.draw_string(10, 420, "low contrast? CLAHE / L+- = lens corr", image.COLOR_WHITE, scale=1.5)
        gui.show(img)

        ix, iy, clicked = gui.get_click()
        if clicked:
            if gui.hit(ix, iy, 90, 65, bw, bh):
                cs["exposure"] = max(0, exp - 500)
            elif gui.hit(ix, iy, 200, 65, bw, bh):
                cs["exposure"] = max(0, exp - 100)
            elif gui.hit(ix, iy, 310, 65, bw, bh):
                cs["exposure"] = exp + 100
            elif gui.hit(ix, iy, 420, 65, bw, bh):
                cs["exposure"] = exp + 500
            elif gui.hit(ix, iy, 90, 135, bw, bh):
                cs["gain"] = max(0, gn - 20)
            elif gui.hit(ix, iy, 200, 135, bw, bh):
                cs["gain"] = max(0, gn - 5)
            elif gui.hit(ix, iy, 310, 135, bw, bh):
                cs["gain"] = gn + 5
            elif gui.hit(ix, iy, 420, 135, bw, bh):
                cs["gain"] = gn + 20
            elif gui.hit(ix, iy, 90, 205, 200, 55):
                pp["clahe_enabled"] = not pp.get("clahe_enabled", False)
            elif gui.hit(ix, iy, 320, 205, 200, 55):
                cs["exposure"] = 0
                cs["gain"] = 0
                try:
                    cam.exp_mode(camera.AeMode.Auto)
                except Exception:
                    pass
            elif gui.hit(ix, iy, 90, 270, 200, 55):
                save_config()
                apply_camera_settings(cam, cfg)
            elif gui.hit(ix, iy, 320, 270, 200, 55):
                state = "menu"
            elif gui.hit(ix, iy, 150, 335, 100, 50):
                # 畸变矫正强度-0.1: PNT边缘点放射状渐偏时微调此值
                cs["lens_corr"] = max(0.0, round(cs.get("lens_corr", 0) - 0.1, 2))
                LENS_STRENGTH = cs["lens_corr"]
                save_config()
            elif gui.hit(ix, iy, 260, 335, 100, 50):
                cs["lens_corr"] = min(3.0, round(cs.get("lens_corr", 0) + 0.1, 2))
                LENS_STRENGTH = cs["lens_corr"]
                save_config()
            apply_camera_settings(cam, cfg)

    # ==================== 等待揭盖: 先按Start->遮挡->打乱->揭开自动触发 ====================
    elif state == "wait_uncover":
        pieces, binary, otsu_val, roi_info = vision.detect_pieces(img, cfg)
        n = len(pieces)
        st = cfg["start"]
        # 连续K帧检出碎片才认定已揭开(遮挡时纯黑图/手晃过的模糊帧无法连续通过)
        if n >= st["min_pieces"]:
            arm_cnt += 1
        else:
            arm_cnt = 0

        draw_roi_overlay(img, roi_info)
        for p in pieces:
            cx, cy = int(p["centroid"][0]), int(p["centroid"][1])
            img.draw_circle(cx, cy, 4, image.COLOR_RED, -1)
        img.draw_rect(0, 0, W, 30, image.COLOR_BLACK, -1)
        img.draw_string(6, 4, "WAIT UNCOVER N:%d arm:%d/%d" % (n, arm_cnt, st["arm_frames"]),
                        image.COLOR_YELLOW, scale=1.5)
        img.draw_string(6, H - 24, "tap to cancel", image.COLOR_WHITE, scale=1)
        gui.show(img)

        ix, iy, clicked = gui.get_click()
        if clicked:
            arm_cnt = 0
            state = "menu"
            continue

        if arm_cnt >= st["arm_frames"]:
            settle_cnt = 0
            state = "settle"

    # ==================== 揭开后曝光稳定倒计时 ====================
    elif state == "settle":
        settle_cnt += 1
        st = cfg["start"]
        img.draw_rect(0, 0, W, H, image.COLOR_BLACK, -1)
        img.draw_string(W // 2 - 60, H // 2 - 40, "START...", image.COLOR_GREEN, scale=3)
        bar_w = int((W - 20) * settle_cnt / st["settle_frames"])
        img.draw_rect(10, H - 30, bar_w, 20, image.COLOR_GREEN, -1)
        gui.show(img)
        if settle_cnt >= st["settle_frames"]:
            # 揭开瞬间=计时原点t0 (正式2分钟从移开遮挡开始)
            t0_ms = time.ticks_ms()
            run_ctx = {"phase": "detect", "retry": 0, "used": set(),
                       "cx": 0, "cy": 0, "ok": True, "pass2": False}
            state = "running"

    # ==================== 运行: 慢速逐片发送 + 二次复查 + 声光提示 ====================
    elif state == "running":
        st = cfg["start"]
        al = cfg["alert"]
        ctx = run_ctx
        phase = ctx["phase"]

        # ---------- detect: 检测+匹配, 0匹配可重试 ----------
        if phase == "detect":
            pieces, binary, otsu_val, roi_info = vision.detect_pieces(img, cfg)
            results = solver.solve_all(pieces, cfg)

            if not results:
                ctx["retry"] += 1
                if ctx["retry"] < st["retry_detects"]:
                    img.draw_rect(0, 0, W, 30, image.COLOR_BLACK, -1)
                    img.draw_string(6, 4, "RETRY %d/%d" % (ctx["retry"], st["retry_detects"]),
                                    image.COLOR_YELLOW, scale=1.5)
                    gui.show(img)
                    time.sleep_ms(800)
                    continue
                img.draw_rect(0, 0, W, H, image.COLOR_BLACK, -1)
                img.draw_string(150, 180, "NO PIECE MATCHED", image.COLOR_RED, scale=2)
                img.draw_string(150, 240, "tap to menu", image.COLOR_WHITE, scale=1)
                gui.show(img)
                for _ in range(30):
                    time.sleep_ms(100)
                    ix, iy, clicked = gui.get_click()
                    if clicked or app.need_exit():
                        break
                state = "menu"
                run_ctx = None
                continue

            puzzle_rect = roi_info["puzzle_rect"]
            px_per_mm = cfg["calibration"]["px_per_mm"]
            target_cx, target_cy = 0, 0
            if puzzle_rect is not None:
                ux, uy, uw, uh = puzzle_rect
                target_cx = ux + uw // 2
                target_cy = uy + uh // 2
            ctx["cx"], ctx["cy"] = target_cx, target_cy

            for r in results:
                off_mm = r["target_offset_mm"]
                r["target_px"] = (int(target_cx + off_mm[0] * px_per_mm),
                                  int(target_cy + off_mm[1] * px_per_mm))
            ctx["results"] = results
            ctx["pieces"] = pieces
            ctx["used"] = set(r["id"] for r in results)
            ctx["idx"] = 0
            ctx["phase"] = "preview"
            print("=== matched %d pieces, T=%ds ===" % (len(results), elapsed_s(t0_ms)))
            continue

        # ---------- preview: 发送前预览1.2秒(可视化确认) ----------
        elif phase == "preview":
            results = ctx["results"]
            target_cx, target_cy = ctx["cx"], ctx["cy"]
            px_per_mm = cfg["calibration"]["px_per_mm"]
            tw_px = int(cfg["display"]["target_rect_mm"][0] * px_per_mm)
            th_px = int(cfg["display"]["target_rect_mm"][1] * px_per_mm)
            img.draw_rect(target_cx - tw_px // 2, target_cy - th_px // 2,
                          tw_px, th_px, image.COLOR_YELLOW, 1)
            img.draw_cross(target_cx, target_cy, image.COLOR_YELLOW, size=6, thickness=1)
            # 当前碎片轮廓(绿虚线) + 质心
            for p in ctx.get("pieces", []):
                v = p["vertices"]
                nvp = len(v)
                for i in range(nvp):
                    x1, y1 = int(v[i][0]), int(v[i][1])
                    x2, y2 = int(v[(i + 1) % nvp][0]), int(v[(i + 1) % nvp][1])
                    draw_dashed_line(img, x1, y1, x2, y2, image.COLOR_GREEN, 2)
                pcx, pcy = int(p["centroid"][0]), int(p["centroid"][1])
                img.draw_circle(pcx, pcy, 4, image.COLOR_RED, -1)
            # 目标拼图: 连接线(软连接) + 每片目标轮廓(红), 即拼好后的整体形状
            tg_list = []
            for r in results:
                cx, cy = int(r["from_px"][0]), int(r["from_px"][1])
                pid = r["id"]
                dth = r["theta"]
                img.draw_string(cx - 10, cy - 25, "%s d%d" % (pid, int(dth)),
                                image.COLOR_YELLOW, scale=1.5)
                piece_data = {"centroid": r["from_px"], "vertices": r["vertices"]}
                tverts, tcenter = solver.compute_target_contour(
                    piece_data, pid, dth, (target_cx, target_cy), cfg)
                tg_list.append((pid, r["from_px"], tverts, tcenter))
                img.draw_line(cx, cy, tcenter[0], tcenter[1], image.COLOR_RED, 2)
                ntv = len(tverts)
                for i in range(ntv):
                    x1, y1 = tverts[i]
                    x2, y2 = tverts[(i + 1) % ntv]
                    img.draw_line(x1, y1, x2, y2, image.COLOR_RED, 2)
                img.draw_circle(tcenter[0], tcenter[1], 3, image.COLOR_RED, -1)
            ctx["targets"] = tg_list
            img.draw_rect(0, 0, W, 30, image.COLOR_BLACK, -1)
            img.draw_string(6, 4, "GO! %d pieces  T:%ds" % (len(results), elapsed_s(t0_ms)),
                            image.COLOR_GREEN, scale=1.5)
            gui.show(img)
            time.sleep_ms(1200)
            ctx["phase"] = "send"
            continue

        # ---------- send: 慢速逐片发送(下位机缓冲小, 宁慢勿爆) ----------
        elif phase == "send":
            results = ctx["results"]
            idx = ctx["idx"]
            if deadline_hit(t0_ms, cfg):
                # deadline到: 停止发送, 已发块自行执行完, 直接声光提示
                ctx["ok"] = False
                ctx["phase"] = "alert"
                continue
            if idx < len(results):
                r = results[idx]
                draw_run_overlay(img)
                img.draw_rect(0, 0, W, 30, image.COLOR_BLACK, -1)
                img.draw_string(6, 4, "SEND %s (%d/%d)  T:%ds" % (
                    r["id"], idx + 1, len(results), elapsed_s(t0_ms)),
                    image.COLOR_YELLOW, scale=1.5)
                gui.show(img)
                print("  send %s from=(%d,%d) to=(%d,%d) th=%.1f T=%ds" % (
                    r["id"], int(r["from_px"][0]), int(r["from_px"][1]),
                    r["target_px"][0], r["target_px"][1], r["theta"], elapsed_s(t0_ms)))
                try:
                    sent = utils.send_piece(r, make_tick_cb("SEND %s" % r["id"],
                                                          scale=2, bar_h=34, overlay=True))
                except Exception as e:
                    print("  send fail:", e)
                    sent = False
                if not sent:
                    # 发送被中止(deadline/退出/异常): 不再发送, 直接提示
                    ctx["ok"] = False
                    ctx["phase"] = "alert"
                    continue
                ctx["idx"] += 1
                # 非最后一片: 片间长等待, 让下位机充分消化
                if ctx["idx"] < len(results):
                    okwait = wait_with_screen(st["piece_interval_s"], "%s sent" % r["id"],
                                              t0_ms, cfg, gui, cam, overlay=True)
                    if not okwait:
                        ctx["ok"] = False
                        ctx["phase"] = "alert"
                continue
            # 全部发完 -> 回零
            try:
                utils.finish(make_tick_cb("HOME", scale=2, bar_h=34, overlay=True))
            except Exception as e:
                print("  finish fail:", e)
            ctx["phase"] = "settle1"
            continue

        # ---------- settle1: 等机械臂物理完成(回零停稳) ----------
        elif phase == "settle1":
            okwait = wait_with_screen(al["finish_settle_s"], "ARM HOMING",
                                      t0_ms, cfg, gui, cam, overlay=True)
            if not okwait:
                ctx["ok"] = False
                ctx["phase"] = "alert"
            elif al.get("second_pass", True):
                ctx["phase"] = "check"
            else:
                ctx["phase"] = "alert"
            continue

        # ---------- check: 视觉复查碎片区是否有漏抓 ----------
        elif phase == "check":
            extra = []
            if elapsed_s(t0_ms) < al["deadline_s"] - 25:
                extra = find_leftover_results(cam, cfg, ctx["used"])
            if extra:
                px_per_mm = cfg["calibration"]["px_per_mm"]
                for r in extra:
                    off_mm = r["target_offset_mm"]
                    r["target_px"] = (int(ctx["cx"] + off_mm[0] * px_per_mm),
                                      int(ctx["cy"] + off_mm[1] * px_per_mm))
                ctx["results"] = extra
                ctx["used"] |= set(r["id"] for r in extra)
                ctx["idx"] = 0
                ctx["pass2"] = True
                ctx["phase"] = "send2"
                print("=== pass2: %d leftover pieces ===" % len(extra))
            else:
                print("=== check: no leftover ===")
                ctx["phase"] = "alert"
            continue

        # ---------- send2: 二次补检发送(短间隔, 时间预算有限) ----------
        elif phase == "send2":
            results = ctx["results"]
            idx = ctx["idx"]
            if deadline_hit(t0_ms, cfg):
                ctx["ok"] = False
                ctx["phase"] = "alert"
                continue
            if idx < len(results):
                r = results[idx]
                draw_run_overlay(img)
                img.draw_rect(0, 0, W, 30, image.COLOR_BLACK, -1)
                img.draw_string(6, 4, "PASS2 %s (%d/%d)  T:%ds" % (
                    r["id"], idx + 1, len(results), elapsed_s(t0_ms)),
                    image.COLOR_YELLOW, scale=1.5)
                gui.show(img)
                try:
                    sent = utils.send_piece(r, make_tick_cb("PASS2 %s" % r["id"],
                                                          scale=2, bar_h=34, overlay=True))
                except Exception as e:
                    print("  pass2 send fail:", e)
                    sent = False
                if not sent:
                    ctx["ok"] = False
                    ctx["phase"] = "alert"
                    continue
                ctx["idx"] += 1
                if ctx["idx"] < len(results):
                    wait_with_screen(st["second_pass_interval_s"], "%s sent" % r["id"],
                                     t0_ms, cfg, gui, cam, overlay=True)
                continue
            try:
                utils.finish(make_tick_cb("HOME", scale=2, bar_h=34, overlay=True))
            except Exception:
                pass
            ctx["phase"] = "settle2"
            continue

        # ---------- settle2: 补捡后等停稳 ----------
        elif phase == "settle2":
            wait_with_screen(al.get("second_pass_settle_s", 6), "FINAL HOME",
                             t0_ms, cfg, gui, cam, overlay=True)
            ctx["phase"] = "alert"
            continue

        # ---------- alert: 声光提示(LED+喇叭+屏幕闪) ----------
        elif phase == "alert":
            print("=== ALERT ok=%s T=%ds ===" % (ctx.get("ok", True), elapsed_s(t0_ms)))
            if al.get("enabled", True):
                run_alert(gui, cam, cfg, aio, ctx.get("ok", True))
            ctx["phase"] = "done"
            continue

        # ---------- done/fail: 结果画面, 点击回菜单 ----------
        else:
            img.draw_rect(0, 0, W, H, image.COLOR_BLACK, -1)
            if phase == "done" and ctx.get("ok", True):
                img.draw_string(180, 180, "DONE", image.COLOR_GREEN, scale=2)
            else:
                img.draw_string(180, 180, "ERROR" if phase == "fail" else "DONE?",
                                image.COLOR_RED, scale=2)
            img.draw_string(180, 240, "tap to menu", image.COLOR_WHITE, scale=1)
            gui.show(img)
            for _ in range(50):
                time.sleep_ms(100)
                ix, iy, clicked = gui.get_click()
                if clicked or app.need_exit():
                    break
            state = "menu"
            run_ctx = None
