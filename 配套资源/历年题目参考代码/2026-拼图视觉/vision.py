"""
视觉管线 - 极性自适应版
完整管线: lens_corr -> ROI(找A4纸,支持暗/亮/自动极性) -> 分区 -> 掩膜 -> 预处理 -> 二值化 -> 轮廓 -> 碎片列表

现场适应性:
- roi.paper_polarity: "dark"(黑纸) / "light"(白纸) / "auto"(自动尝试)
- binary.mode: otsu / fixed / hsv(彩色碎片) / adaptive(光照不均)
- binary.invert: 碎片比纸亮=False, 碎片比纸暗=True
"""
import numpy as np
import cv2
from maix import image as maix_image


# ==================== 工具函数 ====================

def align4(val, cfg):
    """向下对齐到4的倍数 (K230 VPSS硬件要求)"""
    a = cfg["roi"]["align"]
    return (val // a) * a


def enhance_gray(gray, cfg):
    """
    可选 CLAHE 局部对比度增强 (暗光/低对比度场景)
    preprocess.clahe_enabled=True 时生效, 同时作用于找纸和碎片检测
    """
    p = cfg["preprocess"]
    if not p.get("clahe_enabled", False):
        return gray
    clip = p.get("clahe_clip", 2.0)
    grid = p.get("clahe_grid", 8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def _roi_from_binary(binary, gray, cfg):
    """
    从已二值化的图(纸=白前景)中定位A4纸。
    5重筛选: 面积/顶点/长宽比/实心度/填充率
    返回: (roi_tuple, binary, angle_deg, contour) 或 (None, binary, 0, None)
    """
    r = cfg["roi"]
    h_img, w_img = gray.shape[:2]

    # 大核闭运算 (填平碎片空洞 + 桥接白线)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (r["morph_kernel"], r["morph_kernel"]))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=r["morph_iter"])

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None, binary, 0.0, None

    candidates = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < r["min_area"] or area > r["max_area"]:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        nv = len(approx)
        if nv < r["min_vertices"] or nv > r["max_vertices"]:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if abs(aspect - r["aspect_ratio"]) > r["aspect_tol"]:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area < 1 or area / hull_area < r["min_solidity"]:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 1 or area / (bw * bh) < r["min_fill_ratio"]:
            continue
        aspect_score = 1.0 - abs(aspect - r["aspect_ratio"]) / r["aspect_tol"]
        solidity = area / hull_area
        score = area * solidity * (0.5 + 0.5 * aspect_score)
        has_children = (hierarchy[0][i][2] != -1)
        candidates.append({
            "contour": cnt, "score": score * (1.2 if has_children else 1.0),
            "bounding": (bx, by, bw, bh), "rect": rect
        })

    if not candidates:
        return None, binary, 0.0, None

    best = max(candidates, key=lambda c: c["score"])
    angle = best["rect"][2]

    bx, by, bw, bh = best["bounding"]
    x = max(0, bx + r["shrink"])
    y = max(0, by + r["shrink"])
    w = align4(min(bw - 2 * r["shrink"], w_img - x), cfg)
    h = align4(min(bh - 2 * r["shrink"], h_img - y), cfg)

    if w < r["min_size"] or h < r["min_size"]:
        return None, binary, angle, None

    return (x, y, w, h), binary, angle, best["contour"]


def find_a4_roi(gray, cfg):
    """
    在灰度图中定位A4纸, 支持极性配置。
    roi.paper_polarity:
      "dark"  : 深色纸(黑A4), 固定阈值找暗区 (默认, 最快)
      "light" : 浅色纸(白A4), Otsu找亮区
      "auto"  : 先试dark, 失败再试light (现场未知时用)
    返回: (roi_tuple, binary, angle_deg, contour)
    """
    r = cfg["roi"]
    polarity = r.get("paper_polarity", "dark")

    if polarity == "light":
        # 浅色纸: Otsu 找亮区 (对未知背景鲁棒)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return _roi_from_binary(binary, gray, cfg)

    # dark 或 auto: 先按深色纸处理
    _, binary = cv2.threshold(gray, r["binary_thresh"], 255, cv2.THRESH_BINARY_INV)
    roi, b, ang, cnt = _roi_from_binary(binary, gray, cfg)
    if roi is not None or polarity == "dark":
        return roi, b, ang, cnt

    # auto 且 dark 失败 -> 试浅色纸 (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi, b, ang, cnt = _roi_from_binary(binary, gray, cfg)
    if roi is not None:
        return roi, b, ang, cnt
    return None, binary, 0.0, None


def compute_split(roi, cfg, img_w, img_h):
    """
    沿长边对半分割, 返回 (piece_rect, puzzle_rect)
    rect = (x, y, w, h) 全图坐标
    """
    s = cfg["split"]
    if roi is not None:
        rx, ry, rw, rh = roi
    else:
        rx, ry, rw, rh = 0, 0, img_w, img_h

    if not s["enabled"]:
        return (rx, ry, rw, rh), None

    gap = s["gap"]
    if rw >= rh:
        half = align4(rw // 2, cfg)
        left_w = align4(half - gap, cfg)
        right_x = rx + half + gap
        right_w = align4(rx + rw - right_x, cfg)
        left_rect = (rx, ry, left_w, rh)
        right_rect = (right_x, ry, right_w, rh)
        if s["piece_side"] == "left":
            return left_rect, right_rect
        else:
            return right_rect, left_rect
    else:
        half = align4(rh // 2, cfg)
        top_h = align4(half - gap, cfg)
        bottom_y = ry + half + gap
        bottom_h = align4(ry + rh - bottom_y, cfg)
        top_rect = (rx, ry, rw, top_h)
        bottom_rect = (rx, bottom_y, rw, bottom_h)
        if s["piece_side"] == "top":
            return top_rect, bottom_rect
        else:
            return bottom_rect, top_rect


# ==================== 核心管线 ====================

def preprocess(gray, cfg):
    """灰度 -> 中值 -> 高斯 -> 闭运算"""
    p = cfg["preprocess"]
    median = cv2.medianBlur(gray, p["smooth_kernel"])
    ks = p["gaussian_kernel"]
    gaussian = cv2.GaussianBlur(median, (ks, ks), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p["morph_kernel_size"], p["morph_kernel_size"]))
    closed = cv2.morphologyEx(gaussian, cv2.MORPH_CLOSE, kernel, iterations=p["morph_close_iter"])
    return closed


def binarize(img_cv, processed, cfg):
    """
    二值化, 返回 (binary, otsu_value)
    mode:
      otsu     : 自动阈值(双峰), 最常用
      fixed    : 固定阈值(光照稳定时手动调)
      hsv      : 彩色碎片(按色相范围)
      adaptive : 自适应阈值(光照不均/阴影)
    invert: False=碎片比纸亮, True=碎片比纸暗
    """
    b = cfg["binary"]
    p = cfg["preprocess"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p["morph_kernel_size"], p["morph_kernel_size"]))
    flag = cv2.THRESH_BINARY_INV if b["invert"] else cv2.THRESH_BINARY
    otsu_value = 0

    if b["mode"] == "otsu":
        otsu_value, binary = cv2.threshold(processed, 0, 255, flag + cv2.THRESH_OTSU)
    elif b["mode"] == "fixed":
        _, binary = cv2.threshold(processed, b["fixed_threshold"], 255, flag)
    elif b["mode"] == "adaptive":
        blk = b.get("adaptive_block", 31)
        if blk % 2 == 0:
            blk += 1
        cval = b.get("adaptive_C", 5)
        binary = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, blk, cval)
    elif b["mode"] == "hsv":
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        lower = np.array(b["hsv_lower"], dtype=np.uint8)
        upper = np.array(b["hsv_upper"], dtype=np.uint8)
        binary = cv2.inRange(hsv, lower, upper)
    else:
        otsu_value, binary = cv2.threshold(processed, 0, 255, flag + cv2.THRESH_OTSU)

    binary = cv2.dilate(binary, kernel, iterations=p["morph_dilate_iter"])
    binary = cv2.erode(binary, kernel, iterations=p["morph_erode_iter"])
    return binary, int(otsu_value)


def find_pieces(binary, cfg, wk_w, wk_h):
    """
    轮廓提取 + 多边形逼近 + 过滤(面积/边界/长宽比)
    返回碎片列表 (坐标为工作区局部坐标)
    """
    c = cfg["contour"]
    m = cfg["matching"]
    f = cfg["filter"]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if f["edge_margin_auto"]:
        edge_margin = max(1, min(wk_w, wk_h) // 200)
    else:
        edge_margin = 1

    pieces = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < c["min_area"]:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        if (bx < edge_margin or by < edge_margin or
            bx + bw > wk_w - edge_margin or by + bh > wk_h - edge_margin):
            continue

        ar_w = max(bw, bh)
        ar_h = min(bw, bh)
        if ar_h > 0 and ar_w / ar_h > f["max_aspect_ratio"]:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, c["epsilon"] * peri, True)
        if len(approx) > c["max_vertices"]:
            for eps in [0.03, 0.04, 0.05, 0.06, 0.08]:
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                if len(approx) <= c["max_vertices"]:
                    break
            else:
                continue

        vertices = approx.reshape(-1, 2).astype(np.float64)
        n_v = len(vertices)

        M = cv2.moments(contour)
        cx = M["m10"] / M["m00"] if M["m00"] > 0 else 0.0
        cy = M["m01"] / M["m00"] if M["m00"] > 0 else 0.0

        edge_lengths = []
        for i in range(n_v):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n_v]
            edge_lengths.append(float(np.linalg.norm(p2 - p1)))

        angles = []
        for i in range(n_v):
            v1 = vertices[(i - 1) % n_v] - vertices[i]
            v2 = vertices[(i + 1) % n_v] - vertices[i]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angles.append(float(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))))

        signed = sum(
            vertices[i][0] * vertices[(i+1) % n_v][1]
            - vertices[(i+1) % n_v][0] * vertices[i][1]
            for i in range(n_v)
        )
        if signed < 0:
            vertices = vertices[::-1].copy()

        longest_idx = int(np.argmax(edge_lengths))
        ps = vertices[longest_idx]
        pe = vertices[(longest_idx + 1) % n_v]
        theta = float(np.degrees(np.arctan2(pe[1]-ps[1], pe[0]-ps[0])) % 360.0)

        right_angles = sum(1 for a in angles if abs(a - 90.0) < m["angle_tol"])

        pieces.append({
            "contour": contour,
            "vertices": vertices,
            "n_vertices": n_v,
            "centroid": (cx, cy),
            "area": area,
            "edge_lengths": edge_lengths,
            "longest_edge": edge_lengths[longest_idx],
            "angles": angles,
            "right_angles": right_angles,
            "theta_current": theta,
        })

    return pieces


def detect_pieces(img, cfg):
    """
    完整管线: maix图像 -> (碎片列表, 二值图, otsu阈值, roi_info)
    roi_info = {"roi":..., "angle":..., "contour":..., "piece_rect":..., "puzzle_rect":...}
    碎片坐标已回映到全图。
    """
    img_cv = maix_image.image2cv(img, copy=False)
    h_img, w_img = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = enhance_gray(gray, cfg)

    # --- Stage 1: 找A4纸 (极性自适应) ---
    roi = None
    roi_contour = None
    roi_angle = 0.0
    roi_binary = None

    if cfg["roi"]["enabled"]:
        roi, roi_binary, roi_angle, roi_contour = find_a4_roi(gray, cfg)

    # --- 分区 ---
    piece_rect, puzzle_rect = compute_split(roi, cfg, w_img, h_img)
    wk_x, wk_y, wk_w, wk_h = piece_rect

    work_gray = gray[wk_y:wk_y+wk_h, wk_x:wk_x+wk_w]

    # --- 掩膜: 纸轮廓填白 -> bitwise_and (消灭角落白底) ---
    if roi_contour is not None and roi is not None:
        cnt_local = roi_contour.copy()
        cnt_local[:, 0, 0] -= wk_x
        cnt_local[:, 0, 1] -= wk_y
        mask = np.zeros((wk_h, wk_w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt_local], -1, 255, -1)
        work_gray = cv2.bitwise_and(work_gray, mask)

    # --- Stage 2: 预处理 + 二值化 + 轮廓 ---
    processed = preprocess(work_gray, cfg)

    b = cfg["binary"]
    if b["mode"] == "hsv":
        work_bgr = img_cv[wk_y:wk_y+wk_h, wk_x:wk_x+wk_w]
        binary, otsu_val = binarize(work_bgr, processed, cfg)
    else:
        binary, otsu_val = binarize(img_cv, processed, cfg)

    pieces_local = find_pieces(binary, cfg, wk_w, wk_h)

    # --- 坐标回映: 工作区 -> 全图 ---
    for p in pieces_local:
        v = p["vertices"]
        v[:, 0] += wk_x
        v[:, 1] += wk_y
        cx, cy = p["centroid"]
        p["centroid"] = (cx + wk_x, cy + wk_y)
        cnt = p["contour"]
        cnt[:, 0, 0] += wk_x
        cnt[:, 0, 1] += wk_y

    roi_info = {
        "roi": roi,
        "angle": roi_angle,
        "contour": roi_contour,
        "piece_rect": piece_rect,
        "puzzle_rect": puzzle_rect,
    }

    return pieces_local, binary, otsu_val, roi_info


def detect_in_region(gray_full, img_cv_full, rect, cfg, roi_contour=None):
    """
    在指定矩形区域内检测碎片 (参考历史模板标定工具的 detect_in_region)。
    每区独立做 掩膜->预处理->二值化->轮廓, 坐标最后回映到全图。
    rect = (x, y, w, h) 全图坐标。
    """
    rx, ry, rw, rh = rect
    if rw < 32 or rh < 32:
        return []
    work = gray_full[ry:ry+rh, rx:rx+rw]

    # 掩膜: 纸轮廓 (去掉区域角落的纸外背景)
    if roi_contour is not None:
        cnt_l = roi_contour.copy()
        cnt_l[:, 0, 0] -= rx
        cnt_l[:, 0, 1] -= ry
        msk = np.zeros((rh, rw), np.uint8)
        cv2.drawContours(msk, [cnt_l], -1, 255, -1)
        work = cv2.bitwise_and(work, msk)

    processed = preprocess(work, cfg)
    b = cfg["binary"]
    if b["mode"] == "hsv":
        work_bgr = img_cv_full[ry:ry+rh, rx:rx+rw]
        binary, _ = binarize(work_bgr, processed, cfg)
    else:
        binary, _ = binarize(img_cv_full, processed, cfg)

    pieces_local = find_pieces(binary, cfg, rw, rh)
    # 坐标回映到全图
    for p in pieces_local:
        p["vertices"][:, 0] += rx
        p["vertices"][:, 1] += ry
        cx, cy = p["centroid"]
        p["centroid"] = (cx + rx, cy + ry)
        p["contour"][:, 0, 0] += rx
        p["contour"][:, 0, 1] += ry
    return pieces_local


def detect_pieces_both_regions(img, cfg):
    """
    标定专用 (参考历史模板标定工具): 分别在碎片区(piece_rect)和拼图为(puzzle_rect)
    各自检测再合并, 坐标已回映全图。
    这样"方块放左半区 + 拼好的4片放右半区"两者都能被检测到。
    返回: (all_pieces, roi_info)
    """
    img_cv = maix_image.image2cv(img, copy=False)
    h_img, w_img = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = enhance_gray(gray, cfg)

    roi = None
    roi_contour = None
    roi_angle = 0.0
    if cfg["roi"]["enabled"]:
        roi, _, roi_angle, roi_contour = find_a4_roi(gray, cfg)

    piece_rect, puzzle_rect = compute_split(roi, cfg, w_img, h_img)

    all_pieces = []
    if piece_rect is not None:
        all_pieces += detect_in_region(gray, img_cv, piece_rect, cfg, roi_contour)
    if puzzle_rect is not None:
        all_pieces += detect_in_region(gray, img_cv, puzzle_rect, cfg, roi_contour)

    roi_info = {
        "roi": roi,
        "angle": roi_angle,
        "contour": roi_contour,
        "piece_rect": piece_rect,
        "puzzle_rect": puzzle_rect,
    }
    return all_pieces, roi_info


def sample_center_gray(img_cv, sample_size=5):
    """画面中心 sample_size x sample_size 平均灰度 (QuickSet用)"""
    h, w = img_cv.shape[:2]
    half = sample_size // 2
    cx, cy = w // 2, h // 2
    y1, y2 = max(0, cy-half), min(h, cy+half+1)
    x1, x2 = max(0, cx-half), min(w, cx+half+1)
    region = img_cv[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return int(np.mean(gray))


def count_contours_quick(img, cfg):
    """轻量检测: ROI -> 分区 -> 只数轮廓, 不做多边形逼近"""
    img_cv = maix_image.image2cv(img, copy=False)
    h_img, w_img = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = enhance_gray(gray, cfg)

    roi = None
    roi_contour = None
    if cfg["roi"]["enabled"]:
        roi, _, _, roi_contour = find_a4_roi(gray, cfg)

    piece_rect, _ = compute_split(roi, cfg, w_img, h_img)
    wk_x, wk_y, wk_w, wk_h = piece_rect
    work_gray = gray[wk_y:wk_y+wk_h, wk_x:wk_x+wk_w]

    if roi_contour is not None and roi is not None:
        cnt_local = roi_contour.copy()
        cnt_local[:, 0, 0] -= wk_x
        cnt_local[:, 0, 1] -= wk_y
        mask = np.zeros((wk_h, wk_w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt_local], -1, 255, -1)
        work_gray = cv2.bitwise_and(work_gray, mask)

    processed = preprocess(work_gray, cfg)
    binary, _ = binarize(img_cv, processed, cfg)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = sum(1 for c in contours if cv2.contourArea(c) >= cfg["contour"]["min_area"])
    return n
