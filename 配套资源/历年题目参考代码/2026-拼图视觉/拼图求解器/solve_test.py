"""
2026拼图 - MaixCAM 实时拼图求解 (纯物理 mm 坐标版)
单文件：视觉管线 + 盲拼DFS(SAT碰撞) + 显示
"""

import cv2
from maix import display
import numpy as np
import math
import time

# =====================================================================
# 可配置参数 (物理 mm 标定)
# =====================================================================
CFG = {
    # 1. 坐标系与画布
    "canvas_size": 400.0,      # 画布尺寸，设大一点防止越界
    "px_to_mm": 10.0 / 33.0,   # 标定比例: 10mm 对应 33px

    # 2. 边长匹配容差
    "edge_match_tol_mm": 6.0,  # 单边误差 3mm，两边合起来最大差 6mm

    # 3. 过程剪枝 (防止拼成发散的十字形)
    # 题目最大 12x9cm，加上 20mm 缝隙容差，超过这个尺寸绝对拼错了
    "max_allowed_L": 140.0,    # 拼装过程中，外接矩形长边超过 140mm 直接剪枝
    "max_allowed_W": 110.0,    # 拼装过程中，外接矩形短边超过 110mm 直接剪枝

    # 4. 最终校验 (终点绝对尺寸)
    "valid_final_sizes": [
        (120.0, 90.0),  # 12x9 cm
        (90.0, 50.0),   # 9x5 cm
        # 如果比赛中可能出现其他尺寸，比如 12x5 等，加在这里
    ],
    "size_tol_mm": 20.0,  # 尺寸允许误差 (即你要求的最后缝隙不超过20mm)

    # 5. 调试
    "debug_dfs": False,  # 建议先开着，看它剪枝和成功的日志，稳定后再关掉
    "solve_timeout_s": 5.0, 
}

# 视觉参数 (不变)
VISION_CFG = {
    "smooth_kernel": 3, "gaussian_kernel": 5, "morph_kernel_size": 3,
    "morph_close_iter": 2, "morph_dilate_iter": 2, "morph_erode_iter": 2,
    "binary_mode": "otsu", "binary_invert": False, "fixed_threshold": 127,
    "hsv_lower": [0, 50, 50], "hsv_upper": [10, 255, 255],
    "min_area": 2000, "epsilon": 0.02, "max_vertices": 5,
    "area_tol": 0.3, "edge_tol": 0.3, "angle_tol": 5.0,
}

CAM_W, CAM_H = 640, 448
_morph_kernel = None

# =====================================================================
# Piece 类 — 内部全部使用 mm 坐标
# =====================================================================
class Piece:
    def __init__(self, pid, vertices_px, area_px, centroid_px, px_to_mm):
        self.pid = pid
        k = px_to_mm
        
        # 1. 顶点转换为 mm
        verts = np.array(vertices_px, dtype=np.float32) * k
        self.vertices_local = verts

        # 2. 面积转换为 mm^2
        self.area = float(area_px) * (k ** 2)

        # 3. 质心转换为 mm (算法内部用)
        self.centroid = (float(centroid_px[0]) * k, float(centroid_px[1]) * k)
        # 3b. 保留原始像素质心 (输出给机械臂用)
        self.centroid_px = (float(centroid_px[0]), float(centroid_px[1]))

        # 4. 预计算边信息 (基于 mm 坐标)
        self.edge_info = []
        n = len(verts)
        for i in range(n):
            start = verts[i]
            end = verts[(i + 1) % n]
            length = float(np.linalg.norm(end - start))
            direction = math.degrees(math.atan2(
                end[1] - start[1], end[0] - start[0])) % 360.0
            self.edge_info.append((start, end, length, direction))


# =====================================================================
# 核心算法函数
# =====================================================================

def get_edge(piece, idx):
    return (piece.vertices_local[idx],
            piece.vertices_local[(idx + 1) % len(piece.vertices_local)])


def is_edge_match(len_a_mm, len_b_mm, cfg=CFG):
    """纯 mm 级绝对误差判断"""
    diff = abs(len_a_mm - len_b_mm)
    return diff <= cfg["edge_match_tol_mm"]


def attach_piece(edge_a, edge_b, piece_b_verts, edge_b_idx):
    """将碎片B的一条边反向贴合到碎片A的一条边上 (全在mm坐标系下)"""
    p_start, p_end = edge_a[0], edge_a[1]
    q_start, q_end = edge_b[0], edge_b[1]

    vec_a = p_end - p_start
    vec_b = q_end - q_start

    angle = (math.atan2(vec_a[1], vec_a[0])
             - math.atan2(vec_b[1], vec_b[0])
             + math.pi)

    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle),  math.cos(angle)]], dtype=np.float32)

    rotated = piece_b_verts @ R.T
    T = p_end - rotated[edge_b_idx]
    abs_verts = rotated + T

    return R, abs_verts


def polygons_overlap_sat(poly1, poly2, contact_tol=1e-4):
    """SAT 碰撞检测"""
    max_pen = 0.0
    for poly in (poly1, poly2):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            axis = np.array([-edge[1], edge[0]], dtype=np.float32)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-8:
                continue
            axis /= axis_norm

            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)

            if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
                return False, 0.0

            overlap = min(max1, max2) - max(min1, min2)
            if overlap > contact_tol:
                max_pen = max(max_pen, overlap)

    return max_pen > contact_tol, max_pen


def validate_dynamic_rectangle(abs_verts_list, cfg=CFG):
    """验证拼好的组合体是否为合法矩形 (基于绝对尺寸)"""
    all_pts = np.vstack(abs_verts_list).reshape(-1, 1, 2).astype(np.float32)
    rect = cv2.minAreaRect(all_pts)
    (_, (w, h), _) = rect

    long_mm = max(w, h)
    short_mm = min(w, h)

    # 检查是否匹配任意一个合法尺寸
    for valid_L, valid_W in cfg["valid_final_sizes"]:
        # 长边差值
        diff_L = abs(long_mm - valid_L)
        # 短边差值
        diff_W = abs(short_mm - valid_W)
        
        # 如果长宽都在 20mm 容差范围内，说明拼对了！
        if diff_L <= cfg["size_tol_mm"] and diff_W <= cfg["size_tol_mm"]:
            print(f"  [成功] 尺寸匹配! 当前: {long_mm:.1f}x{short_mm:.1f}mm, 期望: {valid_L}x{valid_W}mm")
            return True, rect

    print(f"  [拒绝] 尺寸不符: 当前={long_mm:.1f}x{short_mm:.1f}mm, 容差={cfg['size_tol_mm']}mm")
    return False, None


# =====================================================================
# 盲拼 DFS 求解器 (修复角度180度反向Bug)
# =====================================================================

def solve_puzzle(pieces, cfg=CFG):
    pieces_dict = {p.pid: p for p in pieces}
    total_area_mm = sum(p.area for p in pieces)
    canvas = cfg["canvas_size"]
    center = canvas / 2.0

    root_pid = max(pieces_dict, key=lambda k: pieces_dict[k].area)
    root = pieces_dict[root_pid]

    root_abs = root.vertices_local + np.array([center, center], dtype=np.float32)
    placed_verts = [root_abs]

    placements = {root_pid: {"R": np.eye(2, dtype=np.float32), "abs_v": root_abs}}
    used = [root_pid]
    n_pieces = len(pieces_dict)
    debug = cfg.get("debug_dfs", False)
    attempt_count = [0]
    result_container = [None]
    
    # ================= 新增：超时控制 =================
    t_start = time.time()
    deadline = t_start + cfg.get("solve_timeout_s", 5.0)
    timeout_flag = [False] # 用来标记是否是因为超时退出的
    # ================================================

    def dfs(cur_placements, cur_placed_verts, cur_used):
        # ================= 新增：每次递归先检查闹钟 =================
        if time.time() > deadline:
            timeout_flag[0] = True
            return False
        # ==========================================================

        if len(cur_placements) == n_pieces:
            abs_list = [d["abs_v"] for d in cur_placements.values()]
            valid, rect_info = validate_dynamic_rectangle(abs_list, cfg)
            if valid:
                _, (w, h), _ = rect_info
                print("  [成功] 拼图有效! 尺寸: %.1f x %.1f mm" % (max(w,h), min(w,h)))
                result_container[0] = dict(cur_placements)
            return valid

        for pid_a, data_a in cur_placements.items():
            verts_a = data_a["abs_v"]
            n_a = len(verts_a)

            for i in range(n_a):
                edge_a = np.array([verts_a[i], verts_a[(i + 1) % n_a]], dtype=np.float32)
                len_a = np.linalg.norm(edge_a[1] - edge_a[0])

                for pid_b, piece_b in pieces_dict.items():
                    if pid_b in cur_used:
                        continue

                    n_b = len(piece_b.vertices_local)
                    for j in range(n_b):
                        edge_b = get_edge(piece_b, j)
                        len_b = np.linalg.norm(edge_b[1] - edge_b[0])

                        if not is_edge_match(len_a, len_b, cfg):
                            continue

                        attempt_count[0] += 1
                        if debug:
                            print("  [DFS] try: %s.edge%d(%.1fmm) <-> %s.edge%d(%.1fmm)" %
                                  (pid_a, i, len_a, pid_b, j, len_b))

                        R, new_abs = attach_piece(edge_a, edge_b, piece_b.vertices_local, j)

                        overlap = False
                        for pv in cur_placed_verts:
                            if pv is cur_placements[pid_a]["abs_v"]:
                                continue
                            ov, _ = polygons_overlap_sat(pv, new_abs, 1e-2)
                            if ov:
                                overlap = True
                                break

                        if overlap:
                            if debug: print("         -> 重叠，放弃")
                            continue

                        new_placed_verts = cur_placed_verts + [new_abs]
                        
                        if len(new_placed_verts) >= 2:
                            all_pts = np.vstack(new_placed_verts).reshape(-1, 1, 2).astype(np.float32)
                            rect = cv2.minAreaRect(all_pts)
                            (_, (cw, ch), _) = rect
                            cur_L = max(cw, ch)
                            cur_W = min(cw, ch)
                            
                            if cur_L > cfg["max_allowed_L"] or cur_W > cfg["max_allowed_W"]:
                                if debug: 
                                    print(f"         -> [剪枝] 外框超标 ({cur_L:.1f}x{cur_W:.1f}mm > {cfg['max_allowed_L']}x{cfg['max_allowed_W']}), 放弃")
                                continue

                        new_placements = dict(cur_placements)
                        new_placements[pid_b] = {"R": R, "abs_v": new_abs}

                        if dfs(new_placements, new_placed_verts, cur_used + [pid_b]):
                            return True
        return False

    found = dfs(placements, placed_verts, used)
    dt = time.time() - t_start
    
    # ================= 新增：打印超时日志 =================
    if timeout_flag[0]:
        print("[超时] 求解超过 %.1f 秒，强制终止！尝试次数=%d" % (cfg.get("solve_timeout_s", 5.0), attempt_count[0]))
    else:
        print("DFS耗时: %.3fs  尝试次数=%d  结果=%s" % (dt, attempt_count[0], found))
    # ======================================================

    if not found:
        return None

    # ---------- 提取结果 (保持上一轮修改的相对坐标输出) ----------
    final_placements = result_container[0]
    all_abs = np.vstack([d["abs_v"] for d in final_placements.values()])
    rect = cv2.minAreaRect(all_abs.reshape(-1, 1, 2).astype(np.float32))
    (puzzle_cx, puzzle_cy), (w, h), rect_angle = rect

    if w < h:
        align_angle_deg = 90.0 + rect_angle
    else:
        align_angle_deg = rect_angle

    align_angle_rad = math.radians(align_angle_deg)
    M_align = np.array([
        [math.cos(align_angle_rad), -math.sin(align_angle_rad)],
        [math.sin(align_angle_rad),  math.cos(align_angle_rad)]
    ], dtype=np.float32)

    # mm -> px 的换算系数
    mm_to_px = 1.0 / cfg["px_to_mm"]

    results = []
    for pid, data in final_placements.items():
        piece = pieces_dict[pid]
        abs_v = data["abs_v"]
        R_place = data["R"]

        # from = 原始像素质心 (摄像头看到的, 机械臂去抓的位置)
        from_x = piece.centroid_px[0]
        from_y = piece.centroid_px[1]

        # 计算该碎片在拼图中的相对偏移 (mm)
        abs_cx = float(np.mean(abs_v[:, 0]))
        abs_cy = float(np.mean(abs_v[:, 1]))

        delta_x = abs_cx - puzzle_cx
        delta_y = abs_cy - puzzle_cy
        delta_vec = np.array([delta_x, delta_y], dtype=np.float32)

        # 对齐到矩形主轴后的偏移 (mm)
        local_vec = M_align @ delta_vec
        # 转为像素偏移
        offset_x_px = float(local_vec[0]) * mm_to_px
        offset_y_px = float(local_vec[1]) * mm_to_px

        # 该碎片在拼图中的旋转角 (度)
        R_total = M_align @ R_place
        theta = math.degrees(math.atan2(R_total[1, 0], R_total[0, 0]))
        theta = (theta + 180.0) % 360.0 - 180.0

        results.append({
            "id": pid,
            "from_x": from_x,          # 像素质心 x (抓取点)
            "from_y": from_y,          # 像素质心 y (抓取点)
            "offset_x": offset_x_px,   # 相对拼图中心的像素偏移 x
            "offset_y": offset_y_px,   # 相对拼图中心的像素偏移 y
            "theta": theta             # 该碎片在拼图中的旋转角 (度)
        })

    return results, final_placements

# =====================================================================
# 全局变换: 把相对偏移 → 绝对像素目标位置
# =====================================================================

def compute_arm_commands(results, target_center_px, global_angle_deg=0.0):
    """
    把拼图的相对结果转换为机械臂的绝对像素指令。

    参数:
        results: solve_puzzle() 的输出 (每片的 from/offset/theta)
        target_center_px: 拼图中心要放在哪 (像素坐标), 例如 (320, 224)
        global_angle_deg: 拼图整体朝向 (度), 0=不转, 正=逆时针

    返回:
        [
            {"id": "P0", "from": (px, py), "to": (tx, ty), "th": 角度},
            ...
        ]
        - from: 抓取点 (像素质心, 摄像头看到的)
        - to:   放置点 (像素坐标, 机械臂要放的位置)
        - th:   旋转角度 (度, 正=逆时针)
    """
    tx, ty = target_center_px
    g_rad = math.radians(global_angle_deg)
    cos_g = math.cos(g_rad)
    sin_g = math.sin(g_rad)

    commands = []
    for r in results:
        # 1. 把偏移向量旋转 global_angle
        ox = r["offset_x"]
        oy = r["offset_y"]
        rot_x = ox * cos_g - oy * sin_g
        rot_y = ox * sin_g + oy * cos_g

        # 2. 加上拼图中心 → 得到绝对像素目标
        to_x = tx + rot_x
        to_y = ty + rot_y

        # 3. 总旋转 = 碎片在拼图中的角度 + 整体朝向
        total_th = r["theta"] + global_angle_deg
        # 归一化到 [-180, 180]
        total_th = (total_th + 180.0) % 360.0 - 180.0

        commands.append({
            "id": r["id"],
            "from": (r["from_x"], r["from_y"]),  # 像素 (抓取)
            "to": (to_x, to_y),                   # 像素 (放置)
            "th": total_th,                        # 度 (旋转)
        })

    return commands


# =====================================================================
# 软拼接可视化: 在图像上模拟拼合效果
# =====================================================================

def transform_contour(vertices, centroid_px, to_px, theta_deg):
    """
    对一个碎片的轮廓做刚体变换 (旋转+平移):
      1. 绕质心旋转 theta_deg 度
      2. 质心平移到 to_px
    输入:
        vertices:    [(x,y), ...] 原始像素顶点
        centroid_px: (cx, cy) 当前质心 (像素)
        to_px:       (tx, ty) 目标质心 (像素)
        theta_deg:   旋转角度 (度, 正=逆时针)
    输出:
        [(x,y), ...] 变换后的像素顶点
    """
    cx, cy = centroid_px
    tx, ty = to_px
    rad = math.radians(theta_deg)
    cos_t = math.cos(rad)
    sin_t = math.sin(rad)

    result = []
    for pt in vertices:
        x, y = float(pt[0]), float(pt[1])
        dx = x - cx
        dy = y - cy
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        result.append((rx + tx, ry + ty))
    return result


def draw_assembly_preview(pieces_raw, arm_cmds, canvas_size=(640, 448)):
    """
    在空白画布上画出所有碎片变换后的轮廓, 预览拼合效果。

    参数:
        pieces_raw:  vision 输出的 pieces 列表 (含 "vertices", "centroid")
        arm_cmds:    compute_arm_commands() 的输出
        canvas_size: 画布尺寸 (宽, 高)
    返回:
        canvas (numpy BGR 图像)
    """
    W, H = canvas_size
    canvas = np.full((H, W, 3), (40, 40, 40), dtype=np.uint8)

    colors = [(0, 255, 0), (255, 100, 0), (0, 200, 255), (255, 0, 255)]
    n = min(len(pieces_raw), len(arm_cmds))

    for i in range(n):
        p = pieces_raw[i]
        c = arm_cmds[i]
        color = colors[i % len(colors)]

        # 原始轮廓 (灰色, 表示当前位置)
        orig_pts = np.array(p["vertices"], dtype=np.int32)
        cv2.polylines(canvas, [orig_pts], True, (80, 80, 80), 1)

        # 变换后轮廓 (彩色, 表示目标位置)
        transformed = transform_contour(
            p["vertices"],
            centroid_px=(c["from"][0], c["from"][1]),
            to_px=(c["to"][0], c["to"][1]),
            theta_deg=c["th"]
        )
        t_pts = np.array(transformed, dtype=np.int32)
        cv2.polylines(canvas, [t_pts], True, color, 2)

        # 标注
        tcx = int(np.mean([pt[0] for pt in transformed]))
        tcy = int(np.mean([pt[1] for pt in transformed]))
        cv2.putText(canvas, "%s(%.0f)" % (c["id"], c["th"]),
                    (tcx - 15, tcy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # from -> to 箭头
        fx, fy = int(c["from"][0]), int(c["from"][1])
        tx, ty = int(c["to"][0]), int(c["to"][1])
        cv2.arrowedLine(canvas, (fx, fy), (tx, ty), (255, 255, 255), 1, tipLength=0.1)

    cv2.putText(canvas, "Assembly Preview", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


# =====================================================================
# 视觉管线 (不变)
# =====================================================================
def _get_morph_kernel():
    global _morph_kernel
    if _morph_kernel is None:
        _morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (VISION_CFG["morph_kernel_size"], VISION_CFG["morph_kernel_size"]))
    return _morph_kernel

def preprocess(img_cv):
    p = VISION_CFG
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, p["smooth_kernel"])
    ksize = p["gaussian_kernel"]
    gaussian = cv2.GaussianBlur(median, (ksize, ksize), 0)
    kernel = _get_morph_kernel()
    closed = cv2.morphologyEx(gaussian, cv2.MORPH_CLOSE, kernel, iterations=p["morph_close_iter"])
    return closed

def binarize(img_cv, processed):
    b = VISION_CFG
    kernel = _get_morph_kernel()
    flag = cv2.THRESH_BINARY_INV if b["binary_invert"] else cv2.THRESH_BINARY
    otsu_value = 0
    if b["binary_mode"] == "otsu":
        otsu_value, binary = cv2.threshold(processed, 0, 255, flag + cv2.THRESH_OTSU)
    elif b["binary_mode"] == "fixed":
        _, binary = cv2.threshold(processed, b["fixed_threshold"], 255, flag)
    elif b["binary_mode"] == "hsv":
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        lower = np.array(b["hsv_lower"], dtype=np.uint8)
        upper = np.array(b["hsv_upper"], dtype=np.uint8)
        binary = cv2.inRange(hsv, lower, upper)
    else:
        otsu_value, binary = cv2.threshold(processed, 0, 255, flag + cv2.THRESH_OTSU)
    dilated = cv2.dilate(binary, kernel, iterations=b["morph_dilate_iter"])
    eroded = cv2.erode(dilated, kernel, iterations=b["morph_erode_iter"])
    return eroded, int(otsu_value)

def find_pieces(binary):
    c = VISION_CFG
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < c["min_area"]: continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, c["epsilon"] * peri, True)
        if len(approx) > c["max_vertices"]:
            for eps in [0.03, 0.04, 0.05, 0.06, 0.08]:
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                if len(approx) <= c["max_vertices"]: break
            else: continue
        vertices = approx.reshape(-1, 2).astype(np.float64)
        n_v = len(vertices)
        M = cv2.moments(contour)
        cx = M["m10"] / M["m00"] if M["m00"] > 0 else 0.0
        cy = M["m01"] / M["m00"] if M["m00"] > 0 else 0.0
        edge_lengths = []
        for i in range(n_v):
            p1 = vertices[i]; p2 = vertices[(i + 1) % n_v]
            edge_lengths.append(float(np.linalg.norm(p2 - p1)))
        angles = []
        for i in range(n_v):
            v1 = vertices[(i - 1) % n_v] - vertices[i]
            v2 = vertices[(i + 1) % n_v] - vertices[i]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angles.append(float(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))))
        signed = sum(vertices[i][0] * vertices[(i + 1) % n_v][1] - vertices[(i + 1) % n_v][0] * vertices[i][1] for i in range(n_v))
        if signed < 0: vertices = vertices[::-1].copy()
        longest_idx = int(np.argmax(edge_lengths))
        ps = vertices[longest_idx]; pe = vertices[(longest_idx + 1) % n_v]
        theta = float(np.degrees(np.arctan2(pe[1] - ps[1], pe[0] - ps[0])) % 360.0)
        right_angles = sum(1 for a in angles if abs(a - 90.0) < c["angle_tol"])
        pieces.append({
            "contour": contour, "vertices": vertices, "n_vertices": n_v,
            "centroid": (cx, cy), "area": area, "edge_lengths": edge_lengths,
            "longest_edge": edge_lengths[longest_idx], "right_angles": right_angles,
            "theta_current": theta,
        })
    return pieces

def detect_pieces_full(img_cv):
    processed = preprocess(img_cv)
    binary, otsu_val = binarize(img_cv, processed)
    pieces = find_pieces(binary)
    return pieces, binary, otsu_val


# =====================================================================
# MaixCAM 实时主循环
# =====================================================================
def main():
    from maix import camera, app, time, image, display

    cam = camera.Camera(CAM_W, CAM_H)
    cam.skip_frames(30)
    disp = display.Display()

    print("=" * 50)
    print("2026拼图 - MaixCAM 实时求解 (纯 mm 坐标版)")
    print("标定比例: 1 px = %.4f mm" % CFG["px_to_mm"])
    print("=" * 50)

    frame_count = 0
    last_solve_time = 0
    last_results = None

    while not app.need_exit():
        t0 = time.ticks_ms()
        img = cam.read()
        img = img.lens_corr(strength=1.55)  # 畸变矫正 (临时, 之后换 remap)
        frame_count += 1

        img_cv = image.image2cv(img, copy=False)

        # 视觉管线 (输出像素坐标)
        pieces, binary, otsu_val = detect_pieces_full(img_cv)

        # 拼图求解 (限制频率: 200毫秒秒求解一次，避免卡顿)
        solve_now = (len(pieces) >= 2) and (time.ticks_ms() - last_solve_time > 200)
        if solve_now:
            piece_objs = []
            for i, p in enumerate(pieces):
                # 传入像素坐标和 px_to_mm，Piece 内部会自动转为 mm
                piece_objs.append(Piece("P%d" % i, p["vertices"], p["area"], p["centroid"], CFG["px_to_mm"]))

            print("\n--- Frame %d: 开始求解 (%d 片) ---" % (frame_count, len(pieces)))
            result = solve_puzzle(piece_objs)
            last_solve_time = time.ticks_ms()
            if result is not None:
                last_results, _ = result
                # ===== 全局变换: 拼图中心放哪 + 整体朝向 =====
                ARM_TARGET = (320, 224)   # 拼图中心像素位置 (你改这里!)
                ARM_ANGLE  = 0.0          # 整体朝向角度 (你改这里!) --- 之后会放在A4纸张的一边
                arm_cmds = compute_arm_commands(last_results, ARM_TARGET, ARM_ANGLE)
                print(">> 求解成功! 机械臂指令 (像素):")
                for c in arm_cmds:
                    print("  %s: from=(%.0f,%.0f) -> to=(%.0f,%.0f)  th=%.1f" %
                          (c["id"], c["from"][0], c["from"][1],
                                    c["to"][0], c["to"][1], c["th"]))
            else:
                last_results = None
                print(">> 求解失败 (无解)")

        # 可视化 (在屏幕上画的东西还是用像素画，因为屏幕是像素的)
        show = img_cv.copy()
        for i, p in enumerate(pieces):
            v = p["vertices"].astype(np.int32)
            cv2.polylines(show, [v], True, (0, 255, 0), 3)
            cx, cy = int(p["centroid"][0]), int(p["centroid"][1])
            cv2.circle(show, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(show, "P%d" % i, (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 1)
        # 画目标位置 (已经是像素坐标, 直接画)
        if last_results:
            ARM_TARGET_VIZ = (320, 224)
            arm_viz = compute_arm_commands(last_results, ARM_TARGET_VIZ, 0.0)
            for c in arm_viz:
                tx_px = int(c["to"][0])
                ty_px = int(c["to"][1])
                cv2.circle(show, (tx_px, ty_px), 12, (255, 0, 255), 3)
                cv2.putText(show, "T_%s" % c["id"], (tx_px - 20, ty_px - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        fps = 1000 / max(1, (time.ticks_ms() - t0))
        cv2.putText(show, "FPS: %.1f  Pieces: %d" % (fps, len(pieces)),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(show, "Otsu: %d" % otsu_val, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        # ===== 软拼接预览 (右半屏) =====
        if last_results and len(pieces) >= 2:
            ARM_T = (320, 224)
            cmds_viz = compute_arm_commands(last_results, ARM_T, 0.0)
            preview = draw_assembly_preview(pieces, cmds_viz, (320, 448))
            # 左半: 摄像头缩小到 320x448
            show_left = cv2.resize(show, (320, 448))
            # 拼接左右
            show = np.hstack([show_left, preview])

        show_maix = image.cv2image(show, copy=False)
        disp.show(show_maix)

if __name__ == "__main__":
    main()