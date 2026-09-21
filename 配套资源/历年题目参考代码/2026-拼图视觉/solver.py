"""
数学推理 - 核心逻辑来自 数学推理.py
指纹匹配(边长比例模式) -> 识别碎片 -> 计算旋转角 -> 目标轮廓
"""
import numpy as np


def sorted_edge_ratios(edges):
    """归一化边长比例 (最长边=1.0), 降序排列"""
    mx = max(edges) if edges else 1.0
    return sorted([e / (mx + 1e-10) for e in edges], reverse=True)


def identify_piece(piece, cfg):
    """
    匹配模板, 返回 (piece_id或None, score)
    评分: 面积0.35 + 边长比例模式0.40 + 直角数0.25
    """
    m = cfg["matching"]
    templates = cfg["templates"]

    n_v = piece["n_vertices"]
    area = piece["area"]
    edges = piece["edge_lengths"]
    longest = piece["longest_edge"]
    ra = piece["right_angles"]

    det_ratios = sorted_edge_ratios(edges)

    best_id = None
    best_score = -1.0

    for pid, t in templates.items():
        # 硬约束: 顶点数
        if n_v != t["n_vertices"]:
            continue
        # 软约束: 面积
        area_ratio = area / (t["area"] + 1e-10)
        if abs(area_ratio - 1.0) > m["area_tol"]:
            continue
        # 软约束: 最长边
        tmpl_longest = max(t["edges"]) if "edges" in t else t.get("longest_edge", 1)
        edge_ratio = longest / (tmpl_longest + 1e-10)
        if abs(edge_ratio - 1.0) > m["edge_tol"]:
            continue

        # 评分
        area_sim = max(0, 1.0 - abs(area_ratio - 1.0))

        # 边长比例模式 (归一化后逐边比)
        tmpl_ratios = sorted_edge_ratios(t["edges"]) if "edges" in t else []
        if len(det_ratios) == len(tmpl_ratios) and len(det_ratios) > 0:
            edge_sim = max(0, 1.0 - sum(abs(d - t_) for d, t_ in zip(det_ratios, tmpl_ratios)) / len(det_ratios))
        else:
            edge_sim = max(0, 1.0 - abs(edge_ratio - 1.0))

        r_sim = max(0, 1.0 - abs(ra - t["right_angles"]) * 0.5)

        score = area_sim * 0.35 + edge_sim * 0.40 + r_sim * 0.25
        if score > best_score:
            best_score = score
            best_id = pid

    if best_id and best_score >= 0.3:
        return best_id, best_score

    # 兜底: 去掉硬约束
    fb_best = -1.0
    fb_id = None
    for pid, t in templates.items():
        a_sim = max(0, 1 - abs(area / (t["area"] + 1e-10) - 1))
        tmpl_longest = max(t["edges"]) if "edges" in t else t.get("longest_edge", 1)
        e_sim = max(0, 1 - abs(longest / (tmpl_longest + 1e-10) - 1))
        v_sim = max(0, 1 - abs(n_v - t["n_vertices"]) * 0.5)
        r_sim = max(0, 1 - abs(ra - t["right_angles"]) * 0.5)
        s = a_sim * 0.35 + e_sim * 0.35 + v_sim * 0.15 + r_sim * 0.15
        if s > fb_best:
            fb_best = s
            fb_id = pid
    if fb_best >= m["fallback_min"]:
        return fb_id, fb_best

    return None, 0.0


def compute_delta_theta(piece, piece_id, cfg):
    """计算旋转角, 归一化到[-180, 180]"""
    target = cfg["templates"][piece_id]["target_theta"]
    current = piece["theta_current"]
    offset = cfg["display"]["global_rotation_offset"]
    dth = target - current + offset
    dth = (dth + 180.0) % 360.0 - 180.0
    return dth


def compute_target_contour(piece, piece_id, dth, puzzle_center, cfg):
    """
    计算碎片旋转dth后在puzzle区的目标轮廓顶点 (全图像素坐标)
    puzzle_center = (cx, cy) puzzle区中心
    返回: list of (x, y) int tuples
    """
    px_per_mm = cfg["calibration"]["px_per_mm"]
    off_mm = cfg["templates"][piece_id].get("target_offset_mm", [0, 0])
    tx = puzzle_center[0] + int(off_mm[0] * px_per_mm)
    ty = puzzle_center[1] + int(off_mm[1] * px_per_mm)

    cx, cy = piece["centroid"]
    rad = np.radians(dth)
    cos_t, sin_t = np.cos(rad), np.sin(rad)

    target_verts = []
    for v in piece["vertices"]:
        dx, dy = v[0] - cx, v[1] - cy
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        target_verts.append((int(tx + rx), int(ty + ry)))

    return target_verts, (tx, ty)


def solve_all(pieces, cfg):
    """
    识别所有碎片 + 算角度 + 查目标位置 + 计算目标轮廓
    每个模板只匹配一次(防重复), 高分优先
    """
    results = []
    used_templates = set()

    # 先对所有碎片做匹配
    candidates = []
    for piece in pieces:
        pid, score = identify_piece(piece, cfg)
        if pid is not None:
            candidates.append((score, piece, pid))
    candidates.sort(key=lambda x: x[0], reverse=True)

    # puzzle区中心 (从roi_info获取, 这里先用占位)
    # 实际中心在main.py里根据roi_info计算后传入

    for score, piece, pid in candidates:
        if pid in used_templates:
            continue
        used_templates.add(pid)
        dth = compute_delta_theta(piece, pid, cfg)
        t = cfg["templates"][pid]
        results.append({
        "id": pid,                        # 匹配到的模板ID: "A"/"B"/"C"/"D"
        "from_px": piece["centroid"],     # 碎片当前质心 (像素坐标), 机械臂去这里抓
        "target_offset_mm": t["target_offset_mm"],  # 目标位置相对puzzle中心的偏移(mm), main.py用它算target_px
        "theta": dth,                     # 需要旋转的角度(度), 归一化到[-180,+180], 发给E轴
        "score": score,                   # 匹配置信度 0~1, 越高越确定是这片
        "vertices": piece["vertices"],    # 碎片顶点坐标(全图像素), main.py画目标轮廓用
        "edge_lengths": piece["edge_lengths"],  # 各边长度(px), 备用/调试
        "n_vertices": piece["n_vertices"],      # 顶点数(3或4), 备用/调试
        })

    return results


def count_matched(pieces, cfg):
    """统计能匹配的碎片数 (主菜单显示用)"""
    used = set()
    n = 0
    for piece in pieces:
        pid, _ = identify_piece(piece, cfg)
        if pid is not None and pid not in used:
            used.add(pid)
            n += 1
    return n
