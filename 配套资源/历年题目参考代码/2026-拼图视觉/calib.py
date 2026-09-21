"""
比赛现场标定 - 30mm方块标定 + 模板特征重提取
Step2: 检测30mm方块算px_per_mm, 识别拼好的碎片更新模板
"""
import numpy as np
import cv2

SQUARE_SIZE_MM = 30.0          # 标定方块物理边长(mm)
PX_PER_MM_MIN = 0.5            # 合理下限
PX_PER_MM_MAX = 5.0            # 合理上限


def detect_calib_square(pieces):
    """
    在pieces中检测30mm标定方块
    判定: 4顶点 + 直角>=3 + 边长CV<0.18 + minAreaRect长宽比<1.25
    返回: (square_piece, px_per_mm) 或 (None, None)
    """
    best = None
    best_score = 0.0
    best_side_px = 0.0

    for p in pieces:
        if p["n_vertices"] != 4:
            continue
        if p["right_angles"] < 3:
            continue

        els = p["edge_lengths"]
        if len(els) < 4:
            continue
        mean_e = float(np.mean(els))
        if mean_e <= 0:
            continue
        cv_val = float(np.std(els)) / mean_e
        if cv_val > 0.18:
            continue

        rect = cv2.minAreaRect(p["contour"])
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > 1.25:
            continue

        side_px = (rw + rh) / 2.0
        score = (p["right_angles"] / 4.0) * (1.0 - cv_val) * (1.0 - abs(aspect - 1.0))
        if score > best_score:
            best_score = score
            best = p
            best_side_px = side_px

    if best is None:
        return None, None

    px_per_mm = best_side_px / SQUARE_SIZE_MM
    # 合理性检查
    if px_per_mm < PX_PER_MM_MIN or px_per_mm > PX_PER_MM_MAX:
        return best, None

    return best, px_per_mm


def map_pieces_to_templates(pieces):
    """
    几何启发映射: A=唯一三角形(面积最大), B/C/D=四边形按面积降序
    返回: {"A": piece, "B": piece, "C": piece, "D": piece} 或 None
    """
    tris = [p for p in pieces if p["n_vertices"] == 3]
    quads = [p for p in pieces if p["n_vertices"] == 4]

    if len(tris) != 1 or len(quads) != 3:
        return None

    tri = tris[0]
    quads.sort(key=lambda p: p["area"], reverse=True)

    max_quad_area = max(q["area"] for q in quads)
    if tri["area"] < max_quad_area:
        return None

    return {"A": tri, "B": quads[0], "C": quads[1], "D": quads[2]}


def center_of_mapping(mapping):
    """所有已映射碎片质心的平均值 (拼图几何中心)"""
    xs = [mapping[k]["centroid"][0] for k in mapping]
    ys = [mapping[k]["centroid"][1] for k in mapping]
    return (float(np.mean(xs)), float(np.mean(ys)))


def apply_to_cfg(cfg, mapping, px_per_mm):
    """
    把标定结果写入cfg内存 (不写文件, 由main.save_config写盘)
    更新: px_per_mm + 模板特征 + target_offset_mm
    """
    center = center_of_mapping(mapping)

    # 更新像素密度
    cfg["calibration"]["px_per_mm"] = round(px_per_mm, 4)

    # 更新每个模板
    for pid in ("A", "B", "C", "D"):
        p = mapping[pid]
        t = cfg["templates"][pid]

        t["n_vertices"] = int(p["n_vertices"])
        t["area"] = int(round(p["area"]))
        t["edges"] = [round(float(e), 1) for e in p["edge_lengths"]]
        t["right_angles"] = int(p["right_angles"])
        t["target_theta"] = round(float(p["theta_current"]), 1)

        # target_offset_mm: 碎片质心相对拼图中心的偏移, px -> mm
        off_x = (p["centroid"][0] - center[0]) / px_per_mm
        off_y = (p["centroid"][1] - center[1]) / px_per_mm
        t["target_offset_mm"] = [round(off_x, 1), round(off_y, 1)]

    return center
