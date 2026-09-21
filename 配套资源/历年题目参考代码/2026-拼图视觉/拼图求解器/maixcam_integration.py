"""
MaixCAM Pro 集成使用指南
===========================

数据流向：
[摄像头] -> [检测分割] -> [透视矫正+物理标定] -> [Piece数据构造] -> [求解器] -> [拼接位姿] -> [机械臂/显示]

关键：输入必须是**物理毫米空间**的准确几何量。
"""

import json
import numpy as np
from embedded_puzzle_solver import (
    PuzzleSolver, Piece, SolverConfig, Placement,
    min_area_rect_numpy, polygons_overlap_sat
)

# ============================================================
# 1. 从标定/检测模块获取数据 -> 构造 Piece
# ============================================================

def build_pieces_from_detection(
    contours_px: list,           # 各片轮廓像素坐标 (N,1,2) 或 (N,2)
    homography_px2mm: np.ndarray, # 3x3 像素->毫米单应性矩阵
    piece_numbers: list,         # 编号
    mm_per_px: float = None      # 备选：直接像素换算比例
) -> list:
    """
    将检测到的像素轮廓转为物理毫米空间 Piece 对象。
    要求：透视已矫正（或由 homography 矫正），物理尺度已知。
    """
    pieces = []
    for idx, (contour, number) in enumerate(zip(contours_px, piece_numbers)):
        # 1. 转为 (N,2)
        if contour.ndim == 3:
            contour = contour.reshape(-1, 2)
        contour = contour.astype(np.float64)

        # 2. 多边形近似（可选，检测端通常已输出多边形）
        # epsilon = 0.01 * cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)

        # 3. 像素 -> 毫米
        if mm_per_px is not None:
            # 简单比例缩放（仅正交相机适用）
            verts_mm = contour * mm_per_px
        else:
            # 透视变换
            pts_h = np.hstack([contour, np.ones((len(contour), 1))])  # (N,3)
            verts_mm = (homography_px2mm @ pts_h.T).T
            verts_mm = verts_mm[:, :2] / verts_mm[:, 2:3]

        # 4. 质心归一化（局部坐标系）
        centroid = np.mean(verts_mm, axis=0)
        verts_local = verts_mm - centroid

        # 5. 确保 CCW（逆时针）
        area = polygon_area_signed(verts_local)
        if area < 0:
            verts_local = verts_local[::-1]  # 反转为 CCW
            area = -area

        # 6. 边长
        edges = np.roll(verts_local, -1, axis=0) - verts_local
        edge_lens = np.linalg.norm(edges, axis=1)

        pieces.append(Piece(
            number=number,
            vertices_mm=verts_local,
            edge_lengths_mm=edge_lens,
            area_mm2=abs(area),
            centroid_px=(float(np.mean(contour[:, 0])), float(np.mean(contour[:, 1])))
        ))

    return pieces


def polygon_area_signed(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


# ============================================================
# 2. 加载配置 & 运行求解
# ============================================================

def load_config(path: str) -> SolverConfig:
    with open(path, 'r') as f:
        data = json.load(f)
    return SolverConfig(**data['solver'])


def solve_puzzle(pieces: list, config: SolverConfig) -> tuple:
    """
    返回: (placements_dict, score, nodes, success_flag)
    placements: {piece_id: Placement}
    """
    solver = PuzzleSolver(pieces, config)
    placements, score, nodes = solver.solve()
    return placements, score, nodes, placements is not None


# ============================================================
# 3. 结果后处理：供机械臂/上位机使用
# ============================================================

def placements_to_robot_commands(
    placements: dict,
    homography_mm2px: np.ndarray = None,
    origin_offset_mm: tuple = (0.0, 0.0)
) -> list:
    """
    将拼接位姿转为机械臂目标点。
    每片输出: {piece_id, x_mm, y_mm, angle_deg, cx_px, cy_px}
    """
    cmds = []
    for pid, p in sorted(placements.items()):
        cx, cy = p.translation_mm[0] + origin_offset_mm[0], p.translation_mm[1] + origin_offset_mm[1]
        angle_deg = float(np.degrees(p.angle_rad))

        # 可选：投影回像素坐标给视觉校验
        px, py = None, None
        if homography_mm2px is not None:
            pt_mm = np.array([cx, cy, 1.0])
            pt_px = homography_mm2px @ pt_mm
            px, py = pt_px[0] / pt_px[2], pt_px[1] / pt_px[2]

        cmds.append({
            'piece_id': pid,
            'x_mm': round(cx, 2),
            'y_mm': round(cy, 2),
            'angle_deg': round(angle_deg, 3),
            'center_px': [round(px, 1), round(py, 1)] if px is not None else None
        })
    return cmds


def validate_assembly(placements: dict, pieces: dict, config: SolverConfig) -> dict:
    """
    最终验证：零重叠、尺寸贴合、边匹配
    """
    all_verts = np.vstack([p.vertices_world for p in placements.values()])
    long_s, short_s, angle, rect_pts = min_area_rect_numpy(all_verts)

    # 重叠检查
    max_overlap = 0.0
    overlap_pairs = []
    ids = sorted(placements.keys())
    for i, a_id in enumerate(ids):
        for b_id in ids[i+1:]:
            ov, depth, _ = polygons_overlap_sat(placements[a_id].vertices_world, placements[b_id].vertices_world)
            if ov:
                max_overlap = max(max_overlap, depth)
                overlap_pairs.append({'a': a_id, 'b': b_id, 'depth_mm': round(depth, 3)})

    # 尺寸误差
    size_err_long = abs(long_s - config.target_long_mm) / config.target_long_mm
    size_err_short = abs(short_s - config.target_short_mm) / config.target_short_mm
    aspect_err = abs((long_s/short_s) / (config.target_long_mm/config.target_short_mm) - 1.0)

    return {
        'assembly_bbox_mm': [round(long_s, 2), round(short_s, 2)],
        'target_bbox_mm': [config.target_long_mm, config.target_short_mm],
        'size_error_pct': [round(size_err_long*100, 2), round(size_err_short*100, 2)],
        'aspect_error_pct': round(aspect_err*100, 2),
        'max_overlap_depth_mm': round(max_overlap, 3),
        'overlap_pairs': overlap_pairs,
        'passed': max_overlap <= config.overlap_eps_mm2 and size_err_long <= 0.02 and size_err_short <= 0.02
    }


# ============================================================
# 4. 完整示例：从 JSON 配置 + 模拟数据运行
# ============================================================

if __name__ == '__main__':
    # 1. 加载配置
    config = load_config('solver_config.json')

    # 2. 模拟 4 片数据（实际来自检测模块）
    # A4 297x210，切成 2x2
    W, H = 297.0, 210.0
    hw, hh = W/2, H/2
    rects_mm = [
        np.array([[-hw, -hh], [0, -hh], [0, 0], [-hw, 0]], dtype=np.float64),
        np.array([[0, -hh], [hw, -hh], [hw, 0], [0, 0]], dtype=np.float64),
        np.array([[-hw, 0], [0, 0], [0, hh], [-hw, hh]], dtype=np.float64),
        np.array([[0, 0], [hw, 0], [hw, hh], [0, hh]], dtype=np.float64),
    ]

    pieces = []
    for i, v in enumerate(rects_mm):
        c = np.mean(v, axis=0)
        v_local = v - c
        edges = np.roll(v_local, -1, axis=0) - v_local
        lens = np.linalg.norm(edges, axis=1)
        area = abs(polygon_area_signed(v_local))
        pieces.append(Piece(i+1, v_local, lens, area, (0,0)))

    # 3. 求解
    import time
    t0 = time.time()
    placements, score, nodes, ok = solve_puzzle(pieces, config)
    t1 = time.time()

    print(f"=== 求解结果 ===")
    print(f"耗时: {(t1-t0)*1000:.1f} ms, 节点: {nodes}, 得分: {score:.6f}, 成功: {ok}")

    if ok:
        # 4. 机械臂指令
        cmds = placements_to_robot_commands(placements)
        print(f"机械臂指令: {json.dumps(cmds, indent=2, ensure_ascii=False)}")

        # 5. 验证
        piece_dict = {p.number: p for p in pieces}
        val = validate_assembly(placements, piece_dict, config)
        print(f"验证: {json.dumps(val, indent=2, ensure_ascii=False)}")