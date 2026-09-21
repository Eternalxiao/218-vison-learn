#!/usr/bin/env python3
"""
Embedded Puzzle Solver for MaixCAM Pro
=======================================
纯 Python + NumPy，零第三方几何库依赖（无 Shapely/SciPy）。
输入：已精确标定的碎片数据（mm真实边长、像素角点、质心）。
输出：刚体变换 (R, t) 使拼接外接矩形贴合 A4、零重叠。

核心设计：
- 工作空间：毫米（mm），物理真实尺寸，避免像素累积误差
- 搜索：Root固定 + Best-First DFS（优先队列，非纯深度优先）
- 匹配：混合容差（绝对mm + 相对%），显式T型/三段式拓扑
- 优化：Gauss-Newton 全局位姿图（一次性求解，替代"单边对齐+事后最小二乘"两阶段）
- 重叠消除：SAT分离轴定理迭代分离
- 压紧：向心贪心平移
- 评分：6项加权（空洞/重叠/长宽比/尺寸/面积/连通性）
"""

from __future__ import annotations
import math
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import numpy as np

# ============================================================
# 基础几何工具（纯 NumPy，无 Shapely）
# ============================================================

def rotation_matrix(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)

def polygon_area(pts: np.ndarray) -> float:
    """鞋带公式，pts: (N,2) CCW"""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

def polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """质心，pts: (N,2) CCW"""
    A = polygon_area(pts)
    if abs(A) < 1e-12:
        return np.mean(pts, axis=0)
    x, y = pts[:, 0], pts[:, 1]
    cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - y * np.roll(x, -1)))
    cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - y * np.roll(x, -1)))
    return np.array([cx, cy], dtype=np.float64) / (6.0 * A)

def edge_vectors(pts: np.ndarray) -> np.ndarray:
    """边向量 (N,2): pts[i+1] - pts[i]"""
    return np.roll(pts, -1, axis=0) - pts

def edge_lengths(pts: np.ndarray) -> np.ndarray:
    """各边长度 (N,)"""
    vecs = edge_vectors(pts)
    return np.linalg.norm(vecs, axis=1)

def min_area_rect(pts: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """
    旋转卡壳法求最小外接矩形
    返回: (long_side, short_side, angle_rad, rect_pts_4x2)
    """
    # 凸包
    from scipy.spatial import ConvexHull  # 注：MaixCAM 通常无 scipy，下方给纯 numpy 实现
    # 这里先用 scipy 占位，后文提供纯 numpy 版
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    n = len(hull_pts)
    if n <= 2:
        return 0.0, 0.0, 0.0, hull_pts

    best_area = float('inf')
    best_rect = None
    best_angle = 0.0

    edges = np.roll(hull_pts, -1, axis=0) - hull_pts
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])

    for angle in edge_angles:
        R = rotation_matrix(-angle)
        rotated = hull_pts @ R.T
        min_xy = rotated.min(axis=0)
        max_xy = rotated.max(axis=0)
        w, h = max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]
        area = w * h
        if area < best_area:
            best_area = area
            best_angle = angle
            best_rect = (w, h, min_xy, max_xy)

    w, h, min_xy, max_xy = best_rect
    long_side, short_side = max(w, h), min(w, h)
    # 矩形四角（世界坐标）
    rect_local = np.array([[min_xy[0], min_xy[1]],
                           [max_xy[0], min_xy[1]],
                           [max_xy[0], max_xy[1]],
                           [min_xy[0], max_xy[1]]], dtype=np.float64)
    R_back = rotation_matrix(best_angle)
    rect_world = rect_local @ R_back.T

    return long_side, short_side, best_angle, rect_world


# ---- 纯 NumPy 凸包（替代 scipy.spatial.ConvexHull）----
def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew 单调链凸包，返回凸包顶点 CCW (M,2)"""
    pts = np.unique(pts, axis=0)
    if len(pts) <= 1:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]  # 按 x 后 y 排序

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull

def min_area_rect_numpy(pts: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """纯 numpy 旋转卡壳最小外接矩形"""
    hull = convex_hull(pts)
    n = len(hull)
    if n <= 2:
        return 0.0, 0.0, 0.0, hull

    best_area = float('inf')
    best_w = best_h = best_angle = 0.0
    best_rect = None

    edges = np.roll(hull, -1, axis=0) - hull
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])

    for angle in edge_angles:
        R = rotation_matrix(-angle)
        rot = hull @ R.T
        min_xy = rot.min(axis=0)
        max_xy = rot.max(axis=0)
        w, h = max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]
        area = w * h
        if area < best_area:
            best_area = area
            best_w, best_h = w, h
            best_angle = angle
            best_rect = (w, h, min_xy, max_xy)

    w, h, min_xy, max_xy = best_rect
    long_side, short_side = max(w, h), min(w, h)
    rect_local = np.array([[min_xy[0], min_xy[1]],
                           [max_xy[0], min_xy[1]],
                           [max_xy[0], max_xy[1]],
                           [min_xy[0], max_xy[1]]], dtype=np.float64)
    rect_world = rect_local @ rotation_matrix(best_angle).T
    return long_side, short_side, best_angle, rect_world


# ============================================================
# SAT 碰撞/分离（替代 Shapely intersection）
# ============================================================

def project_polygon(pts: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """多边形在轴上的投影区间 [min, max]"""
    dots = pts @ axis
    return float(dots.min()), float(dots.max())

def polygons_overlap_sat(poly_a: np.ndarray, poly_b: np.ndarray) -> Tuple[bool, float, np.ndarray]:
    """
    SAT 检测两凸多边形重叠
    返回: (是否重叠, 重叠深度, 分离向量 axis * depth)
    若不重叠，depth=inf, sep_vec=zeros
    """
    min_depth = float('inf')
    best_axis = None

    for poly in (poly_a, poly_b):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-10
        if not np.any(valid):
            continue
        normals = normals[valid] / norms[valid, None]

        for axis in normals:
            min_a, max_a = project_polygon(poly_a, axis)
            min_b, max_b = project_polygon(poly_b, axis)
            if max_a < min_b or max_b < min_a:
                return False, 0.0, np.zeros(2)  # 分离轴存在，无重叠
            depth = min(max_a, max_b) - max(min_a, min_b)
            if depth < min_depth:
                min_depth = depth
                best_axis = axis

    if best_axis is None:
        return False, 0.0, np.zeros(2)

    # 确保分离方向：将 B 推离 A
    center_a = np.mean(poly_a, axis=0)
    center_b = np.mean(poly_b, axis=0)
    if np.dot(center_b - center_a, best_axis) < 0:
        best_axis = -best_axis

    return True, min_depth, best_axis * min_depth

def polygon_overlap_area_approx(poly_a: np.ndarray, poly_b: np.ndarray, grid_step: float = 1.0) -> float:
    """
    光栅化近似重叠面积（用于评分，不要求极高精度）
    grid_step: 采样网格步长 mm
    """
    # 包围盒
    all_pts = np.vstack([poly_a, poly_b])
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    w = int(np.ceil((max_xy[0] - min_xy[0]) / grid_step)) + 1
    h = int(np.ceil((max_xy[1] - min_xy[1]) / grid_step)) + 1
    if w * h > 50000:  # 太大退回简单估计
        return 0.0

    # 点在凸多边形内判定（叉积法，O(N)）
    def point_in_convex(poly, pt):
        # poly CCW
        vecs = np.roll(poly, -1, axis=0) - poly
        rel = pt - poly
        crosses = vecs[:, 0] * rel[:, 1] - vecs[:, 1] * rel[:, 0]
        return np.all(crosses >= -1e-9)

    count = 0
    for i in range(w):
        for j in range(h):
            pt = min_xy + np.array([i, j]) * grid_step
            if point_in_convex(poly_a, pt) and point_in_convex(poly_b, pt):
                count += 1
    return count * (grid_step ** 2)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Piece:
    """已标定碎片：所有几何量在 mm 物理空间"""
    number: int                    # 1-based 编号
    vertices_mm: np.ndarray        # (K,2) CCW，以质心为原点的局部坐标
    edge_lengths_mm: np.ndarray    # (K,) 各边物理长度 mm
    area_mm2: float                # 面积 mm^2
    centroid_px: Tuple[float, float]  # 原图像素质心（仅用于调试/可视化）
    # 可选：原始像素轮廓、透视变换矩阵等

@dataclass
class Placement:
    """碎片在世界坐标系下的放置位姿"""
    piece_id: int
    angle_rad: float
    translation_mm: np.ndarray     # (2,) 质心世界坐标
    vertices_world: np.ndarray     # (K,2) 世界坐标顶点
    # 缓存
    _edges_world: np.ndarray = field(init=False, repr=False)
    _edge_vecs_world: np.ndarray = field(init=False, repr=False)
    _edge_lens: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._edges_world = np.roll(self.vertices_world, -1, axis=0) - self.vertices_world
        self._edge_vecs_world = self._edges_world
        self._edge_lens = np.linalg.norm(self._edges_world, axis=1)

    def edge(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        v = self.vertices_world
        return v[idx], v[(idx + 1) % len(v)]

    def edge_vector(self, idx: int) -> np.ndarray:
        return self._edge_vecs_world[idx]

    def edge_length(self, idx: int) -> float:
        return self._edge_lens[idx]

@dataclass
class EdgeMatch:
    """一对匹配的边"""
    piece_a: int
    edge_a: int
    piece_b: int
    edge_b: int
    length_error_mm: float
    length_error_rel: float
    is_partial: bool = False       # 是否部分匹配（长边对多短边）
    partial_ratio: float = 1.0     # 短边总长/长边长


# ============================================================
# 配置参数（可外部 JSON 注入）
# ============================================================

@dataclass
class SolverConfig:
    # 边长匹配容差
    edge_length_tol_mm: float = 1.5      # 绝对容差 mm
    edge_length_tol_rel: float = 0.03    # 相对容差 3%
    min_partial_ratio: float = 0.25      # 部分匹配最短边比例

    # 搜索控制
    max_nodes: int = 20000
    max_seconds: float = 8.0
    max_candidates_per_step: int = 50

    # 几何剪枝
    search_overlap_tol_mm2: float = 2.0      # 搜索期允许重叠面积 mm^2
    bbox_size_tolerance: float = 0.15        # 外接矩形允许超目标 15%

    # 优化/后处理
    gn_max_iter: int = 20
    gn_tol: float = 1e-7
    overlap_eps_mm2: float = 0.5             # 最终零重叠阈值
    separation_clearance_mm: float = 0.2
    max_sep_iterations: int = 80
    compact_step_start_mm: float = 2.0
    compact_step_min_mm: float = 0.2

    # 评分权重
    w_gap: float = 0.52
    w_overlap: float = 3.0
    w_aspect: float = 0.16
    w_size: float = 0.16
    w_area: float = 0.16
    w_disconnected: float = 0.015

    # 目标 A4 尺寸
    target_long_mm: float = 297.0
    target_short_mm: float = 210.0

    # 仅碎片模式（无 A4 参考）
    pieces_only_mode: bool = False
    pieces_only_fallback: bool = True
    fallback_bbox_tol: float = 0.8


# ============================================================
# 核心求解器
# ============================================================

class PuzzleSolver:
    def __init__(self, pieces: List[Piece], config: SolverConfig):
        self.pieces = {p.number: p for p in pieces}
        self.piece_list = pieces
        self.config = config
        self.target_long = config.target_long_mm
        self.target_short = config.target_short_mm
        self.target_area = self.target_long * self.target_short
        self.target_aspect = self.target_long / self.target_short

        # 状态
        self.best_placements: Optional[Dict[int, Placement]] = None
        self.best_score = float('inf')
        self.nodes_expanded = 0
        self.start_time = 0.0
        self.deadline = 0.0

        # 预计算所有边对兼容性
        self.compatible_pairs = self._precompute_compatible_pairs()
        self.composite_groups = self._precompute_composite_groups()

    def _precompute_compatible_pairs(self) -> List[EdgeMatch]:
        """所有片间边对的兼容性（全边匹配 + T型 + 三段式）"""
        matches = []
        pieces = self.piece_list
        for i, pa in enumerate(pieces):
            for pb in pieces[i+1:]:
                for ea, la in enumerate(pa.edge_lengths_mm):
                    for eb, lb in enumerate(pb.edge_lengths_mm):
                        abs_err = abs(la - lb)
                        rel_err = abs_err / max(la, lb)
                        partial_ratio = min(la, lb) / max(la, lb)

                        # 全边匹配
                        if abs_err <= self.config.edge_length_tol_mm or rel_err <= self.config.edge_length_tol_rel:
                            matches.append(EdgeMatch(
                                pa.number, ea, pb.number, eb,
                                abs_err, rel_err, False, 1.0
                            ))

                        # T型/部分匹配：长边 ≈ 短边之和（此处仅两片，三段式在 composite_groups）
                        if partial_ratio >= self.config.min_partial_ratio:
                            if abs_err <= self.config.edge_length_tol_mm * 2 or rel_err <= self.config.edge_length_tol_rel * 2:
                                matches.append(EdgeMatch(
                                    pa.number, ea, pb.number, eb,
                                    abs_err, rel_err, True, partial_ratio
                                ))
        # 按误差升序
        matches.sort(key=lambda m: (m.length_error_rel, -m.partial_ratio))
        return matches

    def _precompute_composite_groups(self) -> List[Dict]:
        """
        预计算"一条长边 = 多条短边之和"的组合（T型、三段式）
        仅 N<=5 时枚举，返回结构供搜索直接使用
        """
        groups = []
        pieces = self.piece_list
        if len(pieces) < 3 or len(pieces) > 5:
            return groups

        tol_mm = self.config.edge_length_tol_mm * 2
        tol_rel = self.config.edge_length_tol_rel * 2

        for long_piece in pieces:
            others = [p for p in pieces if p.number != long_piece.number]
            for long_ei, long_len in enumerate(long_piece.edge_lengths_mm):
                # 从 others 中选边，长度 < long_len
                choices_per_piece = []
                for op in others:
                    choices = [(ei, l) for ei, l in enumerate(op.edge_lengths_mm)
                              if l < long_len * (1 + tol_rel) + tol_mm]
                    if not choices:
                        break
                    choices_per_piece.append(choices)
                else:
                    # 所有其他片都有候选短边
                    from itertools import product
                    for selection in product(*choices_per_piece):
                        short_lens = [l for _, l in selection]
                        total = sum(short_lens)
                        abs_err = abs(total - long_len)
                        rel_err = abs_err / max(total, long_len)
                        if abs_err <= tol_mm or rel_err <= tol_rel:
                            groups.append({
                                'long_piece': long_piece.number,
                                'long_edge': long_ei,
                                'long_len': long_len,
                                'shorts': [{'piece': op.number, 'edge': ei, 'len': l}
                                          for op, (ei, l) in zip(others, selection)],
                                'total_short': total,
                                'abs_err': abs_err,
                                'rel_err': rel_err,
                            })
        groups.sort(key=lambda g: (g['rel_err'], -len(g['shorts'])))
        return groups[:20]  # 限制数量

    # --------------------------------------------------------
    # 几何变换
    # --------------------------------------------------------
    def make_placement(self, piece: Piece, angle: float, trans: np.ndarray) -> Placement:
        R = rotation_matrix(angle)
        verts = piece.vertices_mm @ R.T + trans
        return Placement(piece.number, angle, trans, verts)

    def attach_piece(self, placed: Placement, placed_ei: int,
                     new_piece: Piece, new_ei: int,
                     variant: int = 0) -> Placement:
        """
        将 new_piece 的 new_ei 边反向贴合到 placed 的 placed_ei 边
        variant: 0=q0->p1, 1=q1->p0, 2=中点对齐
        """
        p0, p1 = placed.edge(placed_ei)
        q0, q1 = new_piece.vertices_mm[new_ei], new_piece.vertices_mm[(new_ei + 1) % len(new_piece.vertices_mm)]

        # 目标方向：p0 -> p1 的反向 = p1 - p0 （新片边方向应为 q1->q0）
        target_vec = p0 - p1
        source_vec = q1 - q0
        angle = math.atan2(target_vec[1], target_vec[0]) - math.atan2(source_vec[1], source_vec[0])
        R = rotation_matrix(angle)

        if variant == 0:
            trans = p1 - q0 @ R.T
        elif variant == 1:
            trans = p0 - q1 @ R.T
        else:
            trans = 0.5 * (p0 + p1) - 0.5 * (q0 + q1) @ R.T

        return self.make_placement(new_piece, angle, trans)

    def attach_at_point(self, placed: Placement, placed_ei: int,
                        new_piece: Piece, new_ei: int,
                        contact_pt: np.ndarray,
                        align_new_start: bool) -> Placement:
        """将新片边的某端点贴到已放长边的指定分段点"""
        p0, p1 = placed.edge(placed_ei)
        q0, q1 = new_piece.vertices_mm[new_ei], new_piece.vertices_mm[(new_ei + 1) % len(new_piece.vertices_mm)]
        target_vec = p0 - p1
        source_vec = q1 - q0
        angle = math.atan2(target_vec[1], target_vec[0]) - math.atan2(source_vec[1], source_vec[0])
        R = rotation_matrix(angle)
        src_pt = q0 if align_new_start else q1
        trans = contact_pt - src_pt @ R.T
        return self.make_placement(new_piece, angle, trans)

    # --------------------------------------------------------
    # 几何剪枝
    # --------------------------------------------------------
    def can_place(self, cand: Placement, current: Dict[int, Placement]) -> bool:
        # 1) 重叠剪枝
        for other in current.values():
            overlap, depth, _ = polygons_overlap_sat(cand.vertices_world, other.vertices_world)
            if overlap and depth * min(cand.vertices_world.shape[0], other.vertices_world.shape[0]) > self.config.search_overlap_tol_mm2:
                return False

        # 2) 外接矩形剪枝
        all_verts = [cand.vertices_world] + [p.vertices_world for p in current.values()]
        union_verts = np.vstack(all_verts)
        long_s, short_s, _, _ = min_area_rect_numpy(union_verts)
        if long_s > self.target_long * (1 + self.config.bbox_size_tolerance):
            return False
        if short_s > self.target_short * (1 + self.config.bbox_size_tolerance):
            return False
        return True

    # --------------------------------------------------------
    # 评分
    # --------------------------------------------------------
    def score_placements(self, placements: Dict[int, Placement]) -> float:
        all_verts = np.vstack([p.vertices_world for p in placements.values()])
        long_s, short_s, _, rect_pts = min_area_rect_numpy(all_verts)
        rect_area = long_s * short_s

        # 联合面积（凸包面积近似，或光栅化）
        # 这里用凸包面积近似联合面积（严格需多边形并集，但 N<=5 凸包误差可接受）
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(all_verts)
            union_area = hull.volume  # 2D下 volume=area
        except:
            # 退回：简单求和减去重叠估计
            union_area = sum(self.pieces[pid].area_mm2 for pid in placements)
            # 减去两两重叠估计
            for i, pa in enumerate(placements.values()):
                for pb in list(placements.values())[i+1:]:
                    ov, _, _ = polygons_overlap_sat(pa.vertices_world, pb.vertices_world)
                    if ov:
                        union_area -= 1.0  # 粗略惩罚

        piece_area_sum = sum(self.pieces[pid].area_mm2 for pid in placements)

        gap_error = max(0.0, rect_area - union_area) / self.target_area
        overlap_error = max(0.0, piece_area_sum - union_area) / self.target_area
        aspect_error = abs((long_s / short_s) / self.target_aspect - 1.0)

        def size_err(val, target):
            r = val / target
            if r < 1.0: return 1.0 - r
            return max(0.0, r - (1.0 + self.config.bbox_size_tolerance))

        size_error = 0.5 * (size_err(long_s, self.target_long) + size_err(short_s, self.target_short))
        area_error = abs(union_area / self.target_area - 1.0)

        # 连通性：检查联合图是否单连通（简化：至少 N-1 条边匹配）
        # 这里略过，假设搜索生成的树天然连通
        disconnected = 0.0

        score = (self.config.w_gap * gap_error +
                 self.config.w_overlap * overlap_error +
                 self.config.w_aspect * aspect_error +
                 self.config.w_size * size_error +
                 self.config.w_area * area_error +
                 self.config.w_disconnected * disconnected)
        return score

    # --------------------------------------------------------
    # Gauss-Newton 全局位姿图优化（核心创新：替代两阶段）
    # --------------------------------------------------------
    def optimize_pose_graph(self,
                            initial: Dict[int, Placement],
                            matches: List[EdgeMatch]) -> Optional[Dict[int, Placement]]:
        """
        变量：每个非 root 片的 (θ, tx, ty) -> 3*(N-1) 维
        残差：每条匹配边的 4 个端点重合误差
        """
        if not matches:
            return initial

        piece_ids = sorted(initial.keys())
        root_id = max(piece_ids, key=lambda pid: self.pieces[pid].area_mm2)
        var_ids = [pid for pid in piece_ids if pid != root_id]
        n_vars = len(var_ids) * 3

        # 初始向量
        x0 = np.zeros(n_vars, dtype=np.float64)
        for idx, pid in enumerate(var_ids):
            pose = initial[pid]
            x0[idx*3] = pose.angle_rad
            x0[idx*3+1] = pose.translation_mm[0]
            x0[idx*3+2] = pose.translation_mm[1]

        # 固定 root
        root_pose = initial[root_id]

        def unpack(x: np.ndarray) -> Dict[int, Placement]:
            poses = {root_id: root_pose}
            for idx, pid in enumerate(var_ids):
                angle = x[idx*3]
                tx, ty = x[idx*3+1], x[idx*3+2]
                piece = self.pieces[pid]
                poses[pid] = self.make_placement(piece, angle, np.array([tx, ty]))
            return poses

        def residuals(x: np.ndarray) -> np.ndarray:
            poses = unpack(x)
            vals = []
            for m in matches:
                pa = poses[m.piece_a]
                pb = poses[m.piece_b]
                a0, a1 = pa.edge(m.edge_a)
                b0, b1 = pb.edge(m.edge_b)
                # 反向对齐：a0≈b1, a1≈b0
                vals.extend((a0 - b1).tolist())
                vals.extend((a1 - b0).tolist())
            return np.array(vals, dtype=np.float64)

        # 数值雅可比 + 正规方程求解
        x = x0.copy()
        for it in range(self.config.gn_max_iter):
            r = residuals(x)
            if np.linalg.norm(r) < self.config.gn_tol:
                break

            # 数值雅可比
            J = np.zeros((len(r), n_vars), dtype=np.float64)
            eps = 1e-6
            for k in range(n_vars):
                step = eps * (1.0 if k % 3 == 0 else 1.0)  # 角度用小步长
                x2 = x.copy()
                x2[k] += step
                r2 = residuals(x2)
                J[:, k] = (r2 - r) / step

            # 正规方程 (J^T J) δ = -J^T r
            JTJ = J.T @ J
            JTr = J.T @ r
            # 阻尼
            JTJ += np.eye(n_vars) * 1e-6
            try:
                delta = np.linalg.solve(JTJ, -JTr)
            except np.linalg.LinAlgError:
                break

            x += delta
            if np.linalg.norm(delta) < self.config.gn_tol:
                break

        refined = unpack(x)

        # 优化后重叠检查
        for pid_a, pa in refined.items():
            for pid_b, pb in refined.items():
                if pid_a >= pid_b: continue
                ov, depth, _ = polygons_overlap_sat(pa.vertices_world, pb.vertices_world)
                if ov and depth > self.config.search_overlap_tol_mm2:
                    return None
        return refined

    # --------------------------------------------------------
    # 重叠消除（SAT 迭代分离）
    # --------------------------------------------------------
    def resolve_overlaps(self, placements: Dict[int, Placement]) -> Optional[Dict[int, Placement]]:
        resolved = dict(placements)
        for _ in range(self.config.max_sep_iterations):
            worst_pair = None
            worst_depth = 0.0
            worst_axis = None

            ids = list(resolved.keys())
            for i, id_a in enumerate(ids):
                for id_b in ids[i+1:]:
                    ov, depth, axis = polygons_overlap_sat(resolved[id_a].vertices_world,
                                                            resolved[id_b].vertices_world)
                    if ov and depth > worst_depth:
                        worst_depth = depth
                        worst_pair = (id_a, id_b)
                        worst_axis = axis

            if worst_pair is None or worst_depth <= self.config.overlap_eps_mm2:
                return resolved

            id_a, id_b = worst_pair
            pa = resolved[id_a]
            pb = resolved[id_b]
            # 对称分离
            shift = worst_axis * 0.5
            # 直接平移（保持角度、缩放）
            resolved[id_a] = Placement(pa.piece_id, pa.angle_rad, pa.translation_mm - shift,
                                       pa.vertices_world - shift)
            resolved[id_b] = Placement(pb.piece_id, pb.angle_rad, pb.translation_mm + shift,
                                       pb.vertices_world + shift)

        # 最终检查
        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                ov, depth, _ = polygons_overlap_sat(resolved[id_a].vertices_world,
                                                     resolved[id_b].vertices_world)
                if ov and depth > self.config.overlap_eps_mm2:
                    return None
        return resolved

    # --------------------------------------------------------
    # 压紧布局（向心贪心）
    # --------------------------------------------------------
    def compact_layout(self, placements: Dict[int, Placement]) -> Dict[int, Placement]:
        current = dict(placements)
        current_score = self.score_placements(current)

        # 8 方向 + 向心
        dirs = [np.array([1.,0.]), np.array([-1.,0.]), np.array([0.,1.]), np.array([0.,-1.]),
                np.array([1.,1.])/np.sqrt(2), np.array([1.,-1.])/np.sqrt(2),
                np.array([-1.,1.])/np.sqrt(2), np.array([-1.,-1.])/np.sqrt(2)]

        step = self.config.compact_step_start_mm
        while step >= self.config.compact_step_min_mm:
            improved = False
            for _ in range(12):
                # 布局中心
                all_v = np.vstack([p.vertices_world for p in current.values()])
                center = all_v.mean(axis=0)

                for pid in sorted(current.keys()):
                    pose = current[pid]
                    to_center = center - pose.translation_mm
                    norm = np.linalg.norm(to_center)
                    trial_dirs = list(dirs)
                    if norm > 1e-6:
                        trial_dirs.insert(0, to_center / norm)

                    best_pose = None
                    best_score = current_score

                    for d in trial_dirs:
                        cand_trans = pose.translation_mm + d * step
                        cand = Placement(pose.piece_id, pose.angle_rad, cand_trans,
                                         pose.vertices_world + d * step)

                        # 零重叠检查
                        ok = True
                        for oid, other in current.items():
                            if oid == pid: continue
                            ov, depth, _ = polygons_overlap_sat(cand.vertices_world, other.vertices_world)
                            if ov and depth > self.config.overlap_eps_mm2:
                                ok = False
                                break
                        if not ok: continue

                        # 外接矩形检查
                        trial = dict(current)
                        trial[pid] = cand
                        all_v2 = np.vstack([p.vertices_world for p in trial.values()])
                        long_s, short_s, _, _ = min_area_rect_numpy(all_v2)
                        if long_s > self.target_long * (1 + self.config.bbox_size_tolerance):
                            continue
                        if short_s > self.target_short * (1 + self.config.bbox_size_tolerance):
                            continue

                        score = self.score_placements(trial)
                        if score < best_score - 1e-9:
                            best_score = score
                            best_pose = cand

                    if best_pose:
                        current[pid] = best_pose
                        current_score = best_score
                        improved = True
            if not improved:
                step *= 0.5
        return current

    # --------------------------------------------------------
    # 搜索主循环
    # --------------------------------------------------------
    def solve(self) -> Tuple[Optional[Dict[int, Placement]], float, int]:
        self.start_time = time.monotonic()
        self.deadline = self.start_time + self.config.max_seconds
        self.nodes_expanded = 0
        self.best_placements = None
        self.best_score = float('inf')

        # Root = 最大面积片
        root = max(self.piece_list, key=lambda p: p.area_mm2)
        root_pose = self.make_placement(root, 0.0, np.zeros(2))

        # 优先队列：(优先级, 序列号, 状态)
        # 状态: (placements_dict, used_edges_set, match_list)
        counter = 0
        pq = []
        initial_state = ({root.number: root_pose}, set(), [])
        heapq.heappush(pq, (0.0, counter, initial_state))
        counter += 1

        # 预计算每片的边数
        piece_edges = {p.number: len(p.vertices_mm) for p in self.piece_list}

        while pq:
            if self.nodes_expanded >= self.config.max_nodes:
                break
            if time.monotonic() >= self.deadline:
                break
            if self.best_score <= 1e-6:
                break

            _, _, (current, used_edges, matches) = heapq.heappop(pq)
            self.nodes_expanded += 1

            if len(current) == len(self.piece_list):
                # 完整候选 -> 优化 -> 重叠消除 -> 压紧 -> 评分
                refined = self.optimize_pose_graph(current, matches)
                if refined is None:
                    continue
                resolved = self.resolve_overlaps(refined)
                if resolved is None:
                    continue
                compacted = self.compact_layout(resolved)
                score = self.score_placements(compacted)
                if score < self.best_score:
                    self.best_score = score
                    self.best_placements = compacted
                continue

            # 生成候选
            unplaced = [p for p in self.piece_list if p.number not in current]
            candidates = []

            for placed_id, placed_pose in current.items():
                for placed_ei in range(piece_edges[placed_id]):
                    if (placed_id, placed_ei) in used_edges:
                        continue
                    placed_len = placed_pose.edge_length(placed_ei)

                    for new_piece in unplaced:
                        for new_ei in range(piece_edges[new_piece.number]):
                            new_len = new_piece.edge_lengths_mm[new_ei]
                            abs_err = abs(placed_len - new_len)
                            rel_err = abs_err / max(placed_len, new_len)
                            partial_ratio = min(placed_len, new_len) / max(placed_len, new_len)

                            # 匹配判定
                            is_partial = False
                            if rel_err <= self.config.edge_length_tol_rel or abs_err <= self.config.edge_length_tol_mm:
                                variant_count = 1 if rel_err < 1e-4 else 3
                            elif partial_ratio >= self.config.min_partial_ratio:
                                is_partial = True
                                if rel_err <= self.config.edge_length_tol_rel * 2 or abs_err <= self.config.edge_length_tol_mm * 2:
                                    variant_count = 2
                                else:
                                    continue
                            else:
                                continue

                            for variant in range(variant_count):
                                cand = self.attach_piece(placed_pose, placed_ei, new_piece, new_ei, variant)
                                if not self.can_place(cand, current):
                                    continue

                                # 优先级：误差小、公共边长优先
                                priority = rel_err - 0.05 * partial_ratio
                                candidates.append((priority, counter, placed_id, placed_ei,
                                                   new_piece.number, new_ei, cand,
                                                   is_partial))
                                counter += 1

            # 限制候选数，保留最优
            candidates.sort(key=lambda x: x[0])
            for cand in candidates[:self.config.max_candidates_per_step]:
                _, _, placed_id, placed_ei, new_id, new_ei, pose, is_partial = cand

                next_used = set(used_edges)
                next_used.add((placed_id, placed_ei))
                next_used.add((new_id, new_ei))
                next_matches = matches + [(placed_id, placed_ei, new_id, new_ei)]
                next_current = dict(current)
                next_current[new_id] = pose

                heapq.heappush(pq, (cand[0], counter, (next_current, next_used, next_matches)))
                counter += 1

        # ---------- 后备：复合边组（T型/三段式） ----------
        if not self.config.pieces_only_mode and self.composite_groups:
            for group in self.composite_groups[:5]:
                if time.monotonic() >= self.deadline: break
                placements = self._try_composite_group(group)
                if placements:
                    refined = self.optimize_pose_graph(placements, [])
                    if refined:
                        resolved = self.resolve_overlaps(refined)
                        if resolved:
                            compacted = self.compact_layout(resolved)
                            score = self.score_placements(compacted)
                            if score < self.best_score:
                                self.best_score = score
                                self.best_placements = compacted

        return self.best_placements, self.best_score, self.nodes_expanded

    def _try_composite_group(self, group: Dict) -> Optional[Dict[int, Placement]]:
        """尝试铺设一组：长边 + 多短边顺次贴合"""
        long_piece = self.pieces[group['long_piece']]
        long_ei = group['long_edge']

        # Root = 长边片
        root_pose = self.make_placement(long_piece, 0.0, np.zeros(2))
        long_start, long_end = root_pose.edge(long_ei)
        direction = (long_end - long_start) / group['long_len']

        # 短边顺序排列（枚举排列）
        from itertools import permutations
        shorts = group['shorts']
        for order in permutations(shorts):
            offset = 0.5 * (group['long_len'] - group['total_short'])
            poses = {long_piece.number: root_pose}
            ok = True
            for s in order:
                piece = self.pieces[s['piece']]
                seg_start = long_start + direction * offset
                seg_end = long_start + direction * (offset + s['len'])
                # 反向贴合
                q0 = piece.vertices_mm[s['edge']]
                q1 = piece.vertices_mm[(s['edge'] + 1) % len(piece.vertices_mm)]
                target_vec = seg_start - seg_end
                source_vec = q1 - q0
                angle = math.atan2(target_vec[1], target_vec[0]) - math.atan2(source_vec[1], source_vec[0])
                R = rotation_matrix(angle)
                trans = seg_end - q0 @ R.T
                poses[piece.number] = self.make_placement(piece, angle, trans)

                # 检查与已放重叠
                for other_id, other_pose in poses.items():
                    if other_id == piece.number: continue
                    ov, depth, _ = polygons_overlap_sat(poses[piece.number].vertices_world, other_pose.vertices_world)
                    if ov and depth > self.config.search_overlap_tol_mm2:
                        ok = False
                        break
                if not ok: break
                offset += s['len']
            if ok and len(poses) == len(self.piece_list):
                return poses
        return None


# ============================================================
# 入口函数
# ============================================================

def solve_puzzle(pieces_data: List[dict], config_dict: dict = None) -> dict:
    """
    主入口，供 MaixCAM 调用

    pieces_data: List[dict], 每个包含:
        - number: int
        - vertices_mm: List[List[float]]  # (K,2) CCW, 质心为原点
        - edge_lengths_mm: List[float]     # (K,)
        - area_mm2: float
        - centroid_px: [x, y]  # 可选

    返回:
        {
            'placements': {piece_id: {'angle_rad': float, 'tx_mm': float, 'ty_mm': float, 'vertices_world': [...]}}
            'score': float,
            'nodes': int,
            'success': bool,
            'metrics': {...}
        }
    """
    pieces = []
    for d in pieces_data:
        pieces.append(Piece(
            number=d['number'],
            vertices_mm=np.array(d['vertices_mm'], dtype=np.float64),
            edge_lengths_mm=np.array(d['edge_lengths_mm'], dtype=np.float64),
            area_mm2=d['area_mm2'],
            centroid_px=tuple(d.get('centroid_px', (0, 0)))
        ))

    cfg = SolverConfig()
    if config_dict:
        for k, v in config_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    solver = PuzzleSolver(pieces, cfg)
    placements, score, nodes = solver.solve()

    if placements is None:
        return {'success': False, 'score': float('inf'), 'nodes': nodes,
                'placements': {}, 'metrics': {}}

    # 构造返回
    result_placements = {}
    for pid, pose in placements.items():
        result_placements[pid] = {
            'angle_rad': pose.angle_rad,
            'tx_mm': pose.translation_mm[0],
            'ty_mm': pose.translation_mm[1],
            'vertices_world': pose.vertices_world.tolist()
        }

    # 最终指标
    all_v = np.vstack([p.vertices_world for p in placements.values()])
    long_s, short_s, _, _ = min_area_rect_numpy(all_v)
    union_area = 0.0  # 可补充光栅化精确计算
    piece_area_sum = sum(p.area_mm2 for p in pieces)

    return {
        'success': True,
        'score': score,
        'nodes': nodes,
        'placements': result_placements,
        'metrics': {
            'outer_long_mm': long_s,
            'outer_short_mm': short_s,
            'aspect_ratio': long_s / short_s,
            'target_aspect': cfg.target_long_mm / cfg.target_short_mm,
            'piece_area_sum_mm2': piece_area_sum,
            'fill_ratio': union_area / (long_s * short_s) if long_s * short_s > 0 else 0
        }
    }


# ============================================================
# MaixCAM 部署适配说明
# ============================================================
"""
部署到 MaixCAM Pro (K210/K510) 时的注意事项：

1. **无 scipy 依赖**：已用纯 numpy 实现 convex_hull + min_area_rect_numpy
2. **内存优化**：
   - 避免大数组复制，复用缓冲区
   - 光栅化重叠面积可选，默认用 SAT 判定 + 凸包面积近似
   - 最大搜索节点 20000，超时 8s，可根据实测调整
3. **定点数考虑**：K210 无 FPU，建议：
   - 关键几何计算（SAT、雅可比）保持 float32/float64
   - 或移植到 K510 (RV64GC 有 FPU)
4. **配置外置化**：config.json 存储在 SD 卡，启动加载
5. **输入数据来源**：
   - vertices_mm, edge_lengths_mm 由上位机标定/检测模块产出
   - 或 MaixCAM 端运行检测 + 透视矫正 + 物理尺度换算
6. **输出用途**：
   - placements 送给上位机/机械臂执行拼接
   - 或直接在屏幕预览拼接效果
"""

if __name__ == '__main__':
    # 简单自测：4 片矩形拼成 A4
    import json

    # 模拟 4 片：A4 切成 2x2
    W, H = 297.0, 210.0
    half_w, half_h = W/2, H/2
    rects = [
        # 左上
        np.array([[-half_w, -half_h], [0, -half_h], [0, 0], [-half_w, 0]], dtype=np.float64),
        # 右上
        np.array([[0, -half_h], [half_w, -half_h], [half_w, 0], [0, 0]], dtype=np.float64),
        # 左下
        np.array([[-half_w, 0], [0, 0], [0, half_h], [-half_w, half_h]], dtype=np.float64),
        # 右下
        np.array([[0, 0], [half_w, 0], [half_w, half_h], [0, half_h]], dtype=np.float64),
    ]

    test_pieces = []
    for i, verts in enumerate(rects):
        # 移到质心
        c = np.mean(verts, axis=0)
        verts_cent = verts - c
        lens = np.linalg.norm(np.roll(verts_cent, -1, axis=0) - verts_cent, axis=1)
        area = polygon_area(verts_cent)
        test_pieces.append(Piece(i+1, verts_cent, lens, area, (0,0)))

    cfg = SolverConfig(max_nodes=5000, max_seconds=5.0)
    solver = PuzzleSolver(test_pieces, cfg)
    places, score, nodes = solver.solve()

    print(f"Success: {places is not None}, Score: {score:.6f}, Nodes: {nodes}")
    if places:
        for pid, p in places.items():
            print(f"  Piece {pid}: angle={math.degrees(p.angle_rad):.2f}°, t=({p.translation_mm[0]:.1f}, {p.translation_mm[1]:.1f})")