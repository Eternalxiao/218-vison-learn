#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
puzzle_solver.py - 2026 拼图求解器 (基于项目六 blackbox_assembler 算法)

算法: 回溯搜索 + 边长匹配 + 碰撞检测
  1. 选一个碎片做锚点 (顶点最多, 约束最强)
  2. DFS 逐步贴合: 把未放置碎片的边贴到已拼好碎片的边上
  3. 边长匹配: 等长边或 T 形接缝 (短边贴长边)
  4. 重叠检测: 包围盒快速排除 + 精确多边形相交面积
  5. 全部放完后: 凸包矩形度 + 覆盖率验收
  6. 时间片轮询: 每个锚点先快速试一轮

单位: 输入像素, 内部转 cm (25px = 1cm), 输出像素
依赖: 标准库 + numpy + cv2
"""

import math
import time as _time
# numpy 和 cv2 在需要时导入 (extract_features, draw_results)
# (cv2 lazy import)


# ================================================================
#  Part 1: 配置常量
# ================================================================

PX_PER_CM = 33       # 33 pixels = 1 cm (50px = 2cm)
TARGET_W_CM = 10.0      # 目标矩形宽度
TARGET_H_CM = 6.0       # 目标矩形高度
TOLERANCE_MM = 5.0      # 几何容差 (mm), 图像识别建议 3~5
MAX_NODES = 500000      # 最大搜索节点
MAX_SECONDS = 15.0      # 最大搜索秒数

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 448

# 特征提取参数 (与 特征提取.py 一致)
SMOOTH_KERNEL = 3
GAUSSIAN_KERNEL = 5
MORPH_KERNEL_SIZE = 3
MORPH_CLOSE_ITER = 2
MORPH_DILATE_ITER = 2
MORPH_ERODE_ITER = 2
BINARY_MODE = "otsu"
INVERT = False
FIXED_THRESHOLD = 127
MIN_AREA = 1000
EPSILON = 0.02
MAX_VERTICES = 5


# ================================================================
#  Part 2: 几何工具函数 (源自项目六 puzzle_geometry.py)
# ================================================================

def _dist(a, b):
    """两点距离"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cross2d(a, b, c):
    """叉积 (b-a) x (c-a)"""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _signed_area(verts):
    """有向面积 (正=逆时针)"""
    n = len(verts)
    s = 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _area(verts):
    """多边形面积"""
    return abs(_signed_area(verts))


def _centroid(verts):
    """质心"""
    sa = _signed_area(verts)
    if abs(sa) < 1e-10:
        n = len(verts)
        return (sum(v[0] for v in verts) / n, sum(v[1] for v in verts) / n)
    cx = cy = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        f = x1 * y2 - x2 * y1
        cx += (x1 + x2) * f
        cy += (y1 + y2) * f
    return (cx / (6.0 * sa), cy / (6.0 * sa))


def _ensure_ccw(verts):
    """确保逆时针 (有向面积为正)"""
    v = list(verts)
    if _signed_area(v) < 0:
        v.reverse()
    return tuple(v)


def _edge_lengths(verts):
    """各边长度"""
    n = len(verts)
    return tuple(_dist(verts[i], verts[(i + 1) % n]) for i in range(n))


def _rotate_point(pt, angle):
    """旋转一个点"""
    c, s = math.cos(angle), math.sin(angle)
    return (pt[0] * c - pt[1] * s, pt[0] * s + pt[1] * c)


def _transform_poly(polygon, angle, tx, ty):
    """刚体变换: 旋转 + 平移"""
    return tuple(
        (p[0] * math.cos(angle) - p[1] * math.sin(angle) + tx,
         p[0] * math.sin(angle) + p[1] * math.cos(angle) + ty)
        for p in polygon
    )


def _point_on_seg(pt, start, end, tol=1e-6):
    """点是否在线段上"""
    if abs(_cross2d(start, end, pt)) > tol:
        return False
    return (min(start[0], end[0]) - tol <= pt[0] <= max(start[0], end[0]) + tol
            and min(start[1], end[1]) - tol <= pt[1] <= max(start[1], end[1]) + tol)


def _point_in_poly_strict(pt, poly):
    """射线法: 点是否在多边形内部 (不含边界)"""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_poly(pt, poly):
    """点是否在多边形内或边界上"""
    n = len(poly)
    for i in range(n):
        if _point_on_seg(pt, poly[i], poly[(i + 1) % n]):
            return True
    return _point_in_poly_strict(pt, poly)


def _segs_intersect_proper(a, b, c, d):
    """两线段是否有严格交点 (不含端点)"""
    c1 = _cross2d(a, b, c)
    c2 = _cross2d(a, b, d)
    c3 = _cross2d(c, d, a)
    c4 = _cross2d(c, d, b)
    return c1 * c2 < -1e-8 and c3 * c4 < -1e-8


def _polys_overlap(a, b, tol=0.5):
    """两个多边形是否重叠 (允许边界接触)"""
    na, nb = len(a), len(b)
    
    # 检查边的严格交点
    for i in range(na):
        for j in range(nb):
            if _segs_intersect_proper(a[i], a[(i+1) % na], b[j], b[(j+1) % nb]):
                return True
    
    # 检查质心是否在另一个多边形内部
    ca = _centroid(a)
    cb = _centroid(b)
    if _point_in_poly_strict(ca, b):
        return True
    if _point_in_poly_strict(cb, a):
        return True
    
    # 检查顶点是否严格在另一个多边形内部 (距离边界 > tol)
    def deep_interior(pt, poly):
        if not _point_in_poly_strict(pt, poly):
            return False
        n = len(poly)
        for k in range(n):
            p1 = poly[k]
            p2 = poly[(k+1) % n]
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            lsq = dx*dx + dy*dy
            if lsq < 1e-10:
                dist = math.hypot(pt[0]-p1[0], pt[1]-p1[1])
            else:
                t = max(0, min(1, ((pt[0]-p1[0])*dx + (pt[1]-p1[1])*dy) / lsq))
                dist = math.hypot(pt[0]-(p1[0]+t*dx), pt[1]-(p1[1]+t*dy))
            if dist < tol:
                return False
        return True
    
    if any(deep_interior(p, b) for p in a):
        return True
    if any(deep_interior(p, a) for p in b):
        return True
    
    return False

def _convex_hull(points):
    """Andrew 单调链凸包"""
    unique = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(unique) <= 1:
        return tuple(unique)
    lower = []
    for pt in unique:
        while len(lower) >= 2 and _cross2d(lower[-2], lower[-1], pt) <= 1e-4:
            lower.pop()
        lower.append(pt)
    upper = []
    for pt in reversed(unique):
        while len(upper) >= 2 and _cross2d(upper[-2], upper[-1], pt) <= 1e-4:
            upper.pop()
        upper.append(pt)
    return tuple(lower[:-1] + upper[:-1])


def _bbox(pts):
    """包围盒 (min_x, min_y, max_x, max_y)"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _collinear_overlap(a1, a2, b1, b2):
    """两条共线线段的重叠长度"""
    length = _dist(a1, a2)
    if length < 1e-7:
        return 0.0
    ux, uy = (a2[0] - a1[0]) / length, (a2[1] - a1[1]) / length

    def proj(pt):
        return (pt[0] - a1[0]) * ux + (pt[1] - a1[1]) * uy

    s, e = proj(b1), proj(b2)
    lo = max(0.0, min(s, e))
    hi = min(length, max(s, e))
    return max(0.0, hi - lo)


def _rigid_angle(src, tgt):
    """从 src 到 tgt 的刚体旋转角 (弧度)"""
    sv = (src[1][0] - src[0][0], src[1][1] - src[0][1])
    tv = (tgt[1][0] - tgt[0][0], tgt[1][1] - tgt[0][1])
    return math.atan2(tv[1], tv[0]) - math.atan2(sv[1], sv[0])


# ================================================================
#  Part 3: 轮廓预处理 (源自项目六 blackbox_assembler.py)
# ================================================================

def preprocess_contours(contours, tol_cm):
    """清理轮廓: 去重复顶点, 去共线点"""
    result = []
    for contour in contours:
        pts = []
        for p in contour:
            pt = (float(p[0]), float(p[1]))
            if not pts or _dist(pts[-1], pt) > 1e-7:
                pts.append(pt)
        if len(pts) > 1 and _dist(pts[0], pts[-1]) < 1e-7:
            pts.pop()

        # 去除共线的中间顶点
        changed = True
        while changed and len(pts) >= 3:
            changed = False
            new_pts = []
            for i, cur in enumerate(pts):
                prev = pts[i - 1]
                nxt = pts[(i + 1) % len(pts)]
                cr = abs(_cross2d(prev, cur, nxt))
                dot = ((cur[0]-prev[0])*(cur[0]-nxt[0]) + (cur[1]-prev[1])*(cur[1]-nxt[1]))
                if cr <= 1e-8 and dot <= 1e-8:
                    changed = True
                else:
                    new_pts.append(cur)
            pts = new_pts

        if len(pts) >= 3:
            result.append(tuple(pts))
    return result


# ================================================================
#  Part 4: 黑盒拼接求解器 (核心, 源自项目六)
# ================================================================

class BlackBoxAssembler:
    """
    纯几何拼图求解器.
    输入: 4 个多边形轮廓 (cm 坐标)
    输出: 每个轮廓在目标矩形中的位置 (cm 坐标)
    """

    def __init__(self, contours, tolerance_mm=5.0,
                 target_w_cm=10.0, target_h_cm=6.0,
                 infer_target_size=False, preprocess=False):
        self.tolerance_cm = tolerance_mm / 10.0
        self.target_w = float(target_w_cm)
        self.target_h = float(target_h_cm)
        self.infer_target = infer_target_size
        self.max_nodes = MAX_NODES
        self.max_seconds = MAX_SECONDS

        if preprocess:
            contours = preprocess_contours(contours, self.tolerance_cm)

        self.contours = [_ensure_ccw(c) for c in contours]
        self._edge_lens = [_edge_lengths(c) for c in self.contours]
        self._input_area = sum(_area(c) for c in self.contours)

        self.nodes = 0
        self.cache_hits = 0
        self.limit_hit = False
        self._failed = set()
        self._deadline = 0

    def _state_key(self, placed):
        """已放置状态的哈希键 (用于去重)"""
        keys = []
        for idx in sorted(placed.keys()):
            poly = placed[idx]
            a = round(_area(poly), 4)
            e = tuple(sorted(round(l, 3) for l in _edge_lengths(poly)))
            keys.append((a, e))
        keys.sort()
        return tuple(keys)

    def solve(self):
        """求解拼图, 返回 4 个目标多边形或 None"""
        n = len(self.contours)
        if n != 4:
            print(f"  [solver] 需要 4 个碎片, 当前 {n} 个")
            return None

        self.nodes = 0
        self.cache_hits = 0
        self.limit_hit = False
        self._failed = set()
        self._deadline = _time.time() + self.max_seconds

        # 锚点排序: 顶点多 + 面积大 的优先 (约束更强)
        anchor_order = sorted(
            range(n),
            key=lambda i: (len(self.contours[i]), _area(self.contours[i])),
            reverse=True
        )

        # 三轮时间片轮询 (避免单个锚点占满时间)
        slices = [0.75, 2.0, self.max_seconds]
        cumulative = 0.0
        t0 = _time.time()

        for round_idx, slice_sec in enumerate(slices):
            cumulative = slice_sec
            rd = t0 + min(cumulative, self.max_seconds)
            for anchor in anchor_order:
                if _time.time() > rd:
                    break
                placed = {anchor: self.contours[anchor]}
                remaining = set(range(n)) - {anchor}
                result = self._search(placed, remaining, rd)
                if result is not None:
                    return self._normalize(result)

        return None

    def _search(self, placed, remaining, deadline):
        """DFS 回溯搜索"""
        self.nodes += 1
        if self.nodes > self.max_nodes:
            self.limit_hit = True
            return None
        if _time.time() > deadline:
            return None

        # 状态去重
        sk = self._state_key(placed)
        if sk in self._failed:
            self.cache_hits += 1
            return None

        # 全部放完 -> 验收
        if not remaining:
            polys = list(placed.values())
            if self._is_rect(polys):
                return dict(placed)
            self._failed.add(sk)
            return None

        placed_list = list(placed.items())
        tol = self.tolerance_cm
        partial_tol = tol * 3

        for partial_mode in (False, True):
            cur_tol = partial_tol if partial_mode else tol

            # 剩余碎片按边数排序
            for moving_idx in sorted(remaining, key=lambda i: -len(self.contours[i])):
                m_edges = self._edge_lens[moving_idx]
                m_poly = self.contours[moving_idx]
                nm = len(m_poly)

                # 枚举边对
                for fi, (fixed_idx, fixed_poly) in enumerate(placed_list):
                    f_edges = self._edge_lens[fixed_idx]
                    nf = len(fixed_poly)

                    for mj in range(nm):
                        ml = m_edges[mj]
                        for fj in range(nf):
                            fl = f_edges[fj]
                            short = min(fl, ml)
                            long = max(fl, ml)

                            # 边长匹配检查
                            if not partial_mode:
                                if short > long * (1.0 + tol):
                                    continue
                            else:
                                if short > long * (1.0 + partial_tol):
                                    continue

                            # 计算旋转角: moving 边反向贴合到 fixed 边
                            angle = (math.atan2(
                                fixed_poly[(fj+1)%nf][1] - fixed_poly[fj][1],
                                fixed_poly[(fj+1)%nf][0] - fixed_poly[fj][0]
                            ) - math.atan2(
                                m_poly[(mj+1)%nm][1] - m_poly[mj][1],
                                m_poly[(mj+1)%nm][0] - m_poly[mj][0]
                            ) + math.pi)

                            # 确定锚点对齐方式
                            if not partial_mode:
                                if abs(fl - ml) < tol:
                                    # 等长: 中点对齐
                                    anchors = [(0.5, 0.5)]
                                else:
                                    # 不等长: 短边两端对齐到长边
                                    if fl >= ml:
                                        anchors = [(0.0, 0.0), (1.0 - ml/fl, 0.0)]
                                    else:
                                        anchors = [(0.0, 0.0), (0.0, 1.0 - fl/ml)]

                                for f_t, m_t in anchors:
                                    fa = fixed_poly[fj]
                                    fb = fixed_poly[(fj+1)%nf]
                                    fx = fa[0] + f_t*(fb[0]-fa[0])
                                    fy = fa[1] + f_t*(fb[1]-fa[1])

                                    ma = m_poly[mj]
                                    mb = m_poly[(mj+1)%nm]
                                    mx = ma[0] + m_t*(mb[0]-ma[0])
                                    my = ma[1] + m_t*(mb[1]-ma[1])

                                    rm = _rotate_point((mx, my), angle)
                                    tx = fx - rm[0]
                                    ty = fy - rm[1]

                                    candidate = _transform_poly(m_poly, angle, tx, ty)

                                    if self._try_place(candidate, placed, remaining - {moving_idx}, deadline):
                                        result = self._search(
                                            {**placed, moving_idx: candidate},
                                            remaining - {moving_idx},
                                            deadline
                                        )
                                        if result is not None:
                                            return result

                            else:
                                # T 形: 尝试端点对齐
                                for m_t in (0.0, 1.0):
                                    for f_t in (0.0, 0.5, 1.0):
                                        fa = fixed_poly[fj]
                                        fb = fixed_poly[(fj+1)%nf]
                                        fx = fa[0] + f_t*(fb[0]-fa[0])
                                        fy = fa[1] + f_t*(fb[1]-fa[1])

                                        ma = m_poly[mj]
                                        mb = m_poly[(mj+1)%nm]
                                        mx = ma[0] + m_t*(mb[0]-ma[0])
                                        my = ma[1] + m_t*(mb[1]-ma[1])

                                        rm = _rotate_point((mx, my), angle)
                                        tx = fx - rm[0]
                                        ty = fy - rm[1]

                                        candidate = _transform_poly(m_poly, angle, tx, ty)

                                        if self._try_place(candidate, placed, remaining - {moving_idx}, deadline):
                                            result = self._search(
                                                {**placed, moving_idx: candidate},
                                                remaining - {moving_idx},
                                                deadline
                                            )
                                            if result is not None:
                                                return result

        self._failed.add(sk)
        return None

    def _try_place(self, candidate, placed, new_remaining, deadline):
        """检查候选放置是否可行 (快速剪枝)"""
        cb = _bbox(candidate)

        # 与已放置碎片的重叠检测
        for existing in placed.values():
            eb = _bbox(existing)
            # 包围盒快速排除
            if (cb[2] < eb[0] or cb[0] > eb[2] or cb[3] < eb[1] or cb[1] > eb[3]):
                continue
            # 精确检测
            if _polys_overlap(candidate, existing):
                return False

        # 凸包面积剪枝: 当前凸包面积不应远超目标面积
        all_pts = []
        for p in placed.values():
            all_pts.extend(p)
        all_pts.extend(candidate)
        hull = _convex_hull(all_pts)
        hull_a = _area(hull)
        if hull_a > self._input_area * 1.5:
            return False

        return True

    def _target_orientation(self, polys):
        """从凸包推断矩形的朝向角"""
        hull = _convex_hull(p for poly in polys for p in poly)
        if len(hull) != 4:
            return None
        edges = _edge_lengths(hull)
        # 找到两条长边 (或两条短边)
        sorted_idx = sorted(range(4), key=lambda i: edges[i])
        # 取最长边方向
        long_idx = sorted_idx[-1]
        a = hull[long_idx]
        b = hull[(long_idx + 1) % 4]
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _is_rect(self, polys):
        """验收: 拼成的是否为目标矩形"""
        all_pts = [p for poly in polys for p in poly]
        hull = _convex_hull(all_pts)
        if len(hull) != 4:
            return False

        edges = sorted(_edge_lengths(hull))
        tol = self.tolerance_cm * 2

        if self.infer_target:
            # 自动推断: 检查是否是任意矩形
            w1, w2 = edges[0], edges[1]
            h1, h2 = edges[2], edges[3]
            if abs(w1 - w2) > tol or abs(h1 - h2) > tol:
                return False
            span_x = max(p[0] for p in all_pts) - min(p[0] for p in all_pts)
            span_y = max(p[1] for p in all_pts) - min(p[1] for p in all_pts)
            self.target_w = max(span_x, span_y)
            self.target_h = min(span_x, span_y)
        else:
            expected = sorted([self.target_w, self.target_w, self.target_h, self.target_h])
            if any(abs(a - e) > tol for a, e in zip(edges, expected)):
                return False

        hull_a = _area(hull)
        total_a = sum(_area(p) for p in polys)
        bbox = _bbox(all_pts)
        bbox_a = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        if bbox_a < 1e-8:
            return False
        rect_score = hull_a / bbox_a
        coverage = total_a / hull_a if hull_a > 1e-8 else 0

        min_rect = max(0.94, 0.98 - self.tolerance_cm * 0.04)
        min_cov = max(0.94, 0.98 - self.tolerance_cm * 0.05)

        return rect_score >= min_rect and min_cov <= coverage <= 1.01

    def _normalize(self, solution):
        """归一化: 旋转到轴对齐, 平移到 (0,0)"""
        polys = list(solution.values())
        orient = self._target_orientation(polys)

        if orient is None:
            hull = _convex_hull(p for poly in polys for p in poly)
            orient = math.atan2(hull[1][1]-hull[0][1], hull[1][0]-hull[0][0])

        angle = -orient
        rotated = [tuple(_rotate_point(p, angle) for p in poly) for poly in polys]

        min_x = min(p[0] for poly in rotated for p in poly)
        min_y = min(p[1] for poly in rotated for p in poly)

        result = []
        for poly in rotated:
            snapped = []
            for x, y in poly:
                nx = x - min_x
                ny = y - min_y
                if abs(nx) < 1e-4: nx = 0.0
                elif abs(nx - self.target_w) < 1e-4: nx = self.target_w
                if abs(ny) < 1e-4: ny = 0.0
                elif abs(ny - self.target_h) < 1e-4: ny = self.target_h
                snapped.append((nx, ny))
            result.append(tuple(snapped))
        return result

# ================================================================
#  Part 5: 求解接口 (适配你的 vision.py 管线)
# ================================================================

class PuzzleSolver:
    """
    拼图求解器 - 对接你的特征提取管线.

    使用方法:
        solver = PuzzleSolver()
        result = solver.solve(pieces)
        # result 是 list 或 None
        # 每个元素:
        #   'id':          'P0', 'P1'...
        #   'from_px':     (cx, cy) 当前质心 (像素)
        #   'target_px':   (tx, ty) 目标质心 (像素)
        #   'theta':       旋转角度 (度, 正值=图像坐标系下顺时针)
        #   'from_cm':     当前质心 (cm)
        #   'target_cm':   目标质心 (cm)
        #   'translation_cm': 平移量 (cm)
    """

    def __init__(self, px_per_cm=PX_PER_CM,
                 target_w=TARGET_W_CM, target_h=TARGET_H_CM,
                 tolerance_mm=TOLERANCE_MM):
        self.px_per_cm = float(px_per_cm)
        self.target_w = float(target_w)
        self.target_h = float(target_h)
        self.tolerance_mm = float(tolerance_mm)

    def px_to_cm(self, px, py):
        """像素转 cm"""
        return (px / self.px_per_cm, py / self.px_per_cm)

    def cm_to_px(self, cx, cy):
        """cm 转像素"""
        return (cx * self.px_per_cm, cy * self.px_per_cm)

    def solve(self, pieces):
        """
        pieces: vision.py 输出的碎片列表
        返回: list[dict] 或 None
        """
        if len(pieces) != 4:
            print(f"  [solver] 需要 4 个碎片, 当前 {len(pieces)} 个")
            return None

        # 1. 像素顶点 -> cm 顶点
        contours_cm = []
        for p in pieces:
            verts_cm = tuple(self.px_to_cm(v[0], v[1]) for v in p["vertices"])
            contours_cm.append(verts_cm)

        # 打印输入信息
        total_cm2 = sum(_area(c) for c in contours_cm)
        print(f"  [solver] 4 个碎片, 总面积={total_cm2:.1f} cm2 (目标 60 cm2)")
        for i, c in enumerate(contours_cm):
            el = _edge_lengths(c)
            el_str = ", ".join(f"{e:.2f}" for e in el)
            print(f"    P{i}: {len(c)}V 边长=[{el_str}] cm")

        # 2. 创建求解器并求解
        asm = BlackBoxAssembler(
            contours_cm,
            tolerance_mm=self.tolerance_mm,
            target_w_cm=self.target_w,
            target_h_cm=self.target_h,
            infer_target_size=False,
            preprocess=True,
        )

        t0 = _time.time()
        solution = asm.solve()
        elapsed = _time.time() - t0

        print(f"  [solver] 节点={asm.nodes}, 缓存命中={asm.cache_hits}, "
              f"耗时={elapsed:.1f}s, 超时={asm.limit_hit}")

        # 如果固定尺寸失败, 尝试自动推断
        if solution is None and not asm.limit_hit:
            print("  [solver] 固定尺寸无解, 尝试自动推断目标尺寸...")
            asm2 = BlackBoxAssembler(
                contours_cm,
                tolerance_mm=self.tolerance_mm,
                target_w_cm=self.target_w,
                target_h_cm=self.target_h,
                infer_target_size=True,
                preprocess=True,
            )
            solution = asm2.solve()
            if solution:
                print(f"  [solver] 推断目标: {asm2.target_w:.1f} x {asm2.target_h:.1f} cm")

        if solution is None:
            print("  [solver] 拼接失败!")
            return None

        # 3. 从解中提取运动参数
        results = []
        for i, (orig_cm, tgt_cm) in enumerate(zip(contours_cm, solution)):
            # 旋转角
            angle_rad = _rigid_angle(orig_cm, tgt_cm)
            angle_deg = math.degrees(angle_rad)
            # 归一化到 [-180, 180]
            angle_deg = (angle_deg + 180.0) % 360.0 - 180.0

            # 质心
            orig_center = _centroid(orig_cm)
            tgt_center = _centroid(tgt_cm)

            # cm -> 像素
            from_px = self.cm_to_px(orig_center[0], orig_center[1])
            to_px = self.cm_to_px(tgt_center[0], tgt_center[1])

            results.append({
                "id": f"P{i}",
                "from_px": from_px,
                "target_px": to_px,
                "theta": angle_deg,
                "from_cm": orig_center,
                "target_cm": tgt_center,
                "translation_cm": (
                    tgt_center[0] - orig_center[0],
                    tgt_center[1] - orig_center[1]
                ),
            })

        return results


# ================================================================
#  Part 6: 像素 -> 机械 mm 转换 (与 my_utils.py 一致)
# ================================================================

def pixel_to_mm(pixel_xy, cfg):
    """像素坐标 -> 机械 mm (线性映射)"""
    px, py = pixel_xy
    cal = cfg.get("calibration", {})
    pr = cal.get("pixel_range", [0, 640, 0, 448])
    xr = cal.get("x_range", [0.0, 300.0])
    yr = cal.get("y_range", [0.0, 210.0])
    x = xr[0] + (px - pr[0]) / (pr[1] - pr[0]) * (xr[1] - xr[0])
    y = yr[0] + (py - pr[2]) / (pr[3] - pr[2]) * (yr[1] - yr[0])
    return (x, y)


# ================================================================
#  Part 7: 特征提取 (与 特征提取.py 一致)
# ================================================================

def extract_features(img_bgr):
    import numpy as np
    import cv2
# numpy 和 cv2 在需要时导入 (extract_features, draw_results)
# (cv2 lazy import)
    """
    图像 -> 碎片列表
    返回 list[dict], 每个 dict 包含:
      vertices: [[x,y], ...] 像素坐标
      area: 面积 px2
      centroid: (cx, cy) 像素
      edge_lengths: [l0, l1, ...] 像素
      n_vertices: 顶点数
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(gray, SMOOTH_KERNEL)
    gau = cv2.GaussianBlur(med, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    kern = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    closed = cv2.morphologyEx(
        gau, cv2.MORPH_CLOSE, kern, iterations=MORPH_CLOSE_ITER)

    flag = cv2.THRESH_BINARY_INV if INVERT else cv2.THRESH_BINARY
    if BINARY_MODE == "otsu":
        _, binary = cv2.threshold(closed, 0, 255, flag + cv2.THRESH_OTSU)
    elif BINARY_MODE == "fixed":
        _, binary = cv2.threshold(closed, FIXED_THRESHOLD, 255, flag)
    else:
        _, binary = cv2.threshold(closed, 0, 255, flag + cv2.THRESH_OTSU)

    dil = cv2.dilate(binary, kern, iterations=MORPH_DILATE_ITER)
    ero = cv2.erode(dil, kern, iterations=MORPH_ERODE_ITER)

    contours, _ = cv2.findContours(
        ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, EPSILON * peri, True)
        if len(approx) > MAX_VERTICES:
            for eps in [0.03, 0.04, 0.05, 0.06, 0.08]:
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) <= MAX_VERTICES:
                    break

        nv = len(approx)
        if nv < 3:
            continue
        verts = approx.reshape(-1, 2).tolist()

        M = cv2.moments(c)
        cx = M["m10"] / M["m00"] if M["m00"] > 0 else 0.0
        cy = M["m01"] / M["m00"] if M["m00"] > 0 else 0.0

        elen = []
        for i in range(nv):
            p1 = np.array(verts[i], dtype=np.float64)
            p2 = np.array(verts[(i + 1) % nv], dtype=np.float64)
            elen.append(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))

        raw.append({
            "vertices": verts,
            "area": float(area),
            "centroid": (float(cx), float(cy)),
            "edge_lengths": elen,
            "n_vertices": nv,
        })
    return raw


# ================================================================
#  Part 8: 可视化
# ================================================================

def draw_results(img_bgr, pieces, results):
    import numpy as np
    import cv2
# numpy 和 cv2 在需要时导入 (extract_features, draw_results)
# (cv2 lazy import)
    """在图像上绘制求解结果"""
    vis = img_bgr.copy()

    for piece, r in zip(pieces, results):
        # 当前轮廓 (红)
        pts = np.array(piece["vertices"], dtype=np.int32)
        cv2.drawContours(vis, [pts], -1, (0, 0, 255), 2)

        # 当前位置 (红点)
        fx, fy = r["from_px"]
        cv2.circle(vis, (int(fx), int(fy)), 5, (0, 0, 255), -1)

        # 目标位置 (绿点)
        tx, ty = r["target_px"]
        cv2.circle(vis, (int(tx), int(ty)), 5, (0, 255, 0), -1)

        # 移动方向 (黄线)
        cv2.line(vis, (int(fx), int(fy)), (int(tx), int(ty)),
                 (0, 255, 255), 2)

        # 标注
        label = f"{r['id']} th={r['theta']:.0f}"
        cv2.putText(vis, label, (int(fx) + 8, int(fy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return vis


# ================================================================
#  Part 9: 主循环 (MaixCAM)
# ================================================================

def main():
    """MaixCAM 主循环"""
    from maix import app, time as maix_time, camera, display, image

    cam = camera.Camera(IMAGE_WIDTH, IMAGE_HEIGHT)
    cam.skip_frames(30)
    disp = display.Display()
    solver = PuzzleSolver()
    print(f"[main] MaixCAM {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"[main] PX_PER_CM={PX_PER_CM} TARGET={TARGET_W_CM}x{TARGET_H_CM}cm")

    cooldown = 0
    while not app.need_exit():
        maix_time.fps_start()
        img = cam.read()
        img_cv = image.image2cv(img, copy=False)

        raw = extract_features(img_cv)

        # 可视化检测到的碎片
        vis = img_cv.copy()
        for r in raw:
            pts = np.array(r["vertices"], dtype=np.int32)
            cv2.drawContours(vis, [pts], -1, (0, 255, 0), 2)
            for v in r["vertices"]:
                cv2.circle(vis, tuple(v), 3, (0, 0, 255), -1)
            cx, cy = r["centroid"]
            cv2.circle(vis, (int(cx), int(cy)), 3, (255, 0, 0), -1)

        # 求解
        if len(raw) == 4 and cooldown <= 0:
            results = solver.solve(raw)
            if results is not None:
                vis = draw_results(vis, raw, results)
                for r in results:
                    print(f"  {r['id']}: "
                          f"from=({r['from_px'][0]:.0f},{r['from_px'][1]:.0f}) "
                          f"to=({r['target_px'][0]:.0f},{r['target_px'][1]:.0f}) "
                          f"th={r['theta']:.1f}")
                cooldown = 120  # 求解成功后等待 4 秒
            else:
                cooldown = 30
        elif cooldown > 0:
            cooldown -= 1

        fps = maix_time.fps()
        rm = image.cv2image(vis, copy=False)
        rm.draw_string(10, 10, f"FPS:{fps:.0f} N:{len(raw)}",
                        color=image.COLOR_YELLOW)
        if len(raw) != 4:
            rm.draw_string(10, 30, f"Need 4 pieces!",
                            color=image.COLOR_RED)
        disp.show(rm)


# ================================================================
#  Part 10: PC 端测试
# ================================================================

def test():
    """PC 端测试: 用已知的 10x6 矩形碎片验证"""
    print("=" * 50)
    print("  拼图求解器测试 (基于项目六算法)")
    print("=" * 50)

    # 4 个碎片, 在 10x6 矩形中的原始位置 (cm)
    original_pieces = [
        ((0, 0), (5, 0), (5, 3), (0, 3)),       # 左上 5x3
        ((5, 0), (10, 0), (10, 3), (5, 3)),      # 右上 5x3
        ((0, 3), (4, 3), (4, 6), (0, 6)),        # 左下 4x3
        ((4, 3), (10, 3), (10, 6), (4, 6)),      # 右下 6x3
    ]

    # 随机旋转和平移 (模拟摄像头看到的散乱状态)
    import random
    rng = random.Random(42)
    scattered = []
    for i, piece in enumerate(original_pieces):
        angle = rng.uniform(0, 2 * math.pi)
        tx = rng.uniform(2, 20)
        ty = rng.uniform(2, 12)
        centered = tuple((p[0] - 5, p[1] - 3) for p in piece)
        rotated = tuple(_rotate_point(p, angle) for p in centered)
        translated = tuple((p[0] + tx, p[1] + ty) for p in rotated)
        # 转为像素
        px_verts = [(int(x * PX_PER_CM), int(y * PX_PER_CM))
                     for x, y in translated]
        cx = sum(v[0] for v in px_verts) / len(px_verts)
        cy = sum(v[1] for v in px_verts) / len(px_verts)
        edges = []
        for j in range(len(px_verts)):
            p1 = px_verts[j]
            p2 = px_verts[(j+1) % len(px_verts)]
            edges.append(math.hypot(p2[0]-p1[0], p2[1]-p1[1]))
        scattered.append({
            "vertices": px_verts,
            "area": float(_area(translated) * PX_PER_CM * PX_PER_CM),
            "centroid": (cx, cy),
            "edge_lengths": edges,
            "n_vertices": len(px_verts),
        })
        print(f"  P{i}: {len(px_verts)}V "
              f"center=({cx:.0f},{cy:.0f})px "
              f"edges={[f'{e:.0f}' for e in edges]}px")

    print()
    solver = PuzzleSolver()
    t0 = _time.time()
    results = solver.solve(scattered)
    elapsed = _time.time() - t0

    print()
    if results:
        print(f"  *** 求解成功! 耗时 {elapsed:.1f}s ***")
        for r in results:
            print(f"  {r['id']}: "
                  f"target=({r['target_cm'][0]:.2f},{r['target_cm'][1]:.2f})cm "
                  f"rotation={r['theta']:.1f}deg")
    else:
        print(f"  *** 求解失败! 耗时 {elapsed:.1f}s ***")

    return results


# ================================================================
#  入口
# ================================================================

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        test()
    else:
        main()