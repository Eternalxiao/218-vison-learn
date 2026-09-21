"""
assembler.py - 2026 拼图拼接算法 (最简版)

唯一职责: 拿到 4 个碎片的像素顶点 -> 算出 from / to / th

用法:
    from assembler import solve

    pieces = [
        [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],  # P0
        [(x1,y1), (x2,y2), (x3,y3)],            # P1
        [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],  # P2
        [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],  # P3
    ]

    results = solve(pieces)
    # [{"id":"P0","from":(cx,cy),"to":(tx,ty),"th":45.0}, ...]

纯像素坐标, 不需要任何标定, 零依赖 (仅标准库).
"""

import math
import time as _time

# ===================== 配置 =====================
EDGE_MATCH_ABS = 8.0
EDGE_MATCH_REL = 0.08
GAP_TOL = 0.08
ASPECT_MIN = 1.0
ASPECT_MAX = 2.5
MAX_NODES = 300000
MAX_SECONDS = 15.0
DEBUG = False


# ===================== 碎片类 =====================
class _Piece:
    def __init__(self, pid, vertices_px):
        self.pid = pid
        n = len(vertices_px)
        v = [(float(x), float(y)) for x, y in vertices_px]
        sa = 0.0
        cx = cy = 0.0
        for i in range(n):
            x0, y0 = v[i]
            x1, y1 = v[(i + 1) % n]
            cross = x0 * y1 - x1 * y0
            sa += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        sa *= 0.5
        if abs(sa) > 1e-6:
            cx /= 6.0 * sa
            cy /= 6.0 * sa
        else:
            cx = sum(p[0] for p in v) / n
            cy = sum(p[1] for p in v) / n
        self.centroid = (cx, cy)
        if sa < 0:
            v = v[::-1]
        self.verts = v
        self.area = abs(sa)
        self.edges = []
        for i in range(n):
            sx, sy = v[i]
            ex, ey = v[(i + 1) % n]
            dx, dy = ex - sx, ey - sy
            self.edges.append(((sx, sy), (ex, ey), math.hypot(dx, dy), math.atan2(dy, dx)))


# ===================== 几何工具 =====================
def _attach(ea_s, ea_e, eb_s, eb_e, piece_b_verts, eb_idx):
    ax, ay = ea_e[0] - ea_s[0], ea_e[1] - ea_s[1]
    bx, by = eb_e[0] - eb_s[0], eb_e[1] - eb_s[1]
    angle = math.atan2(ay, ax) - math.atan2(by, bx) + math.pi
    c, s = math.cos(angle), math.sin(angle)
    rotated = [(vx * c - vy * s, vx * s + vy * c) for vx, vy in piece_b_verts]
    rbx, rby = rotated[eb_idx]
    tx, ty = ea_e[0] - rbx, ea_e[1] - rby
    return [(rx + tx, ry + ty) for rx, ry in rotated], math.degrees(angle)


def _sat_overlap(a, b):
    SAT_TOL = 3.0
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            p1x, p1y = poly[i]
            p2x, p2y = poly[(i + 1) % n]
            ax = -(p2y - p1y)
            ay = p2x - p1x
            norm = math.hypot(ax, ay)
            if norm < 1e-10:
                continue
            min_a = max_a = a[0][0] * ax + a[0][1] * ay
            for v in a[1:]:
                d = v[0] * ax + v[1] * ay
                if d < min_a: min_a = d
                if d > max_a: max_a = d
            min_b = max_b = b[0][0] * ax + b[0][1] * ay
            for v in b[1:]:
                d = v[0] * ax + v[1] * ay
                if d < min_b: min_b = d
                if d > max_b: max_b = d
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < SAT_TOL * norm:
                return False
    return True


def _bbox_overlap(a, b, margin=0):
    return not (max(v[0] for v in a) + margin < min(v[0] for v in b)
                or max(v[0] for v in b) + margin < min(v[0] for v in a)
                or max(v[1] for v in a) + margin < min(v[1] for v in b)
                or max(v[1] for v in b) + margin < min(v[1] for v in a))


def _convex_hull(points):
    pts = sorted(set((round(x, 4), round(y, 4)) for x, y in points))
    if len(pts) <= 2:
        return list(pts)
    lower = []
    for p in pts:
        while len(lower) >= 2:
            a, b = lower[-2], lower[-1]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 1e-4:
                lower.pop()
            else:
                break
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2:
            a, b = upper[-2], upper[-1]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 1e-4:
                upper.pop()
            else:
                break
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _poly_area(verts):
    n = len(verts)
    sa = 0.0
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        sa += x0 * y1 - x1 * y0
    return abs(sa) / 2.0


def _centroid_of(verts):
    n = len(verts)
    return (sum(v[0] for v in verts) / n, sum(v[1] for v in verts) / n)


# ===================== 验收 =====================
def _simplify_hull(hull):
    simplified = list(hull)
    changed = True
    while changed and len(simplified) > 4:
        changed = False
        for k in range(len(simplified)):
            pp = simplified[(k - 1) % len(simplified)]
            pc = simplified[k]
            pn = simplified[(k + 1) % len(simplified)]
            dx2, dy2 = pn[0] - pp[0], pn[1] - pp[1]
            seg = math.hypot(dx2, dy2)
            if seg < 1.0:
                continue
            cross = abs((pc[0] - pp[0]) * dy2 - (pc[1] - pp[1]) * dx2) / seg
            if cross < 5.0:
                simplified.pop(k)
                changed = True
                break
    return simplified


def _validate(placements, total_area):
    n = len(placements)
    if n < 2:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            a = placements[i]["abs_verts"]
            b = placements[j]["abs_verts"]
            if not _bbox_overlap(a, b, margin=-2):
                continue
            if _sat_overlap(a, b):
                return False
    all_pts = []
    for p in placements:
        all_pts.extend(p["abs_verts"])
    hull = _convex_hull(all_pts)
    hull = _simplify_hull(hull)
    if len(hull) != 4:
        return False
    hull_area = _poly_area(hull)
    if hull_area < 1.0:
        return False
    gap = hull_area / total_area - 1.0
    if gap > GAP_TOL:
        return False
    edge_lens = []
    for i in range(4):
        h0 = hull[i]
        h1 = hull[(i + 1) % 4]
        edge_lens.append(math.hypot(h1[0] - h0[0], h1[1] - h0[1]))
    long_side = max(edge_lens)
    short_side = min(edge_lens)
    if short_side < 1.0:
        return False
    aspect = long_side / short_side
    if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
        return False
    if DEBUG:
        print(f"    [val] OK! area={hull_area:.0f} gap={gap:.3f} aspect={aspect:.2f}")
    return True


# ===================== DFS =====================
def _dfs(placed, unplaced, total_area, counter, deadline):
    if not unplaced:
        return placed if _validate(placed, total_area) else None
    counter[0] += 1
    if counter[0] > MAX_NODES:
        return None
    if _time.time() > deadline:
        return None
    for pp in placed:
        pv = pp["abs_verts"]
        nf = len(pv)
        for ei in range(nf):
            ep1, ep2 = pv[ei], pv[(ei + 1) % nf]
            elen_a = math.hypot(ep2[0] - ep1[0], ep2[1] - ep1[1])
            for up in unplaced:
                for uj in range(len(up.edges)):
                    elen_b = up.edges[uj][2]
                    diff = abs(elen_a - elen_b)
                    tol = max(EDGE_MATCH_ABS, max(elen_a, elen_b) * EDGE_MATCH_REL)
                    if diff > tol:
                        continue
                    abs_v, ang = _attach(ep1, ep2, up.edges[uj][0], up.edges[uj][1], up.verts, uj)
                    hit = False
                    for pp2 in placed:
                        if not _bbox_overlap(abs_v, pp2["abs_verts"], -2):
                            continue
                        if _sat_overlap(abs_v, pp2["abs_verts"]):
                            hit = True
                            break
                    if hit:
                        continue
                    new_placed = placed + [{"piece": up, "abs_verts": abs_v, "angle_deg": ang}]
                    new_unplaced = [p for p in unplaced if p.pid != up.pid]
                    result = _dfs(new_placed, new_unplaced, total_area, counter, deadline)
                    if result is not None:
                        return result
    return None


# ===================== 对外接口 =====================
def solve(pieces_vertices):
    """
    输入: 4 个碎片的像素顶点列表
        pieces_vertices = [
            [(x1,y1), (x2,y2), ...],  # P0
            [(x1,y1), (x2,y2), ...],  # P1
            [(x1,y1), (x2,y2), ...],  # P2
            [(x1,y1), (x2,y2), ...],  # P3
        ]

    返回: list[dict] 或 None
        [
            {"id":"P0", "from":(cx,cy), "to":(tx,ty), "th":45.0},
            ...
        ]
        from = 当前质心(px)
        to   = 目标质心(px)
        th   = 旋转角(度, 正=逆时针)
    """
    if len(pieces_vertices) != 4:
        if DEBUG:
            print(f"[assembler] need 4, got {len(pieces_vertices)}")
        return None
    pieces = [_Piece(f"P{i}", verts) for i, verts in enumerate(pieces_vertices)]
    total_area = sum(p.area for p in pieces)
    if DEBUG:
        print(f"[assembler] {len(pieces)} pieces, area={total_area:.0f}px2")
        for p in pieces:
            el = ", ".join(f"{e[2]:.1f}" for e in p.edges)
            print(f"  {p.pid}: {len(p.verts)}V edges=[{el}]px")
    order = sorted(range(4), key=lambda i: (len(pieces[i].edges), pieces[i].area), reverse=True)
    deadline = _time.time() + MAX_SECONDS
    counter = [0]
    for ai in order:
        anchor = pieces[ai]
        placed = [{"piece": anchor, "abs_verts": list(anchor.verts), "angle_deg": 0.0}]
        remaining = [p for p in pieces if p.pid != anchor.pid]
        if DEBUG:
            print(f"[assembler] anchor={anchor.pid}...")
        result = _dfs(placed, remaining, total_area, counter, deadline)
        if result is not None:
            if DEBUG:
                print(f"[assembler] SOLVED! nodes={counter[0]}")
            return _format(result, pieces)
    if DEBUG:
        print(f"[assembler] FAILED. nodes={counter[0]}")
    return None


def _format(placements, originals):
    orig_map = {p.pid: p for p in originals}
    results = []
    for pl in placements:
        pid = pl["piece"].pid
        orig = orig_map[pid]
        from_pt = orig.centroid
        to_pt = _centroid_of(pl["abs_verts"])
        th = pl["angle_deg"]
        th = (th + 180.0) % 360.0 - 180.0
        results.append({
            "id": pid,
            "from": (round(from_pt[0], 1), round(from_pt[1], 1)),
            "to": (round(to_pt[0], 1), round(to_pt[1], 1)),
            "th": round(th, 1),
        })
    return results


# ===================== 自测 =====================
if __name__ == "__main__":
    import random
    DEBUG = True
    print("=" * 50)
    print("  assembler.py self-test")
    print("=" * 50)

    PX = 25
    originals = [
        [(0, 0), (150, 0), (150, 50), (0, 50)],
        [(150, 0), (250, 0), (250, 50), (150, 50)],
        [(0, 50), (100, 50), (100, 150), (0, 150)],
        [(100, 50), (250, 50), (250, 150), (100, 150)],
    ]
    total = sum(_poly_area(p) for p in originals)
    print(f"Total area: {total} (target {250*150})")

    rng = random.Random(42)
    scattered = []
    for piece in originals:
        angle = rng.uniform(0, 2 * math.pi)
        tx = rng.uniform(80, 400)
        ty = rng.uniform(80, 300)
        cx = sum(p[0] for p in piece) / len(piece)
        cy = sum(p[1] for p in piece) / len(piece)
        c, s = math.cos(angle), math.sin(angle)
        new_v = []
        for px, py in piece:
            dx, dy = px - cx, py - cy
            new_v.append((dx * c - dy * s + tx, dx * s + dy * c + ty))
        scattered.append(new_v)

    print("\nInput:")
    for i, verts in enumerate(scattered):
        edges = []
        for j in range(len(verts)):
            p1, p2 = verts[j], verts[(j + 1) % len(verts)]
            edges.append(round(math.hypot(p2[0] - p1[0], p2[1] - p1[1]), 1))
        print(f"  P{i}: {len(verts)}V edges={edges}px")

    t0 = _time.time()
    results = solve(scattered)
    elapsed = _time.time() - t0

    print()
    if results:
        print(f"SOLVED in {elapsed:.2f}s")
        for r in results:
            print(f"  {r['id']}: from={r['from']} to={r['to']} th={r['th']}")
    else:
        print(f"FAILED in {elapsed:.2f}s")