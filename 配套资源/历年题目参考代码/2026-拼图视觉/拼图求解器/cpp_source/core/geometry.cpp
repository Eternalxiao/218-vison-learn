#include "geometry.hpp"
#include <numeric>

namespace puzzle {

// ----------------------------------------------------------------
// Area & centroid
// ----------------------------------------------------------------
float polygon_area_signed(const Polygon& poly) {
    int n = (int)poly.size();
    if (n < 3) return 0.0f;
    float a = 0.0f;
    for (int i = 0; i < n; ++i) {
        const auto& p0 = poly[i];
        const auto& p1 = poly[(i + 1) % n];
        a += p0.x * p1.y - p1.x * p0.y;
    }
    return a * 0.5f;
}

float polygon_area(const Polygon& poly) {
    return std::abs(polygon_area_signed(poly));
}

Point2f polygon_centroid(const Polygon& poly) {
    int n = (int)poly.size();
    if (n < 3) {
        float cx = 0, cy = 0;
        for (auto& p : poly) { cx += p.x; cy += p.y; }
        if (n > 0) { cx /= n; cy /= n; }
        return {cx, cy};
    }
    float signed_area = polygon_area_signed(poly);
    if (std::abs(signed_area) < EPS) {
        float cx = 0, cy = 0;
        for (auto& p : poly) { cx += p.x; cy += p.y; }
        return {cx / n, cy / n};
    }
    float cx = 0, cy = 0;
    for (int i = 0; i < n; ++i) {
        const auto& p0 = poly[i];
        const auto& p1 = poly[(i + 1) % n];
        float cross = p0.x * p1.y - p1.x * p0.y;
        cx += (p0.x + p1.x) * cross;
        cy += (p0.y + p1.y) * cross;
    }
    float inv = 1.0f / (6.0f * signed_area);
    return {cx * inv, cy * inv};
}

// ----------------------------------------------------------------
// Winding order
// ----------------------------------------------------------------
bool is_ccw(const Polygon& poly) {
    return polygon_area_signed(poly) > 0.0f;
}

Polygon ensure_ccw(const Polygon& poly) {
    if (is_ccw(poly)) return poly;
    Polygon out = poly;
    std::reverse(out.begin(), out.end());
    return out;
}

// ----------------------------------------------------------------
// Edge helpers
// ----------------------------------------------------------------
float edge_length(const Polygon& poly, int idx) {
    int n = (int)poly.size();
    const auto& p0 = poly[idx];
    const auto& p1 = poly[(idx + 1) % n];
    return std::hypot(p1.x - p0.x, p1.y - p0.y);
}

std::vector<float> edge_lengths(const Polygon& poly) {
    std::vector<float> out(poly.size());
    for (int i = 0; i < (int)poly.size(); ++i)
        out[i] = edge_length(poly, i);
    return out;
}

// ----------------------------------------------------------------
// Convex hull: Andrew's monotone chain
// ----------------------------------------------------------------
Polygon convex_hull(const std::vector<Point2f>& pts_in) {
    if (pts_in.size() <= 2) return pts_in;
    auto pts = pts_in;
    std::sort(pts.begin(), pts.end(),
              [](const Point2f& a, const Point2f& b) {
                  return a.x < b.x || (a.x == b.x && a.y < b.y);
              });
    pts.erase(std::unique(pts.begin(), pts.end(),
              [](const Point2f& a, const Point2f& b) {
                  return std::abs(a.x - b.x) < EPS && std::abs(a.y - b.y) < EPS;
              }), pts.end());

    auto cross_o = [](const Point2f& o, const Point2f& a, const Point2f& b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };

    Polygon lower;
    for (auto& p : pts) {
        while (lower.size() >= 2 && cross_o(lower[lower.size()-2], lower.back(), p) <= EPS)
            lower.pop_back();
        lower.push_back(p);
    }
    Polygon upper;
    for (int i = (int)pts.size() - 1; i >= 0; --i) {
        auto& p = pts[i];
        while (upper.size() >= 2 && cross_o(upper[upper.size()-2], upper.back(), p) <= EPS)
            upper.pop_back();
        upper.push_back(p);
    }
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;
}

// ----------------------------------------------------------------
// Sutherland-Hodgman convex polygon clip (both polys CCW, convex)
// ----------------------------------------------------------------
static bool inside(const Point2f& p, const Point2f& es, const Point2f& ee) {
    return (ee.x - es.x) * (p.y - es.y) - (ee.y - es.y) * (p.x - es.x) >= -EPS;
}

static Point2f line_intersect(const Point2f& p1, const Point2f& p2,
                               const Point2f& p3, const Point2f& p4) {
    float denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
    if (std::abs(denom) < 1e-12f) {
        return {(p1.x + p3.x) * 0.5f, (p1.y + p3.y) * 0.5f};
    }
    float t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom;
    return {p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y)};
}

float convex_overlap_area(const Polygon& poly_a, const Polygon& poly_b) {
    Polygon output = poly_a;
    int nb = (int)poly_b.size();
    for (int i = 0; i < nb && !output.empty(); ++i) {
        const auto& es = poly_b[i];
        const auto& ee = poly_b[(i + 1) % nb];
        Polygon input = std::move(output);
        output.clear();
        int nin = (int)input.size();
        for (int j = 0; j < nin; ++j) {
            const auto& cur = input[j];
            const auto& prev = input[(j - 1 + nin) % nin];
            bool cur_in = inside(cur, es, ee);
            bool prev_in = inside(prev, es, ee);
            if (cur_in) {
                if (!prev_in)
                    output.push_back(line_intersect(prev, cur, es, ee));
                output.push_back(cur);
            } else if (prev_in) {
                output.push_back(line_intersect(prev, cur, es, ee));
            }
        }
    }
    if (output.size() < 3) return 0.0f;
    return polygon_area(output);
}

// ----------------------------------------------------------------
// Min-area bounding rectangle (rotating calipers)
// ----------------------------------------------------------------
MinRectInfo min_area_rect_info(const Polygon& poly) {
    Polygon hull = convex_hull(poly);
    int n = (int)hull.size();
    if (n < 3) return {0, 0, 0, 0};

    MinRectInfo best;
    best.area = INF_F;

    for (int i = 0; i < n; ++i) {
        const auto& p0 = hull[i];
        const auto& p1 = hull[(i + 1) % n];
        float angle = std::atan2(p1.y - p0.y, p1.x - p0.x);
        float cos_a = std::cos(-angle);
        float sin_a = std::sin(-angle);

        float min_x = INF_F, max_x = -INF_F;
        float min_y = INF_F, max_y = -INF_F;
        for (auto& p : hull) {
            float rx = cos_a * p.x - sin_a * p.y;
            float ry = sin_a * p.x + cos_a * p.y;
            min_x = std::min(min_x, rx);
            max_x = std::max(max_x, rx);
            min_y = std::min(min_y, ry);
            max_y = std::max(max_y, ry);
        }
        float w = max_x - min_x;
        float h = max_y - min_y;
        float area = w * h;
        if (area < best.area) {
            best.area = area;
            best.angle_rad = angle;
            best.long_side = std::max(w, h);
            best.short_side = std::min(w, h);
        }
    }
    return best;
}

MinRectInfo min_area_rect_info_multi(const std::vector<Polygon>& polys) {
    std::vector<Point2f> all_pts;
    for (auto& p : polys)
        all_pts.insert(all_pts.end(), p.begin(), p.end());
    return min_area_rect_info(all_pts);
}

// ----------------------------------------------------------------
// Ear-clipping triangulation (for general polygon → convex triangles)
// ----------------------------------------------------------------
std::vector<Polygon> triangulate_polygon(const Polygon& poly) {
    Polygon verts = ensure_ccw(poly);
    if (verts.size() == 3) return {verts};

    std::vector<int> indices(verts.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<Polygon> triangles;

    auto cross_idx = [&](int i0, int i1, int i2) -> float {
        const auto& a = verts[i0];
        const auto& b = verts[i1];
        const auto& c = verts[i2];
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    };

    auto inside_tri = [&](int pi, int i0, int i1, int i2) -> bool {
        const auto& pt = verts[pi];
        return cross_idx(i0, i1, pi) >= -EPS &&
               cross_idx(i1, i2, pi) >= -EPS &&
               cross_idx(i2, i0, pi) >= -EPS;
    };

    while (indices.size() > 3) {
        bool ear_found = false;
        for (int pos = 0; pos < (int)indices.size(); ++pos) {
            int prev = indices[(pos - 1 + (int)indices.size()) % indices.size()];
            int curr = indices[pos];
            int next = indices[(pos + 1) % indices.size()];
            if (cross_idx(prev, curr, next) <= EPS) continue;
            bool any_inside = false;
            for (int idx : indices) {
                if (idx == prev || idx == curr || idx == next) continue;
                if (inside_tri(idx, prev, curr, next)) {
                    any_inside = true;
                    break;
                }
            }
            if (any_inside) continue;
            triangles.push_back({verts[prev], verts[curr], verts[next]});
            indices.erase(indices.begin() + pos);
            ear_found = true;
            break;
        }
        if (!ear_found) {
            // Fallback: fan triangulation
            int anchor = indices[0];
            for (int i = 1; i + 1 < (int)indices.size(); ++i)
                triangles.push_back({verts[anchor], verts[indices[i]], verts[indices[i+1]]});
            return triangles;
        }
    }
    triangles.push_back({verts[indices[0]], verts[indices[1]], verts[indices[2]]});
    return triangles;
}

} // namespace puzzle
