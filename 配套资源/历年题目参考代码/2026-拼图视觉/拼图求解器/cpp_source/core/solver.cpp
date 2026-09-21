#include "solver.hpp"
#include <unordered_set>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>

namespace puzzle {

// ----------------------------------------------------------------
// Attach polygon
// ----------------------------------------------------------------
Polygon attach_polygon(const Polygon& fixed, int fixed_edge,
                       const Polygon& moving, int moving_edge,
                       int alignment) {
    int nf = (int)fixed.size();
    int nm = (int)moving.size();

    const Point2f& fs = fixed[fixed_edge];
    const Point2f& fe = fixed[(fixed_edge + 1) % nf];
    const Point2f& ms = moving[moving_edge];
    const Point2f& me = moving[(moving_edge + 1) % nm];

    // moving edge direction
    float mv_x = me.x - ms.x;
    float mv_y = me.y - ms.y;
    // fixed edge REVERSE direction (anti-parallel)
    float tv_x = fs.x - fe.x;
    float tv_y = fs.y - fe.y;

    float angle = std::atan2(tv_y, tv_x) - std::atan2(mv_y, mv_x);
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);

    // Slide offset
    float f_len = std::hypot(tv_x, tv_y);
    float m_len = std::hypot(mv_x, mv_y);
    if (f_len < EPS) f_len = 1e-9f;
    float td_x = tv_x / f_len;
    float td_y = tv_y / f_len;
    float slide = (alignment + 4.0f) / 8.0f;  // maps [-4,-2,0,2,4] → [0, 0.25, 0.5, 0.75, 1.0]
    float offset = slide * (f_len - m_len);
    float target_x = fe.x + td_x * offset;
    float target_y = fe.y + td_y * offset;

    // Translation: m_start rotated + translate = target
    float ms_rot_x = cos_a * ms.x - sin_a * ms.y;
    float ms_rot_y = sin_a * ms.x + cos_a * ms.y;
    float tx = target_x - ms_rot_x;
    float ty = target_y - ms_rot_y;

    // Transform all vertices
    Polygon result(nm);
    for (int i = 0; i < nm; ++i) {
        result[i].x = cos_a * moving[i].x - sin_a * moving[i].y + tx;
        result[i].y = sin_a * moving[i].x + cos_a * moving[i].y + ty;
    }
    return result;
}

// ----------------------------------------------------------------
// Evaluate
// ----------------------------------------------------------------
double evaluate(const std::vector<Polygon>& polys,
                double& fill_ratio, double& overlap_ratio) {
    int n = (int)polys.size();
    double total_area = 0.0;
    for (auto& p : polys)
        total_area += polygon_area(p);
    if (total_area < 1e-9) {
        fill_ratio = 0.0; overlap_ratio = 1.0;
        return 999.0;
    }

    double overlap = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            overlap += (double)convex_overlap_area(polys[i], polys[j]);

    auto info = min_area_rect_info_multi(polys);
    if (info.area <= 0.0) {
        fill_ratio = 0.0; overlap_ratio = 1.0;
        return 999.0;
    }

    double denom = (double)info.short_side > 1e-9 ? (double)info.short_side : 1e-9;
    double aspect = (double)info.long_side / denom;
    if (aspect < 1.20 || aspect > 1.85) {
        fill_ratio = 0.0; overlap_ratio = 1.0;
        return 999.0;
    }

    fill_ratio = (total_area - overlap) / (double)info.area;
    overlap_ratio = overlap / total_area;
    double score = std::abs(1.0 - fill_ratio) * 8.0 + overlap_ratio * 30.0;
    return score;
}

// ----------------------------------------------------------------
// State hashing for dedup
// ----------------------------------------------------------------
struct StateKey {
    uint64_t key;

    bool operator==(const StateKey& o) const { return key == o.key; }
};

struct StateKeyHash {
    size_t operator()(const StateKey& k) const {
        return std::hash<uint64_t>{}(k.key);
    }
};

static uint64_t hash_state(const std::vector<Polygon>& placed,
                           const std::vector<bool>& mask, int n) {
    // Simple hash: sample first-vertex of each placed polygon
    uint64_t h = 0;
    for (int i = 0; i < n; ++i) {
        if (!mask[i]) continue;
        // Round to ~1mm grid for dedup tolerance
        uint32_t x = (uint32_t)(placed[i][0].x * 4.0f + 0.5f);
        uint32_t y = (uint32_t)(placed[i][0].y * 4.0f + 0.5f);
        h ^= ((uint64_t)x << 32) | y;
        h ^= (uint64_t)placed[i].size() << 16;
        h = h * 0x9e3779b97f4a7c15ULL;
    }
    return h;
}

// ----------------------------------------------------------------
// DFS search
// ----------------------------------------------------------------
SolverResult search(const std::vector<Polygon>& pieces, const SolverConfig& cfg) {
    SolverResult result;
    int n = (int)pieces.size();

    if (n <= 1) {
        result.polygons = pieces;
        result.score = 0.0;
        return result;
    }

    auto t_start = std::chrono::steady_clock::now();
    auto check_timeout = [&]() -> bool {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start).count();
        return (uint32_t)elapsed >= cfg.max_time_ms;
    };

    // Ensure CCW, compute areas, pick anchor (largest area)
    std::vector<Polygon> ccw_pieces = pieces;
    std::vector<double> areas(n);
    int anchor = 0;
    for (int i = 0; i < n; ++i) {
        ccw_pieces[i] = ensure_ccw(pieces[i]);
        areas[i] = (double)polygon_area(ccw_pieces[i]);
        if (areas[i] > areas[anchor]) anchor = i;
    }

    // Pre-compute edge lengths
    std::vector<std::vector<float>> edge_lens(n);
    for (int i = 0; i < n; ++i)
        edge_lens[i] = edge_lengths(ccw_pieces[i]);

    struct Best {
        double score = INF_F;
        std::vector<Polygon> polys;
        double fill = 0.0;
        double overlap = 0.0;
    } best;

    uint32_t nodes = 0;
    uint32_t cache_hits = 0;
    std::unordered_set<StateKey, StateKeyHash> seen;

    std::vector<Polygon> placed(n);
    std::vector<bool> has_placed(n, false);
    placed[anchor] = ccw_pieces[anchor];
    has_placed[anchor] = true;

    // DFS
    std::function<void(int)> dfs = [&](int count) {
        nodes++;
        if (nodes > cfg.max_nodes || check_timeout()) {
            result.limit_hit = true;
            return;
        }

        // Dedup
        {
            StateKey k{hash_state(placed, has_placed, n)};
            if (seen.count(k)) {
                cache_hits++;
                if (cache_hits % 5000 == 0) {
                    // early exit if too many cache hits in a row
                }
                return;
            }
            seen.insert(k);
        }

        if (count == n) {
            std::vector<Polygon> polys;
            for (int i = 0; i < n; ++i)
                polys.push_back(placed[i]);

            double fill_r, overlap_r;
            double score = evaluate(polys, fill_r, overlap_r);
            if (score < best.score) {
                best.score = score;
                best.polys = polys;
                best.fill = fill_r;
                best.overlap = overlap_r;
            }
            return;
        }

        // Try placing each unplaced piece against each placed piece
        for (int mi = 0; mi < n; ++mi) {
            if (has_placed[mi]) continue;
            const auto& mv_poly = ccw_pieces[mi];

            for (int fi = 0; fi < n; ++fi) {
                if (!has_placed[fi]) continue;
                const auto& fp = placed[fi];

                for (int fe = 0; fe < (int)fp.size(); ++fe) {
                    float fl = edge_length(fp, fe);
                    if (fl < cfg.min_edge_mm) continue;

                    for (int me = 0; me < (int)mv_poly.size(); ++me) {
                        float ml = edge_lens[mi][me];
                        if (ml < cfg.min_edge_mm) continue;

                        // Edge length matching
                        float ratio_err = std::abs(fl - ml) / std::max(fl, ml);
                        if (ratio_err > cfg.length_tolerance) continue;

                        // Try alignments
                        const int alignments[] = {-4, -2, 0, 2, 4};
                        for (int align : alignments) {
                            Polygon cand = attach_polygon(fp, fe, mv_poly, me, align);
                            double ca = (double)polygon_area(cand);

                            // Overlap check against all already-placed
                            bool ok = true;
                            for (int k = 0; k < n; ++k) {
                                if (!has_placed[k]) continue;
                                double ov = (double)convex_overlap_area(cand, placed[k]);
                                double min_area = std::min(ca, (double)polygon_area(placed[k]));
                                if (ov > cfg.overlap_thresh * min_area) {
                                    ok = false;
                                    break;
                                }
                            }
                            if (!ok) continue;

                            placed[mi] = cand;
                            has_placed[mi] = true;
                            dfs(count + 1);
                            if (result.limit_hit) return;
                            has_placed[mi] = false;
                        }
                    }
                }
            }
        }
    };

    dfs(1);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count();

    result.nodes_visited = nodes;
    result.cache_hits = cache_hits;
    result.elapsed_ms = (uint32_t)elapsed;

    if (best.polys.empty()) {
        result.score = INF_F;
        return result;
    }

    result.polygons = normalize_solution(best.polys);
    result.score = best.score;
    result.fill_ratio = best.fill;
    result.overlap_ratio = best.overlap;
    return result;
}

// ----------------------------------------------------------------
// Normalize solution
// ----------------------------------------------------------------
std::vector<Polygon> normalize_solution(const std::vector<Polygon>& polys) {
    // Rotate so long edge is horizontal
    auto info = min_area_rect_info_multi(polys);
    float rot = -info.angle_rad;
    float cos_a = std::cos(rot);
    float sin_a = std::sin(rot);

    std::vector<Polygon> normalized;
    float min_x = INF_F, min_y = INF_F;
    for (auto& poly : polys) {
        Polygon np;
        for (auto& p : poly) {
            float nx = cos_a * p.x - sin_a * p.y;
            float ny = sin_a * p.x + cos_a * p.y;
            np.push_back({nx, ny});
            min_x = std::min(min_x, nx);
            min_y = std::min(min_y, ny);
        }
        normalized.push_back(np);
    }

    // Translate to origin
    for (auto& poly : normalized)
        for (auto& p : poly) {
            p.x -= min_x;
            p.y -= min_y;
        }

    return normalized;
}

} // namespace puzzle
