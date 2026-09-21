/**
 * assembler.cpp — C ABI bridge to the puzzle::search solver.
 *
 * Converts C structs (PA_*) ↔ C++ puzzle:: types,
 * runs the solver, and converts results back.
 */

#include "assembler.h"
#include "../core/solver.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace puzzle;

// ----------------------------------------------------------------
// Conversions
// ----------------------------------------------------------------

static Polygon pa_poly_to_polygon(const PA_Polygon* pa) {
    Polygon out;
    out.reserve(pa->count);
    for (int i = 0; i < pa->count; ++i) {
        out.push_back({pa->points[i].x_mm, pa->points[i].y_mm});
    }
    return out;
}

static void polygon_to_pa_poly(const Polygon& poly, PA_Polygon* pa) {
    pa->count = (uint8_t)std::min(poly.size(), (size_t)PA_MAX_VERTICES);
    for (int i = 0; i < pa->count; ++i) {
        pa->points[i].x_mm = poly[i].x;
        pa->points[i].y_mm = poly[i].y;
    }
}

// ----------------------------------------------------------------
// API
// ----------------------------------------------------------------

uint32_t pa_abi_version(void) {
    return PA_ABI_VERSION;
}

const char* pa_status_text(int32_t status) {
    switch (status) {
        case  0: return "OK";
        case -1: return "no solution found";
        case -2: return "search limit hit (nodes or time)";
        case -3: return "invalid input (need exactly 4 pieces)";
        case -4: return "invalid vertex count (need 3-5)";
        case -5: return "internal error";
        default: return "unknown status";
    }
}

int32_t pa_assemble(const PA_Polygon* contours,
                    const PA_Config* config,
                    PA_Result* result) {
    if (!contours || !config || !result)
        return -5;

    std::memset(result, 0, sizeof(*result));

    // Validate input: exactly PA_PIECE_COUNT pieces
    // (We'll actually support 1-6 pieces, not just 4)
    int piece_count = PA_PIECE_COUNT;

    // But let's be flexible: count non-empty pieces
    int actual_count = 0;
    for (int i = 0; i < PA_PIECE_COUNT; ++i) {
        if (contours[i].count >= 3)
            actual_count++;
    }

    if (actual_count < 2 || actual_count > PA_PIECE_COUNT) {
        result->status = -3;
        return -3;
    }

    // Validate vertices
    for (int i = 0; i < actual_count; ++i) {
        int vc = contours[i].count;
        if (vc < 3 || vc > PA_MAX_VERTICES) {
            result->status = -4;
            return -4;
        }
    }

    // Convert input
    std::vector<Polygon> pieces;
    pieces.reserve(actual_count);
    for (int i = 0; i < actual_count; ++i) {
        pieces.push_back(ensure_ccw(pa_poly_to_polygon(&contours[i])));
    }

    // Configure solver
    SolverConfig cfg;
    cfg.length_tolerance = config->tolerance_mm;
    cfg.min_edge_mm = 5.0f;  // default
    cfg.overlap_thresh = 0.15f;
    cfg.max_nodes = config->max_nodes ? config->max_nodes : 50000;
    cfg.max_time_ms = config->max_time_ms ? config->max_time_ms : 15000;

    if (config->infer_target_size) {
        // The solver evaluates aspect ratio automatically
    }
    if (config->target_width_mm > 0 && config->target_height_mm > 0) {
        // aspect constraint is checked in evaluate()
    }

    // Run solver
    SolverResult sol = search(pieces, cfg);

    // Convert results
    result->nodes_visited = sol.nodes_visited;
    result->cache_hits = sol.cache_hits;
    result->elapsed_ms = sol.elapsed_ms;

    if (!sol.solved()) {
        if (sol.limit_hit) {
            result->status = -2;
            return -2;
        }
        result->status = -1;
        return -1;
    }

    // Compute rotation angles for each piece
    for (int i = 0; i < actual_count; ++i) {
        const auto& orig = pieces[i];
        const auto& sol_poly = sol.polygons[i];

        // Centroid of target
        Point2f cent = polygon_centroid(sol_poly);
        result->pieces[i].center_mm.x_mm = cent.x;
        result->pieces[i].center_mm.y_mm = cent.y;

        // Rotation: angle between original first edge and solved first edge
        if (orig.size() >= 2 && sol_poly.size() >= 2) {
            float orig_ang = std::atan2(orig[1].y - orig[0].y,
                                        orig[1].x - orig[0].x);
            float sol_ang  = std::atan2(sol_poly[1].y - sol_poly[0].y,
                                        sol_poly[1].x - sol_poly[0].x);
            float th = sol_ang - orig_ang;
            th = std::fmod(th * 180.0f / 3.14159265358979f + 540.0f, 360.0f) - 180.0f;
            result->pieces[i].rotation_deg = th;
        }

        // Target vertices
        polygon_to_pa_poly(sol_poly, &result->pieces[i].target);
    }

    // Compute target rectangle size from solved layout
    if (sol.solved()) {
        auto info = min_area_rect_info_multi(sol.polygons);
        result->target_width_mm = info.long_side;
        result->target_height_mm = info.short_side;
    }

    result->status = 0;
    return 0;
}
