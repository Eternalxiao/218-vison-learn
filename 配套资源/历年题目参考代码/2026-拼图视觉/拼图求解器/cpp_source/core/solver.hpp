#ifndef PUZZLE_SOLVER_HPP
#define PUZZLE_SOLVER_HPP

#include "geometry.hpp"
#include <vector>
#include <cstdint>
#include <functional>
#include <string>

namespace puzzle {

struct SolverConfig {
    float length_tolerance = 0.50f;   // relative edge length tolerance
    float min_edge_mm = 5.0f;         // minimum valid edge length
    float overlap_thresh = 0.15f;     // max overlap ratio before rejecting
    uint32_t max_nodes = 50000;       // DFS node budget
    uint32_t max_time_ms = 15000;     // time budget

    // Target rectangle (used by evaluate)
    float target_aspect_min = 1.20f;
    float target_aspect_max = 1.85f;
};

struct SolverResult {
    std::vector<Polygon> polygons;    // solved polygons in mm (CCW, y-up, normalized)
    double score = INF_F;
    double fill_ratio = 0.0;
    double overlap_ratio = 0.0;
    uint32_t nodes_visited = 0;
    uint32_t cache_hits = 0;
    uint32_t elapsed_ms = 0;
    bool limit_hit = false;

    bool solved() const { return !polygons.empty() && score < 999.0; }
};

// --- Core: attach_polygon ---
// Places `moving` so that its edge `moving_edge` is anti-parallel-aligned
// to `fixed`'s edge `fixed_edge`, with the given alignment offset.
// alignment ∈ {-4, -2, 0, 2, 4}: -4=far-end align, 0=center, +4=near-end align
Polygon attach_polygon(const Polygon& fixed, int fixed_edge,
                       const Polygon& moving, int moving_edge,
                       int alignment);

// --- DFS search ---
// pieces: each polygon is CCW, y-up, in mm
SolverResult search(const std::vector<Polygon>& pieces, const SolverConfig& cfg);

// --- Evaluation (score, fill_ratio, overlap_ratio) ---
double evaluate(const std::vector<Polygon>& polys,
                double& fill_ratio, double& overlap_ratio);

// --- Normalize: rotate so long edge is horizontal, translate to origin ---
std::vector<Polygon> normalize_solution(const std::vector<Polygon>& polys);

} // namespace puzzle

#endif // PUZZLE_SOLVER_HPP
