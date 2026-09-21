#ifndef PUZZLE_GEOMETRY_HPP
#define PUZZLE_GEOMETRY_HPP

#include <cmath>
#include <vector>
#include <cfloat>
#include <algorithm>
#include <utility>

namespace puzzle {

struct Point2f {
    float x, y;

    Point2f() : x(0), y(0) {}
    Point2f(float x_, float y_) : x(x_), y(y_) {}

    Point2f operator+(const Point2f& o) const { return {x + o.x, y + o.y}; }
    Point2f operator-(const Point2f& o) const { return {x - o.x, y - o.y}; }
    Point2f operator*(float s) const { return {x * s, y * s}; }
    float  dot(const Point2f& o) const { return x * o.x + y * o.y; }
    float  cross(const Point2f& o) const { return x * o.y - y * o.x; }
    float  length() const { return std::hypot(x, y); }
};

using Polygon = std::vector<Point2f>;

// --- Area & centroid ---
float  polygon_area_signed(const Polygon& poly);
float  polygon_area(const Polygon& poly);
Point2f polygon_centroid(const Polygon& poly);

// --- Winding ---
bool   is_ccw(const Polygon& poly);
Polygon ensure_ccw(const Polygon& poly);

// --- Edge helpers ---
float edge_length(const Polygon& poly, int idx);
std::vector<float> edge_lengths(const Polygon& poly);

// --- Convex hull (Andrew's monotone chain) ---
Polygon convex_hull(const std::vector<Point2f>& pts);

// --- Convex polygon overlap (Sutherland-Hodgman) ---
float convex_overlap_area(const Polygon& a, const Polygon& b);

// --- Min-area bounding rectangle (rotating calipers on convex hull) ---
struct MinRectInfo {
    float long_side;   // width of the min-area rect
    float short_side;  // height
    float angle_rad;   // rotation angle of the long edge from +x
    float area;
};
MinRectInfo min_area_rect_info(const Polygon& poly);
MinRectInfo min_area_rect_info_multi(const std::vector<Polygon>& polys);

// --- Simple triangulation for overlap with non-convex polys ---
// We use ear-clipping to decompose a simple polygon into convex pieces.
std::vector<Polygon> triangulate_polygon(const Polygon& poly);

// --- Utility ---
template<typename T> T sq(T v) { return v * v; }
constexpr float EPS = 1e-7f;
constexpr float INF_F = FLT_MAX;

} // namespace puzzle

#endif // PUZZLE_GEOMETRY_HPP
