#ifndef PAPER_ASSEMBLER_H
#define PAPER_ASSEMBLER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* --- ABI constants --- */
#define PA_PIECE_COUNT  4
#define PA_MAX_VERTICES 5
#define PA_ABI_VERSION  1

/* --- DLL export / import --- */
#if defined(_WIN32) || defined(_MSC_VER)
  #ifdef PAPER_ASSEMBLER_EXPORTS
    #define PA_API __declspec(dllexport)
  #else
    #define PA_API __declspec(dllimport)
  #endif
#else
  #define PA_API __attribute__((visibility("default")))
#endif

/* --- Point in mm (C struct) --- */
typedef struct {
    float x_mm;
    float y_mm;
} PA_Point;

/* --- Polygon: vertex count + up to PA_MAX_VERTICES points --- */
typedef struct {
    uint8_t  count;
    uint8_t  reserved[3];
    PA_Point points[PA_MAX_VERTICES];
} PA_Polygon;

/* --- Solver configuration --- */
typedef struct {
    float    tolerance_mm;        /* relative edge length tolerance (0.0-1.0) */
    float    target_width_mm;     /* target rectangle width (mm), 0=infer */
    float    target_height_mm;    /* target rectangle height (mm), 0=infer */
    uint8_t  infer_target_size;   /* 1=auto-infer target size */
    uint8_t  reserved[3];
    uint32_t max_nodes;           /* DFS node budget */
    uint32_t max_time_ms;         /* time budget (ms) */
} PA_Config;

/* --- Single piece result --- */
typedef struct {
    PA_Point  center_mm;          /* centroid of target pose (mm) */
    float     rotation_deg;       /* rotation from input to target (deg, CCW math coords) */
    PA_Polygon target;            /* target vertices (mm, CCW, y-up, normalized) */
} PA_PiecePose;

/* --- Full result --- */
typedef struct {
    int32_t    status;            /* 0=OK, nonzero=see pa_status_text() */
    PA_PiecePose pieces[PA_PIECE_COUNT];
    float      target_width_mm;   /* final target width */
    float      target_height_mm;  /* final target height */
    uint32_t   nodes_visited;
    uint32_t   cache_hits;
    uint32_t   elapsed_ms;
} PA_Result;

/* --- API functions --- */

/** Returns the ABI version. Must match PA_ABI_VERSION. */
PA_API uint32_t pa_abi_version(void);

/** Returns a human-readable description of a status code. */
PA_API const char* pa_status_text(int32_t status);

/**
 * Assembles up to 4 piece contours into a target rectangle.
 *
 * @param contours   [in]  4 polygons, each with 3–5 vertices in mm (any winding, any origin)
 * @param config     [in]  solver configuration
 * @param result     [out] filled on success
 * @return 0 on success, nonzero on error
 */
PA_API int32_t pa_assemble(const PA_Polygon* contours,
                           const PA_Config* config,
                           PA_Result* result);

#ifdef __cplusplus
}
#endif

#endif /* PAPER_ASSEMBLER_H */
