"""
比赛现场标定自检工具 (极性自适应版)
在debug模式下快速评估当前参数是否健康, 发现潜在光照/标定问题
"""
import cv2
import numpy as np


def check_roi_health(gray, cfg):
    """
    检测A4纸ROI定位是否健康 (支持 dark/light/auto 极性)
    返回: (ok, warnings)
    """
    warnings = []
    r = cfg["roi"]
    h, w = gray.shape[:2]
    polarity = r.get("paper_polarity", "dark")
    mean_gray = float(np.mean(gray))

    if polarity == "dark":
        # 深色纸: 期望纸区域比背景暗
        dark_ratio = np.sum(gray < r["binary_thresh"]) / (h * w)
        if dark_ratio < 0.02:
            warnings.append("暗区仅%.1f%%, 黑纸未检出(thresh=%d). 若纸是白色请切P:light/auto" % (
                dark_ratio * 100, r["binary_thresh"]))
        elif dark_ratio > 0.6:
            warnings.append("暗区%.1f%%过大, 可能整画面被误判为纸" % (dark_ratio * 100))
        if r["binary_thresh"] > mean_gray:
            warnings.append("binary_thresh=%d > 平均灰度%.0f, 黑纸检测将失效" % (
                r["binary_thresh"], mean_gray))
    elif polarity == "light":
        # 浅色纸: 期望纸区域比背景亮
        light_ratio = np.sum(gray > r["binary_thresh"]) / (h * w)
        if light_ratio < 0.02:
            warnings.append("亮区仅%.1f%%, 白纸未检出. 若纸是黑色请切P:dark/auto" % (light_ratio * 100))
        elif light_ratio > 0.7:
            warnings.append("亮区%.1f%%过大, 背景过亮或纸与背景对比不足" % (light_ratio * 100))
    else:  # auto
        # auto 模式: 仅提示整体过曝/过暗
        if mean_gray > 230:
            warnings.append("画面过亮(avg=%.0f), 纸与背景对比可能不足" % mean_gray)
        elif mean_gray < 20:
            warnings.append("画面过暗(avg=%.0f), 环境光不足" % mean_gray)

    ok = len(warnings) == 0
    return ok, warnings


def check_otsu_health(work_gray, cfg):
    """
    检查OTSU二值化是否健康 (是否需要切换为fixed/adaptive模式)
    返回: (ok, warnings, suggested_threshold)
    """
    warnings = []
    b = cfg["binary"]
    if b["mode"] != "otsu":
        return True, [], b["fixed_threshold"]

    otsu_val, _ = cv2.threshold(work_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hist = cv2.calcHist([work_gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist_smooth = cv2.GaussianBlur(hist.reshape(-1, 1), (5, 1), 0).flatten()

    peaks = 0
    for i in range(1, 255):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            if hist_smooth[i] > np.max(hist_smooth) * 0.05:
                peaks += 1

    if peaks < 2:
        warnings.append("直方图非双峰(峰=%d), OTSU不稳. 建议Adapt(光照不均)或Fixed/HSV" % peaks)

    if otsu_val < 30:
        warnings.append("OTSU=%.0f过低, 可能大面积误判前景" % otsu_val)
    elif otsu_val > 220:
        warnings.append("OTSU=%.0f过高, 可能丢失碎片" % otsu_val)

    median_val = float(np.median(work_gray))
    suggested = int(median_val * 0.85) if median_val > 80 else int(median_val * 0.7)

    ok = len(warnings) == 0
    return ok, warnings, suggested


def check_fragment_count(pieces, cfg):
    """检查检测到的碎片数量是否符合预期"""
    warnings = []
    expected = len(cfg["templates"])
    actual = len(pieces)

    if actual == 0:
        warnings.append("未检测到碎片! 检查: 二值化模式/碎片明暗(Inv)/纸极性(P)/光照")
    elif actual < expected:
        warnings.append("检测到%d个碎片, 期望%d个 (漏检: 试Inv或调阈值)" % (actual, expected))
    elif actual > expected:
        warnings.append("检测到%d个碎片, 期望%d个 (误检: 提高min_area或调阈值)" % (actual, expected))

    ok = actual == expected
    return ok, warnings


def check_contour_quality(pieces, cfg):
    """检查碎片轮廓质量"""
    warnings = []
    if len(pieces) == 0:
        return False, ["无碎片可检查"]

    templates = cfg["templates"]
    c = cfg["contour"]

    for i, p in enumerate(pieces):
        area = p["area"]
        nv = p["n_vertices"]

        if area < c["min_area"] * 1.5:
            warnings.append("碎片#%d 面积=%d 接近min_area=%d, 可能不完整" % (i, area, c["min_area"]))

        if nv < 3:
            warnings.append("碎片#%d 顶点数=%d (异常)" % (i, nv))

        for j, el in enumerate(p["edge_lengths"]):
            if el < 5:
                warnings.append("碎片#%d 边%d=%.1fpx (极短, 逼近异常)" % (i, j, el))

    available_nv = set(p["n_vertices"] for p in pieces)
    for pid, t in templates.items():
        if t["n_vertices"] not in available_nv:
            warnings.append("模板%s需%d顶点, 未检测到对应碎片" % (pid, t["n_vertices"]))

    ok = len(warnings) == 0
    return ok, warnings


def run_all_checks(img_cv, gray, work_gray, pieces, cfg):
    """运行所有标定自检, 返回结构化诊断结果"""
    all_ok = True
    results = {
        "roi": {"ok": True, "warnings": []},
        "otsu": {"ok": True, "warnings": [], "suggested_threshold": 127},
        "count": {"ok": True, "warnings": []},
        "quality": {"ok": True, "warnings": []},
    }

    if cfg["roi"]["enabled"]:
        ok, warnings = check_roi_health(gray, cfg)
        results["roi"] = {"ok": ok, "warnings": warnings}
        if not ok:
            all_ok = False

    if work_gray is not None:
        ok, warnings, suggested = check_otsu_health(work_gray, cfg)
        results["otsu"] = {"ok": ok, "warnings": warnings, "suggested_threshold": suggested}
        if not ok:
            all_ok = False

    ok, warnings = check_fragment_count(pieces, cfg)
    results["count"] = {"ok": ok, "warnings": warnings}
    if not ok:
        all_ok = False

    ok, warnings = check_contour_quality(pieces, cfg)
    results["quality"] = {"ok": ok, "warnings": warnings}
    if not ok:
        all_ok = False

    return all_ok, results
