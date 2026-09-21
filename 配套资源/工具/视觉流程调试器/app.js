"use strict";

const stageDefinitions = [
  ["original", "00 原图"],
  ["color", "01 颜色空间"],
  ["filtered", "02 滤波"],
  ["binary", "03 二值化"],
  ["eroded", "04 腐蚀"],
  ["dilated", "05 膨胀"],
  ["opened", "06 开运算"],
  ["closed", "07 闭运算"],
  ["contours", "08 轮廓"],
];

let cvReady = false;
let sourceReady = false;
let processingTimer = null;

function byId(id) { return document.getElementById(id); }

function setMessage(message) { byId("message").textContent = message; }

function setStatus(message, state = "") {
  const status = byId("status");
  status.textContent = message;
  status.className = `status ${state}`.trim();
}

function onOpenCvScriptLoaded() {
  if (typeof cv === "undefined") {
    setStatus("OpenCV.js 加载失败", "error");
    return;
  }
  if (cv.Mat) {
    markCvReady();
  } else {
    cv.onRuntimeInitialized = markCvReady;
  }
}

function markCvReady() {
  cvReady = true;
  setStatus("OpenCV.js 已就绪");
  scheduleProcess();
}

function buildStageCards() {
  const grid = byId("stage-grid");
  const template = byId("stage-template");
  for (const [id, title] of stageDefinitions) {
    const fragment = template.content.cloneNode(true);
    const card = fragment.querySelector("article");
    const canvas = fragment.querySelector("canvas");
    card.dataset.stage = id;
    canvas.id = `stage-${id}`;
    fragment.querySelector("h3").textContent = title;
    fragment.querySelector("button").addEventListener("click", () => downloadCanvas(canvas, `${id}.png`));
    grid.appendChild(fragment);
  }
}

function rangeDefaults(space) {
  if (space === "lab") return [0, -128, -128, 100, 127, 127];
  if (space === "gray") return [0, 0, 0, 255, 0, 0];
  return [0, 100, 80, 12, 255, 255];
}

function addRange(values = rangeDefaults(byId("color-space").value)) {
  const row = document.createElement("div");
  row.className = "range-row";
  values.forEach((value, index) => {
    const input = document.createElement("input");
    input.type = "number";
    input.value = value;
    if (index >= 3) input.classList.add("max");
    input.addEventListener("input", scheduleProcess);
    row.appendChild(input);
  });
  const remove = document.createElement("button");
  remove.type = "button";
  remove.textContent = "删除";
  remove.addEventListener("click", () => {
    row.remove();
    scheduleProcess();
  });
  row.appendChild(remove);
  byId("ranges").appendChild(row);
}

function readRanges() {
  return [...document.querySelectorAll(".range-row")].map((row) =>
    [...row.querySelectorAll("input")].map((input) => Number(input.value))
  );
}

function readConfig() {
  return {
    color_space: byId("color-space").value,
    threshold_mode: byId("threshold-mode").value,
    ranges: readRanges(),
    fixed_threshold: Number(byId("fixed-threshold").value),
    adaptive_block_size: Number(byId("adaptive-block").value),
    adaptive_c: Number(byId("adaptive-c").value),
    filter: byId("filter-mode").value,
    filter_kernel: Number(byId("filter-kernel").value),
    morph_kernel: Number(byId("morph-kernel").value),
    erode_iterations: Number(byId("erode-iterations").value),
    dilate_iterations: Number(byId("dilate-iterations").value),
    open_iterations: Number(byId("open-iterations").value),
    close_iterations: Number(byId("close-iterations").value),
    min_contour_area: Number(byId("min-area").value),
  };
}

function setConfig(config) {
  const ids = {
    color_space: "color-space", threshold_mode: "threshold-mode", fixed_threshold: "fixed-threshold",
    adaptive_block_size: "adaptive-block", adaptive_c: "adaptive-c", filter: "filter-mode",
    filter_kernel: "filter-kernel", morph_kernel: "morph-kernel", erode_iterations: "erode-iterations",
    dilate_iterations: "dilate-iterations", open_iterations: "open-iterations",
    close_iterations: "close-iterations", min_contour_area: "min-area",
  };
  for (const [key, id] of Object.entries(ids)) {
    if (config[key] !== undefined) byId(id).value = config[key];
  }
  if (Array.isArray(config.ranges)) {
    byId("ranges").innerHTML = "";
    config.ranges.forEach(addRange);
  }
  refreshControls();
  scheduleProcess();
}

function ensureOdd(value, name, minimum = 1) {
  if (!Number.isInteger(value) || value < minimum || value % 2 === 0) {
    throw new Error(`${name} 必须是不小于 ${minimum} 的奇数`);
  }
  return value;
}

function convertRange(values, space) {
  if (values.length !== 6 || values.some((value) => !Number.isFinite(value))) {
    throw new Error("每组阈值必须包含 6 个有效数字");
  }
  if (space === "lab") {
    return [
      Math.round(values[0] * 255 / 100), values[1] + 128, values[2] + 128,
      Math.round(values[3] * 255 / 100), values[4] + 128, values[5] + 128,
    ];
  }
  return values;
}

function applyFilter(source, output, mode, kernelSize) {
  if (mode === "none") {
    source.copyTo(output);
  } else if (mode === "gaussian") {
    const size = new cv.Size(ensureOdd(kernelSize, "滤波核"), ensureOdd(kernelSize, "滤波核"));
    cv.GaussianBlur(source, output, size, 0, 0, cv.BORDER_DEFAULT);
  } else if (mode === "median") {
    cv.medianBlur(source, output, ensureOdd(kernelSize, "滤波核"));
  } else {
    throw new Error("未知滤波方式");
  }
}

function createMask(filtered, grayFiltered, config) {
  const mask = new cv.Mat();
  if (config.threshold_mode === "ranges") {
    if (!config.ranges.length) throw new Error("至少需要一组阈值范围");
    const combined = cv.Mat.zeros(filtered.rows, filtered.cols, cv.CV_8UC1);
    for (const raw of config.ranges) {
      const values = convertRange(raw, config.color_space);
      const channels = filtered.channels();
      const lowerValues = channels === 1 ? [values[0]] : values.slice(0, 3);
      const upperValues = channels === 1 ? [values[3]] : values.slice(3, 6);
      const lower = new cv.Mat(filtered.rows, filtered.cols, filtered.type(), new cv.Scalar(...lowerValues));
      const upper = new cv.Mat(filtered.rows, filtered.cols, filtered.type(), new cv.Scalar(...upperValues));
      const current = new cv.Mat();
      cv.inRange(filtered, lower, upper, current);
      cv.bitwise_or(combined, current, combined);
      lower.delete(); upper.delete(); current.delete();
    }
    combined.copyTo(mask);
    combined.delete();
  } else if (config.threshold_mode === "fixed") {
    cv.threshold(grayFiltered, mask, config.fixed_threshold, 255, cv.THRESH_BINARY);
  } else if (config.threshold_mode === "otsu") {
    cv.threshold(grayFiltered, mask, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
  } else if (config.threshold_mode === "adaptive") {
    const block = ensureOdd(config.adaptive_block_size, "自适应邻域", 3);
    cv.adaptiveThreshold(grayFiltered, mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block, config.adaptive_c);
  } else {
    throw new Error("未知阈值方式");
  }
  return mask;
}

function repeatedMorph(source, output, operation, kernel, iterations) {
  source.copyTo(output);
  for (let index = 0; index < iterations; index += 1) {
    const next = new cv.Mat();
    if (operation === "erode") cv.erode(output, next, kernel);
    else cv.dilate(output, next, kernel);
    output.delete();
    output = next;
  }
  return output;
}

function openClose(source, kernel, iterations, first, second) {
  let output = source.clone();
  for (let index = 0; index < iterations; index += 1) {
    let temporary = new cv.Mat();
    if (first === "erode") cv.erode(output, temporary, kernel); else cv.dilate(output, temporary, kernel);
    output.delete();
    output = new cv.Mat();
    if (second === "erode") cv.erode(temporary, output, kernel); else cv.dilate(temporary, output, kernel);
    temporary.delete();
  }
  return output;
}

function showColorStage(mat, space) {
  if (space === "gray") return mat.clone();
  const visible = new cv.Mat();
  cv.cvtColor(mat, visible, space === "hsv" ? cv.COLOR_HSV2RGB : cv.COLOR_Lab2RGB);
  return visible;
}

function processImage() {
  if (!cvReady || !sourceReady) return;
  const config = readConfig();
  const mats = [];
  try {
    const source = cv.imread("source-canvas"); mats.push(source);
    const rgb = new cv.Mat(); mats.push(rgb);
    cv.cvtColor(source, rgb, cv.COLOR_RGBA2RGB);
    const color = new cv.Mat(); mats.push(color);
    if (config.color_space === "gray") cv.cvtColor(source, color, cv.COLOR_RGBA2GRAY);
    else if (config.color_space === "hsv") cv.cvtColor(rgb, color, cv.COLOR_RGB2HSV);
    else cv.cvtColor(rgb, color, cv.COLOR_RGB2Lab);

    const filtered = new cv.Mat(); mats.push(filtered);
    applyFilter(color, filtered, config.filter, config.filter_kernel);
    const gray = new cv.Mat(); mats.push(gray);
    cv.cvtColor(source, gray, cv.COLOR_RGBA2GRAY);
    const grayFiltered = new cv.Mat(); mats.push(grayFiltered);
    applyFilter(gray, grayFiltered, config.filter, config.filter_kernel);
    const binary = createMask(filtered, grayFiltered, config); mats.push(binary);

    const size = ensureOdd(config.morph_kernel, "形态学核");
    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(size, size)); mats.push(kernel);
    let eroded = repeatedMorph(binary, new cv.Mat(), "erode", kernel, config.erode_iterations); mats.push(eroded);
    let dilated = repeatedMorph(eroded, new cv.Mat(), "dilate", kernel, config.dilate_iterations); mats.push(dilated);
    let opened = openClose(dilated, kernel, config.open_iterations, "erode", "dilate"); mats.push(opened);
    let closed = openClose(opened, kernel, config.close_iterations, "dilate", "erode"); mats.push(closed);

    const contourView = source.clone(); mats.push(contourView);
    const contours = new cv.MatVector(); mats.push(contours);
    const hierarchy = new cv.Mat(); mats.push(hierarchy);
    cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let kept = 0;
    for (let index = 0; index < contours.size(); index += 1) {
      const contour = contours.get(index);
      if (cv.contourArea(contour) >= config.min_contour_area) {
        cv.drawContours(contourView, contours, index, new cv.Scalar(0, 255, 0, 255), 2);
        kept += 1;
      }
      contour.delete();
    }

    const colorVisible = showColorStage(color, config.color_space); mats.push(colorVisible);
    const filteredVisible = showColorStage(filtered, config.color_space); mats.push(filteredVisible);
    const stages = { original: source, color: colorVisible, filtered: filteredVisible, binary, eroded, dilated, opened, closed, contours: contourView };
    for (const [id, mat] of Object.entries(stages)) cv.imshow(`stage-${id}`, mat);
    setMessage(`处理完成：保留 ${kept} 个轮廓。\n当前阈值组：${config.ranges.length}`);
  } catch (error) {
    setMessage(`处理失败：${error.message || error}`);
  } finally {
    for (const mat of [...new Set(mats)]) {
      try { mat.delete(); } catch (_) { /* Mat may already have been replaced. */ }
    }
  }
}

function scheduleProcess() {
  clearTimeout(processingTimer);
  processingTimer = setTimeout(processImage, 120);
}

function loadImage(file) {
  const image = new Image();
  image.onload = () => {
    const canvas = byId("source-canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    canvas.getContext("2d").drawImage(image, 0, 0);
    URL.revokeObjectURL(image.src);
    sourceReady = true;
    scheduleProcess();
  };
  image.src = URL.createObjectURL(file);
}

function drawSample() {
  const canvas = byId("source-canvas");
  canvas.width = 560; canvas.height = 360;
  const context = canvas.getContext("2d");
  context.fillStyle = "#ece9df"; context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#dd2828"; context.fillRect(40, 50, 180, 140);
  context.fillStyle = "#2aae55"; context.beginPath(); context.arc(390, 125, 72, 0, Math.PI * 2); context.fill();
  context.fillStyle = "#275bd6"; context.beginPath(); context.moveTo(140, 270); context.lineTo(250, 220); context.lineTo(315, 325); context.closePath(); context.fill();
  context.fillStyle = "#292d2b"; context.font = "34px Segoe UI"; context.fillText("TILearn", 350, 290);
  sourceReady = true;
  scheduleProcess();
}

function refreshControls() {
  const mode = byId("threshold-mode").value;
  byId("ranges-panel").classList.toggle("hidden", mode !== "ranges");
  byId("fixed-panel").classList.toggle("hidden", mode !== "fixed");
  byId("adaptive-panel").classList.toggle("hidden", mode !== "adaptive");
  const space = byId("color-space").value;
  byId("range-hint").textContent = space === "lab" ? "L / A / B" : space === "gray" ? "Gray（仅第一列）" : "H / S / V";
}

function downloadCanvas(canvas, filename) {
  if (!canvas.width) return;
  const link = document.createElement("a");
  link.download = filename;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

function downloadJson() {
  const blob = new Blob([JSON.stringify(readConfig(), null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.download = "vision-pipeline.json";
  link.href = URL.createObjectURL(blob);
  link.click();
  URL.revokeObjectURL(link.href);
}

async function copyThresholds() {
  if (byId("threshold-mode").value !== "ranges") {
    setMessage("只有多范围模式可以复制为 MaixPy 阈值列表。");
    return;
  }
  const reordered = readRanges().map((range) => [range[0], range[3], range[1], range[4], range[2], range[5]]);
  const text = JSON.stringify(reordered);
  try {
    await navigator.clipboard.writeText(text);
    setMessage(`已复制：${text}`);
  } catch (_) {
    setMessage(`浏览器不允许自动复制，请手动复制：\n${text}`);
  }
}

function bindEvents() {
  byId("file-input").addEventListener("change", (event) => {
    if (event.target.files[0]) loadImage(event.target.files[0]);
  });
  byId("sample-button").addEventListener("click", drawSample);
  byId("add-range").addEventListener("click", () => { addRange(); scheduleProcess(); });
  byId("export-config").addEventListener("click", downloadJson);
  byId("copy-thresholds").addEventListener("click", copyThresholds);
  byId("import-config").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;
    file.text().then((text) => setConfig(JSON.parse(text))).catch((error) => setMessage(`导入失败：${error.message}`));
  });
  document.querySelectorAll("select, input[type='number']").forEach((control) => control.addEventListener("input", () => {
    refreshControls(); scheduleProcess();
  }));
  byId("color-space").addEventListener("change", () => {
    byId("ranges").innerHTML = ""; addRange(); refreshControls(); scheduleProcess();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  buildStageCards();
  addRange();
  bindEvents();
  refreshControls();
  drawSample();
});
