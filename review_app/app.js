"use strict";

const $ = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (value) => Number(value).toLocaleString("zh-CN");
const timeLabel = (value) => new Intl.DateTimeFormat("zh-CN", {timeZone: "Asia/Shanghai", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false}).format(new Date(value));
let catalog, reviews = {}, storageKey, storageAvailable = true, composing = false;
const pairChoices = {};
let state = readURL();
const titles = {night: "昼夜沿用官方标签。", groups: "逐组看场景，确认哪些画面相关。", source: "来源先对齐，审核才有依据。"};
const modeOptions = {
  night: [["sample", "分层覆盖抽样 · 24 张"], ["flagged", "Agent 初审标记 · 15 张"], ["bright", "高亮度候选 · 24 张"], ["all", "全部官方 Night · 1,656 张"], ["day", "官方 Daytime 对照 · 24 张"]],
  groups: [["sample", "本轮待确认 · 50 个新组"], ["history", "上轮已看 · 12 个组（只读）"]]
};
const nightLabels = {match: "符合官方标签", mismatch: "疑似白天", uncertain: "无法判断"};
const groupLabels = {near_duplicate: "几乎相同", same_scene_changed: "同地点，有变化", scene_only: "仅类型相似", mixed: "组内混杂", uncertain: "无法判断"};

function readURL() {
  const p = new URLSearchParams(location.search);
  return {view: ["night", "groups", "source"].includes(p.get("view")) ? p.get("view") : "groups", mode: p.get("mode") || "sample", q: p.get("q") || "", city: p.get("city") || "all", reviewed: p.get("reviewed") || "all", page: Math.max(0, Number.parseInt(p.get("page") || "0", 10) || 0)};
}
function writeURL() {
  const p = new URLSearchParams();
  for (const [key, value] of Object.entries(state)) if (value !== "" && value !== "all" && !(key === "page" && value === 0)) p.set(key, value);
  history.replaceState(null, "", `?${p}`);
}
function announce(text) { $("live-status").textContent = text; }
function actionButton(label, attributes = {}) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  for (const [key, value] of Object.entries(attributes)) button.setAttribute(key, String(value));
  return button.outerHTML;
}
function loadStored() {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey) || "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch { storageAvailable = false; announce("浏览器存储不可用。判断仍保留在本页，请及时导出。"); return {}; }
}
function updateProgress() {
  const count = Object.keys(reviews).length;
  const done = catalog.sample_group_ids.filter((id) => Boolean(currentReview(id))).length;
  $("saved-count").textContent = `本轮已确认 ${done} / ${catalog.sample_group_ids.length} 组 · 浏览器共存 ${fmt(count)} 项记录`;
  $("export").disabled = count === 0;
}
function currentReview(id) {
  const record = reviews[id];
  return record?.kind === "group" && record.campaign_id === catalog.scene_screening.campaign_id && Object.hasOwn(groupLabels, record.verdict) ? record : undefined;
}
function saveReview(id, kind, verdict) {
  if (kind !== "group" || !catalog.sample_group_ids.includes(id) || ![...Object.keys(groupLabels), "undo"].includes(verdict)) return;
  if (storageAvailable) reviews = {...reviews, ...loadStored()};
  const note = document.querySelector(`[data-review-id="${id}"] textarea`)?.value || "";
  if (verdict === "undo") delete reviews[id];
  else reviews[id] = {object_id: id, kind, verdict, note, campaign_id: catalog.scene_screening.campaign_id, reviewed_at: new Date().toISOString()};
  try { localStorage.setItem(storageKey, JSON.stringify(reviews)); }
  catch { storageAvailable = false; }
  updateProgress();
  announce(storageAvailable ? "审核记录已保存在当前浏览器。" : "浏览器保存失败，记录仍在本页。请使用“导出审核记录”。");
  // Update buttons in place to preserve keyboard focus and filter footprint.
  document.querySelectorAll("[data-review-id]").forEach((panel) => {
    if (panel.dataset.reviewId !== id) return;
    const record = reviews[id];
    panel.querySelectorAll("button[data-verdict]").forEach((button) => button.setAttribute("aria-pressed", String(record?.verdict === button.dataset.verdict)));
    panel.querySelector(".review-note").innerHTML = reviewNote(record);
  });
  if (verdict === "undo" && state.reviewed === "reviewed") { render(); $("workspace").focus(); }
}
function reviewNote(record) {
  return record ? `已记录：${escapeHTML(groupLabels[record.verdict] || record.verdict)} · ${escapeHTML(timeLabel(record.reviewed_at))} ${actionButton("撤销本项", {class: "undo", "data-verdict": "undo"})}` : "未审核。未选择不会被视为通过。";
}
function verdictPanel(id, kind, labels) {
  const record = currentReview(id);
  return `<div class="verdict" data-review-id="${escapeHTML(id)}" data-review-kind="${kind}"><p class="question">查看组内图片后，哪一种关系最符合？</p><p class="helper">几乎相同：画面与目标基本不变；同地点有变化：地标对应，但目标或视角改变；仅类型相似：道路类型接近；混杂：不同配对关系不同。</p><div class="verdict-buttons">${Object.entries(labels).map(([value, label]) => actionButton(label, {"data-verdict": value, "aria-pressed": record?.verdict === value})).join("")}</div><label class="note-label" for="note-${id}">补充说明（可选，混杂组可写图片 ID）</label><textarea id="note-${id}" rows="2" maxlength="2000" placeholder="例如：A 与 B 同地点，C 看起来无关">${escapeHTML(record?.note || "")}</textarea><button data-save-note="${id}">保存说明</button><div class="review-note">${reviewNote(record)}</div></div>`;
}
function imageCard(key, interactive = true) {
  const im = catalog.images[key];
  const labels = im.period === "Daytime" ? {...nightLabels, mismatch: "疑似夜间"} : nightLabels;
  return `<article class="image-card"><div class="image-frame"><img src="/thumb/${encodeURIComponent(key)}" alt="${escapeHTML(`${key}，官方标签 ${im.period}，${im.city} ${im.location}`)}" width="960" height="540" loading="lazy"><span class="image-error" hidden>图片读取失败或来源校验失败。<br>请检查服务与本地数据。</span></div><div class="image-caption"><div class="image-id-line"><span class="image-id">${escapeHTML(key)}</span><span class="raw-tag">RAW / ${im.official_split.toUpperCase()}</span></div><p class="filename">${escapeHTML(im.file_name)}</p><div class="metadata"><span>官方时段 <strong>${escapeHTML(im.period)}</strong></span><span>${escapeHTML(im.city)}</span><span>${escapeHTML(im.weather)}</span><span>${escapeHTML(im.location)}</span></div><details class="image-source"><summary>原始路径与校验值</summary><p>${escapeHTML(im.raw_path)}</p><p>标注：${escapeHTML(im.annotation_file)} / images[id=${im.id}]</p><p>SHA-256 ${im.sha256}</p></details><a class="original-link" href="/image/${encodeURIComponent(key)}" target="_blank" rel="noopener">查看 1920 × 1080 原图 ↗</a>${interactive ? verdictPanel(key, "image", labels) : ""}</div></article>`;
}
function imageMatches(im) {
  const query = state.q.toLocaleLowerCase("zh-CN");
  return (state.city === "all" || im.city === state.city) && (!query || [im.key, im.file_name, im.city, im.location, im.weather, im.period].join(" ").toLocaleLowerCase("zh-CN").includes(query));
}
function statusMatches(id) { const record = state.view === "groups" ? (state.mode === "history" ? true : Boolean(currentReview(id))) : Boolean(reviews[id]); return state.reviewed === "all" || (state.reviewed === "reviewed" ? record : !record); }
function nightItems() { return catalog.sample_sets[state.mode].filter((key) => imageMatches(catalog.images[key]) && statusMatches(key)); }
function groupItems() {
  const ids = state.mode === "history" ? catalog.historical_group_ids : catalog.sample_group_ids;
  const byId = new Map(catalog.groups.map((g) => [g.id, g]));
  const groups = ids.map((id) => byId.get(id));
  return groups.filter((group) => statusMatches(group.id) && (group.id.includes(state.q.toLowerCase()) && state.city === "all" || group.keys.some((key) => imageMatches(catalog.images[key]))));
}
function renderGroup(group) {
  const visible = group.keys.slice(0, 6);
  const pair = pairChoices[group.id] || group.farthest_pair;
  const previous = group.previous_review;
  const historical = state.mode === "history";
  const priorLabel = previous?.original_verdict === "scene" ? "仅场景接近" : "同场景有时间变化（未确认相邻帧）";
  const compare = (label, edge) => actionButton(label, {"data-compare-group": group.id, "data-pair-a": edge.a, "data-pair-b": edge.b, "data-pair-p": edge.phash_distance, "data-pair-d": edge.dhash_distance});
  const position = catalog.sample_group_ids.indexOf(group.id) + 1;
  const metadataWarning = group.attributes.official_split.length > 1 ? "跨官方来源的宽松候选，需看原图确认；不代表已发现泄漏。" : "";
  return `<div class="group-header"><div><h3>${historical ? "上轮记录" : `本轮第 ${position} 组`} <span class="group-code">${group.id}</span></h3><p>${fmt(group.keys.length)} 张图片 · ${fmt(group.edges.length)} 条候选边 · ${escapeHTML(group.attributes.period.join(" / "))}</p></div><span class="raw-tag">${escapeHTML(historical ? "历史判断保留" : group.selection_reasons.join(" / "))}</span></div>${metadataWarning ? `<p class="explanation">${metadataWarning}</p>` : ""}${previous ? `<p class="history-note">上轮人工判断：${priorLabel}。原始记录保留，本页不自动升级为“近重复”。</p>` : ""}<section class="pair-panel"><div class="pair-toolbar"><h4>重点并排对照</h4>${compare("看组内哈希差异最大的一对", group.farthest_pair)}${compare("看最接近的一条候选边", group.closest_pair)}</div><p class="helper">当前 ${escapeHTML(pair.a)} ↔ ${escapeHTML(pair.b)} · pHash ${pair.phash_distance} / dHash ${pair.dhash_distance}。距离只辅助核对，不是人工结论。</p><div class="comparison-grid">${[pair.a, pair.b].map((key) => imageCard(key, false)).join("")}</div></section><details class="evidence-panel" ${group.keys.length <= 2 ? "" : "open"}><summary>查看完整组 · 共 ${group.keys.length} 张${group.keys.length > 6 ? "（下方可展开其余图片）" : ""}</summary><div class="image-grid">${visible.map((key) => imageCard(key, false)).join("")}</div>${group.keys.length > 6 ? `<details class="evidence-panel"><summary>展开其余 ${group.keys.length - 6} 张组内图片</summary><div class="image-grid">${group.keys.slice(6).map((key) => imageCard(key, false)).join("")}</div></details>` : ""}</details>${historical ? "" : `<section class="group-actions">${verdictPanel(group.id, "group", groupLabels)}</section>`}<p class="explanation">${group.origin === "expanded" ? "这是原候选之外的补查组，使用较宽的检索范围（pHash ≤ 10 / dHash ≤ 16），尚未确认实际相似。" : "这是原始哈希候选组（pHash ≤ 6 / dHash ≤ 10）。"}${group.chain_warning ? " 组内存在超出原阈值的配对，请检查是否链式误连。" : ""} 候选组不是官方 sequence。请查看完整组后判断；只要组内关系不一致，就选“组内混杂”。</p><details class="evidence-panel"><summary>具体配对与距离 · ${fmt(group.edges.length)} 条</summary><div class="edge-table"><table><thead><tr><th scope="col">图片 A</th><th scope="col">图片 B</th><th scope="col">pHash</th><th scope="col">dHash</th><th scope="col">对照</th></tr></thead><tbody>${group.edges.map((edge) => `<tr><td>${escapeHTML(edge.a)}</td><td>${escapeHTML(edge.b)}</td><td>${edge.phash_distance}</td><td>${edge.dhash_distance}</td><td>${compare("并排看这一对", edge)}</td></tr>`).join("")}</tbody></table></div></details>`;
}
function renderSource() {
  const a = catalog.alignment;
  return `<div class="source-content"><section class="evidence-panel"><h3>已核对：下载范围与官方 train / val 对齐</h3><ul class="check-list"><li>当前官方下载页仍指向相同 Google Drive 文件；本地完整包哈希保持一致。</li><li>重新获取官方包开头的两份标注 JSON，与本地原始文件逐字节一致。</li><li>全部 10,000 张本地图片文件名与对应官方 JSON 的集合完全匹配。</li><li>8 张远端原图抽查与本地文件哈希一致，包括时段边界疑点与相似图片案例。</li><li>本地仅有原始 train / val 文件夹，没有 night 文件夹，也没有创建自定义实验划分。</li></ul><p>核对时间：${timeLabel(a.checked_at_utc)}（北京时间）。来源对齐通过，不代表标签已通过人工审核。</p><p><a href="https://soda-2d.github.io/download.html" target="_blank" rel="noopener">官方下载页 ↗</a>　<a href="https://soda-2d.github.io/documentation.html" target="_blank" rel="noopener">官方格式说明 ↗</a>　<a href="https://arxiv.org/html/2106.11118v3" target="_blank" rel="noopener">论文第 3.3 节 ↗</a></p></section><section class="evidence-panel"><h3>官方划分与本地用途</h3><div class="table-scroll"><table><thead><tr><th scope="col">官方部分</th><th scope="col">图片</th><th scope="col">本地状态</th><th scope="col">时段标签</th></tr></thead><tbody><tr><td>train</td><td>5,000</td><td>原目录完整保留</td><td>5,000 Daytime</td></tr><tr><td>val</td><td>5,000</td><td>原目录完整保留</td><td>3,344 Daytime / 1,656 Night</td></tr><tr><td>test</td><td>10,000</td><td>按首版计划未下载</td><td>不参与当前复核</td></tr></tbody></table></div><p>“夜间候选”仅表示 <code>instance_val.json → images[].period == "Night"</code>。网页按官方 split 与图片 ID 联合取图；train:1 与 val:1 是不同对象。</p></section><section class="evidence-panel"><h3>需要纠正的理解：官方隔离声明的范围</h3><p>论文声明官方 train / val / test 之间的 sequence 互斥；已查官方下载页、格式说明与站点仓库未找到另行发布的采集组映射。下载包内的非图片文件仅有两份标注 JSON，公开格式也未列 sequence 字段。</p><p>因此“没有重新划分用的 sequence 映射”不能直接认定为下载缺件或官方隔离失效。我们拟定的内部训练／开发／最终测试划分是额外要求，当前仍未冻结。</p><p class="explanation">昼夜直接沿用官方 period，不再单独研究。当前人工任务仅为相似场景确认；不自动删图或正式划分。</p></section><details class="evidence-panel"><summary>完整包与两份原始标注的 SHA-256</summary><p>官方文件 ID：<code>${escapeHTML(a.official_trainval_link)}</code></p><p>本地完整包：<code>${a.local_archive_sha256}</code></p>${Object.entries(catalog.annotation_hashes).map(([split, hash]) => `<p>${split}：<code>${hash}</code></p>`).join("")}<p>这些是 HTTPS 来源比较与本地哈希记录；未取得发布者签名。</p></details></div>`;
}
function bindImages() {
  document.querySelectorAll(".image-frame img").forEach((image) => {
    const error = () => {
      image.hidden = true; image.nextElementSibling.hidden = false;
      image.closest(".image-card").querySelectorAll("button[data-verdict]").forEach((button) => { if (button.dataset.verdict !== "undo") button.disabled = true; });
      document.querySelectorAll(".group-actions button[data-verdict]").forEach((button) => { if (!["uncertain", "undo"].includes(button.dataset.verdict)) button.disabled = true; });
    };
    image.addEventListener("error", error, {once: true});
    if (image.complete && image.naturalWidth === 0) error();
  });
}
function render() {
  if (!catalog) return;
  const options = modeOptions[state.view];
  if (options && !options.some(([value]) => value === state.mode)) state.mode = "sample";
  if (!["all", "Shanghai", "Shenzhen", "Guangzhou"].includes(state.city)) state.city = "all";
  if (!["all", "reviewed", "unreviewed"].includes(state.reviewed)) state.reviewed = "all";
  document.title = `${state.view === "night" ? "夜间属性" : state.view === "groups" ? "相似候选组" : "来源与划分"} · 数据复核台`;
  document.querySelectorAll(".view-nav a").forEach((a) => { if (a.dataset.view === state.view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  $("view-title").textContent = titles[state.view];
  $("filters").hidden = state.view === "source";
  if (options) $("mode").innerHTML = options.map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
  $("mode").value = state.mode; $("city").value = state.city; $("reviewed").value = state.reviewed;
  if (!composing) $("search").value = state.q;
  $("clear-search").hidden = !state.q;
  $("clear-search").disabled = !state.q;
  if (state.view === "source") {
    $("view-description").textContent = "先区分包是否完整、标签是否来自官方，以及额外划分能否实施。";
    $("range-status").textContent = "来源核验已通过 · 昼夜沿用官方 period · 相似场景待确认";
    $("workspace").innerHTML = renderSource(); $("pagination").hidden = true; writeURL(); return;
  }
  const items = state.view === "night" ? nightItems() : groupItems();
  const pageSize = state.view === "night" ? 6 : 1;
  const pages = Math.max(1, Math.ceil(items.length / pageSize));
  state.page = Math.min(state.page, pages - 1);
  const start = state.page * pageSize, end = Math.min(start + pageSize, items.length);
  $("range-status").textContent = items.length ? `当前 ${fmt(start + 1)}–${fmt(end)} / ${fmt(items.length)} ${state.view === "night" ? "张图片" : "个候选组"}` : "没有匹配的审核内容";
  const description = state.view === "groups" ? (state.mode === "history" ? "上轮 12 组的人工判断只读保留，不计入本轮 50 组，不要求重新审核。" : `本轮固定 ${catalog.sample_group_ids.length} 个新组：夜间覆盖 18、大组 10、链式关系 8、场景覆盖 8、原候选外补查 6。按风险和覆盖抽样，不用于推算重复率。`) : "只读查看官方 period；昼夜问题已关闭，不再收集新的时段判断。此前图片审核记录保持原样。";
  $("view-description").textContent = description;
  $("workspace").innerHTML = items.length ? (state.view === "night" ? `<div class="image-grid">${items.slice(start, end).map((key) => imageCard(key, false)).join("")}</div>` : renderGroup(items[start])) : '<div class="state-panel"><h3>没有匹配的图片或分组</h3><p>试着清除搜索，或将城市、审核状态切回全部。</p><button id="reset-filters">重置筛选</button></div>';
  $("pagination").hidden = !items.length;
  $("previous").disabled = state.page === 0; $("next").disabled = state.page >= pages - 1;
  $("page-status").textContent = `第 ${state.page + 1} / ${pages} 页`;
  bindImages(); writeURL();
}
document.addEventListener("click", (event) => {
  const link = event.target.closest("a[data-view]");
  if (link && !event.ctrlKey && !event.metaKey) {
    event.preventDefault(); state = {view: link.dataset.view, mode: "sample", q: "", city: "all", reviewed: "all", page: 0}; render();
  }
  const button = event.target.closest("button[data-verdict]");
  if (button) { const panel = button.closest("[data-review-id]"); saveReview(panel.dataset.reviewId, panel.dataset.reviewKind, button.dataset.verdict); }
  const noteButton = event.target.closest("button[data-save-note]");
  if (noteButton) {
    const id = noteButton.dataset.saveNote;
    const record = currentReview(id);
    if (!record) announce("请先选择本组关系，再保存补充说明。");
    else saveReview(id, "group", record.verdict);
  }
  const compareButton = event.target.closest("button[data-compare-group]");
  if (compareButton) {
    const id = compareButton.dataset.compareGroup;
    const note = document.querySelector(`[data-review-id="${id}"] textarea`);
    // Preserve an in-progress explanation while changing the comparison pair.
    const draft = note?.value;
    pairChoices[id] = {a: compareButton.dataset.pairA, b: compareButton.dataset.pairB, phash_distance: Number(compareButton.dataset.pairP), dhash_distance: Number(compareButton.dataset.pairD)};
    render();
    const replacement = document.querySelector(`[data-review-id="${id}"] textarea`);
    if (replacement && draft !== undefined) replacement.value = draft;
    $("workspace").focus();
  }
  if (event.target.id === "reset-filters") { state.q = ""; state.city = "all"; state.reviewed = "all"; state.page = 0; render(); }
});
for (const key of ["mode", "city", "reviewed"]) $(key).addEventListener("change", (event) => { state[key] = event.target.value; state.page = 0; render(); });
$("search").addEventListener("compositionstart", () => { composing = true; });
$("search").addEventListener("compositionend", (event) => { composing = false; state.q = event.target.value; state.page = 0; render(); });
$("search").addEventListener("input", (event) => { if (composing || event.isComposing) return; state.q = event.target.value; state.page = 0; render(); });
$("clear-search").addEventListener("click", () => { state.q = ""; state.page = 0; render(); $("search").focus(); });
$("previous").addEventListener("click", () => { if (state.page) { state.page--; render(); $("workspace").focus(); } });
$("next").addEventListener("click", () => {
  const id = document.querySelector("[data-review-id]")?.dataset.reviewId;
  // Saving in the unreviewed filter removes the current item on next render.
  // Keep its index so the following group is not skipped.
  if (!(state.view === "groups" && state.reviewed === "unreviewed" && id && currentReview(id))) state.page++;
  render(); $("workspace").focus();
});
window.addEventListener("popstate", () => { state = readURL(); render(); });
window.addEventListener("storage", (event) => { if (event.key === storageKey) { reviews = loadStored(); updateProgress(); render(); announce("已同步当前浏览器其他页面的审核记录。"); } });
$("export").addEventListener("click", () => {
  if (!catalog || !Object.keys(reviews).length) return;
  const records = Object.values(reviews).map((record) => ({...record, images: (record.kind === "image" ? [record.object_id] : catalog.groups.find((g) => g.id === record.object_id)?.keys || []).map((key) => ({key, file_name: catalog.images[key].file_name, official_split: catalog.images[key].official_split, original_period: catalog.images[key].period, sha256: catalog.images[key].sha256}))}));
  const payload = {schema_version: 2, campaign_id: catalog.scene_screening.campaign_id, sample_group_ids: catalog.sample_group_ids, sampling: catalog.scene_screening, dataset_fingerprint: catalog.dataset_fingerprint, annotation_sha256: catalog.annotation_hashes, exported_at: new Date().toISOString(), raw_data_modified: false, formal_split_created: false, historical_group_judgments: catalog.groups.filter((g) => g.previous_review).map((g) => g.previous_review), reviews: records};
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: "application/json;charset=utf-8"}));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = `soda10m-human-review-${Date.now()}.json`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000); announce(`已导出 ${records.length} 项人工判断。原始标签未修改。`);
});
async function start() {
  document.title = "正在读取清单 · 数据复核台";
  $("source-state").textContent = "正在核验来源…";
  $("source-state").dataset.state = "loading";
  $("filters").hidden = true;
  $("pagination").hidden = true;
  $("range-status").textContent = "正在读取清单…";
  $("workspace").innerHTML = '<div class="state-panel"><span class="spinner" aria-hidden="true"></span><h3>正在读取核验后的图片清单</h3></div>';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch("/api/catalog", {signal: controller.signal});
    if (!response.ok) throw new Error("清单服务返回错误");
    catalog = await response.json();
    if (!catalog.alignment.fresh_remote_annotations_match_raw) throw new Error("来源核验未通过");
    storageKey = `soda10m-review-v1:${catalog.dataset_fingerprint}`;
    reviews = loadStored(); updateProgress();
    $("source-state").textContent = "✓ 来源已对齐";
    $("source-state").dataset.state = "passed";
    $("dataset-counts").innerHTML = `<strong>10,000</strong> 张原图<br><strong>1,656</strong> 个官方 Night 标签`;
    render();
  } catch (error) {
    document.title = "清单读取失败 · 数据复核台";
    $("source-state").textContent = "清单未就绪";
    $("source-state").dataset.state = "error";
    $("view-description").textContent = "清单尚未就绪，请确认本地服务后重试。";
    $("range-status").textContent = "尚未开放人工审核";
    $("workspace").innerHTML = '<div class="state-panel"><h3>暂时无法读取审核清单</h3><p>请确认本地服务正在运行，随后重试。尚未保存任何新判断。</p><button id="retry">重新读取</button></div>';
    $("retry").addEventListener("click", start, {once: true});
  } finally { clearTimeout(timeout); }
}
start();
