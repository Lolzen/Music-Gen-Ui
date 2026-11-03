// Music-Gen-UI front end. Plain ES module, no build step.
const $ = (sel, root = document) => root.querySelector(sel);
const form = $("#song-form");
const STATUS = { queued: "wartet", running: "läuft", done: "fertig", failed: "Fehler", cancelled: "abgebrochen" };
const STAGES = {
  download: "Modell herunterladen/prüfen", load: "Modell laden", gguf: "GGUF laden", score: "Partitur planen",
  audio: "Audio generieren", nar: "Klangsynthese", vae: "Dekodieren",
};
const TOKENS_PER_SECOND = 25; // YuE2 semantic token rate

function progressText(p) {
  if (!p || !p.stage) return "";
  const label = STAGES[p.stage] || p.stage;
  if (p.total) return `${label} · ${Math.min(100, Math.round((100 * p.done) / p.total))}%`;
  if (p.stage === "download") return p.done ? `${label} · ${(p.done / 2 ** 30).toFixed(2)} GB im Cache` : `${label} …`;
  if (p.stage === "audio") return `${label} · ${p.done} Tokens ≈ ${Math.round(p.done / TOKENS_PER_SECOND)} s Musik`;
  if (p.stage === "score") return `${label} · ${p.done} Tokens`;
  return label;
}

let info = null;

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail
        : (body.detail || []).map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    } catch { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

// ---------------------------------------------------------------- voices
function addVoice(voice = { id: "lead", gender: "female", descriptors: [] }) {
  const node = $("#voice-template").content.firstElementChild.cloneNode(true);
  $(".v-id", node).value = voice.id;
  $(".v-gender", node).value = voice.gender;
  $(".v-desc", node).value = (voice.descriptors || []).join(", ");
  $(".v-remove", node).addEventListener("click", () => { node.remove(); schedulePreview(); });
  $("#voices").append(node);
}

function nextVoiceId() {
  const used = new Set([...document.querySelectorAll(".v-id")].map((i) => i.value));
  for (const id of ["lead", "duo", "voice3", "voice4", "voice5", "voice6"]) if (!used.has(id)) return id;
  return `v${Date.now() % 1000}`;
}

// ---------------------------------------------------------------- LoRAs
const HALF_LABEL = { ar: "Komposition", nar: "Klang" };

function loraEntry(path) {
  return modelCatalog?.lora.find((e) => e.value === path);
}

function addLoraRow(path, strength = 1, { fromUser = false } = {}) {
  if ([...document.querySelectorAll("#loras .lora")].some((r) => r.dataset.path === path)) return;
  const entry = loraEntry(path);
  const node = $("#lora-template").content.firstElementChild.cloneNode(true);
  node.dataset.path = path;
  $(".lora-name", node).textContent = entry?.label || path.split("/").pop();
  const halves = entry?.meta?.halves || [];
  $(".lora-halves", node).innerHTML = halves.map((h) => `<span class="halfbadge">${HALF_LABEL[h] || h}</span>`).join("");
  const slider = $(".lora-strength", node);
  slider.value = strength;
  $(".lora-value", node).textContent = Number(strength).toFixed(2);
  slider.addEventListener("input", () => { $(".lora-value", node).textContent = Number(slider.value).toFixed(2); });
  $(".lora-remove", node).addEventListener("click", () => { node.remove(); schedulePreview(); });
  $("#loras").append(node);
  if (fromUser) {
    const trigger = entry?.meta?.trigger;
    const style = form.elements.style;
    if (trigger && !style.value.toLowerCase().includes(trigger.toLowerCase())) {
      style.value = style.value.trim() ? `${trigger}, ${style.value.trim()}` : trigger;
    }
    const recommended = entry?.meta?.recommended || {};
    if (recommended.plan_mode || halves.includes("ar")) form.elements.plan_mode.value = recommended.plan_mode || "off";
  }
  schedulePreview();
}

function fillLoraPicker() {
  const select = $("#lora-pick");
  const entries = modelCatalog?.lora || [];
  select.innerHTML = entries.length ? "" : '<option value="">– models/lora/ ist leer –</option>';
  for (const e of entries) {
    const o = document.createElement("option");
    o.value = e.value;
    const halves = (e.meta?.halves || []).map((h) => HALF_LABEL[h] || h).join("+");
    o.textContent = `${e.label}${halves ? ` · ${halves}` : ""}${e.engine ? "" : " ⚠"}`;
    o.disabled = !e.engine;
    o.title = e.note || "";
    select.append(o);
  }
}

// ---------------------------------------------------------------- form <-> spec
function sampling(prefix) {
  const out = {};
  for (const key of ["temperature", "top_p"]) {
    const input = form.elements[`${prefix}_${key}`];
    if (input && !input.disabled) out[key] = Number(input.value);
  }
  return out;
}

function readSpec() {
  const f = form.elements;
  const num = (v) => (v === "" ? null : Number(v));
  return {
    title: f.title.value || "Untitled",
    style: f.style.value,
    language: f.language.value || null,
    bpm: num(f.bpm.value),
    instrumental: f.instrumental.checked,
    voices: [...document.querySelectorAll("#voices .voice")].map((n) => ({
      id: $(".v-id", n).value || "lead",
      gender: $(".v-gender", n).value,
      descriptors: $(".v-desc", n).value.split(",").map((s) => s.trim()).filter(Boolean),
    })),
    lyrics: f.lyrics.value,
    style_override: f.style_override.value || null,
    plan_mode: f.plan_mode.value,
    abc: f.abc.value.trim() ? f.abc.value : null,
    seed: num(f.seed.value),
    cfg_scale: f.cfg_scale.disabled ? null : Number(f.cfg_scale.value),
    audio_sampling: sampling("audio"),
    candidates: Number(f.candidates.value) || 1,
    loras: [...document.querySelectorAll("#loras .lora")].map((r) => ({
      path: r.dataset.path, strength: Number($(".lora-strength", r).value) })),
  };
}

function setSlider(name, value) {
  const input = form.elements[name];
  const toggle = form.querySelector(`[data-enable="${name}"]`);
  const on = value !== null && value !== undefined;
  toggle.checked = on;
  input.disabled = !on;
  if (on) input.value = value;
  updateOutputs();
}

function writeSpec(spec) {
  const f = form.elements;
  const set = (name, v) => { if (v !== undefined) f[name].value = v ?? ""; };
  ["title", "style", "language", "bpm", "lyrics", "style_override", "plan_mode", "abc", "seed", "candidates"]
    .forEach((k) => set(k, spec[k]));
  if (spec.instrumental !== undefined) f.instrumental.checked = spec.instrumental;
  if (spec.voices) { $("#voices").innerHTML = ""; spec.voices.forEach(addVoice); }
  if (spec.loras) { $("#loras").innerHTML = ""; spec.loras.forEach((l) => addLoraRow(l.path, l.strength)); }
  if ("cfg_scale" in spec) setSlider("cfg_scale", spec.cfg_scale);
  if (spec.audio_sampling) {
    setSlider("audio_temperature", spec.audio_sampling.temperature);
    setSlider("audio_top_p", spec.audio_sampling.top_p);
  }
  schedulePreview();
}

function updateOutputs() {
  form.querySelectorAll("output[data-for]").forEach((o) => {
    const input = form.elements[o.dataset.for];
    o.textContent = input.disabled ? "Standard" : input.value;
  });
}

// ---------------------------------------------------------------- preview
let previewTimer = null;
function schedulePreview() {
  schedulePrediction();
  clearTimeout(previewTimer);
  previewTimer = setTimeout(preview, 300);
}

async function preview() {
  try {
    const req = await api("/api/preview", { method: "POST", body: JSON.stringify(readSpec()) });
    $("#preview-style").textContent = req.style;
    $("#preview-lyrics").textContent = req.lyrics;
    $("#preview-warnings").innerHTML = "";
    for (const w of req.warnings) {
      const li = document.createElement("li");
      li.textContent = w;
      $("#preview-warnings").append(li);
    }
    $("#form-error").textContent = "";
  } catch (err) {
    $("#form-error").textContent = err.message;
  }
}

// ---------------------------------------------------------------- jobs
function jobCard(job) {
  const card = document.createElement("article");
  card.className = `job ${job.status}`;
  card.dataset.id = job.id;
  const p = job.progress || {};
  const pct = p.total ? Math.min(100, Math.round((100 * p.done) / p.total)) : null;
  card.innerHTML = `
    <header><strong></strong><span class="badge">${STATUS[job.status] || job.status}</span></header>
    <p class="muted meta"></p>
    ${job.status === "running" ? `<div class="bar"><div style="width:${pct ?? 100}%" class="${pct === null ? "indeterminate" : ""}"></div></div>
      <p class="muted">${progressText(p) || "startet …"}</p>` : ""}
    ${job.status === "done" ? `<audio controls preload="none" src="/api/jobs/${job.id}/audio"></audio>` : ""}
    ${job.error ? `<p class="error"></p>` : ""}
    <div class="actions"></div>`;
  $("strong", card).textContent = job.title;
  const r = job.result || {};
  const took = job.started && job.finished ? Math.round(job.finished - job.started) : null;
  $(".meta", card).textContent = [`Seed ${job.request.seed}`, r.duration_s ? `${r.duration_s} s Musik` : null,
    took !== null && job.status === "done" ? `in ${took} s generiert` : null,
    r.truncated ? `⚠ abgeschnitten (${(r.truncated_parts || []).map((p) => ({ score: "Partitur-Limit",
      audio: "Audio-Limit ≈ 6 min" })[p] || "Token-Limit").join(", ") || "Token-Limit"} erreicht)` : null].filter(Boolean).join(" · ");
  if (job.error) $(".error", card).textContent = job.error;

  const actions = $(".actions", card);
  const button = (label, fn, cls = "ghost") => {
    const b = document.createElement("button");
    b.type = "button"; b.className = cls; b.textContent = label;
    b.addEventListener("click", fn);
    actions.append(b);
  };
  if (job.status === "queued" || job.status === "running") {
    button("Abbrechen", () => api(`/api/jobs/${job.id}/cancel`, { method: "POST" }).then(refreshJobs));
  } else {
    button("Als Vorlage", () => writeSpec({ ...job.spec, seed: job.request.seed, candidates: 1 }));
    if (job.status === "failed" || job.status === "cancelled") {
      button("Wiederholen", () => submit({ ...job.spec, seed: job.request.seed, candidates: 1 }));
    }
    button("Neu würfeln", () => submit({ ...job.spec, seed: null, candidates: 1 }));
    if (r.has_score) button("Partitur", () => showScore(job.id));
    if (r.has_score) button("Partitur übernehmen", () => useScore(job));
    button("Löschen", () => confirm(`„${job.title}“ löschen?`) &&
      api(`/api/jobs/${job.id}`, { method: "DELETE" }).then(refreshJobs));
  }
  return card;
}

let pollTimer = null;
async function refreshJobs() {
  clearTimeout(pollTimer);
  let jobs = [];
  try { jobs = await api("/api/jobs"); } catch { /* server restarting */ }
  const list = $("#jobs");
  // Keep cards whose state didn't change so playing audio is not interrupted.
  const existing = new Map([...list.children].map((c) => [c.dataset.id, c]));
  const fragment = document.createDocumentFragment();
  for (const job of jobs) {
    const key = JSON.stringify([job.status, job.progress, job.error]);
    const old = existing.get(job.id);
    if (old && old.dataset.key === key) { fragment.append(old); continue; }
    const card = jobCard(job);
    card.dataset.key = key;
    fragment.append(card);
  }
  list.replaceChildren(fragment);
  if (!jobs.length) list.innerHTML = `<p class="muted">Noch nichts generiert.</p>`;
  const active = jobs.some((j) => j.status === "queued" || j.status === "running");
  pollTimer = setTimeout(refreshJobs, active ? 1000 : 5000);
}

async function submit(spec) {
  $("#form-error").textContent = "";
  try {
    await api("/api/jobs", { method: "POST", body: JSON.stringify(spec) });
    refreshJobs();
  } catch (err) {
    $("#form-error").textContent = err.message;
  }
}

async function showScore(id) {
  const text = await fetch(`/api/jobs/${id}/score`).then((r) => r.text());
  $("#score-text").textContent = text;
  $("#score-render").innerHTML = "";
  if (window.ABCJS) window.ABCJS.renderAbc("score-render", text, { responsive: "resize" });
  $("#score-dialog").showModal();
}

async function useScore(job) {
  const text = await fetch(`/api/jobs/${job.id}/score`).then((r) => r.text());
  writeSpec({ ...job.spec, abc: text, plan_mode: job.spec.plan_mode === "off" ? "full" : job.spec.plan_mode,
    seed: job.request.seed, candidates: 1 });
  form.querySelectorAll("details")[1].open = true;
}

// ---------------------------------------------------------------- engine status & settings
function showBanner(html) {
  const banner = $("#banner");
  banner.hidden = !html;
  banner.innerHTML = html || "";
}

function renderEngine(e) {
  const box = $("#engine-status");
  const label = $("#engine-label");
  box.className = `status ${e.engine === "mock" ? "mock" : e.state}`;
  const STATE = { unloaded: "nicht geladen", loading: "lädt", ready: "bereit", busy: "generiert", error: "Fehler",
    training: "LoRA-Training läuft (Songs warten)" };
  if (e.state === "training") {
    box.className = "status busy";
    label.textContent = e.message || STATE.training;
  } else if (e.engine === "mock") {
    label.textContent = "Mock-Engine · Platzhalter-Audio";
  } else {
    const mode = [e.info.gguf ? `GGUF ${{ bf16: "BF16", balanced: "ausgewogen", max: "maximal" }[e.info.gguf_mode]}` : null,
      e.info.quantization === "fp8" ? "FP8" : null, { low_vram: "Offload", offload_ar: "Offload (AR-Auslagerung)", unsupported: "Offload nicht verfügbar" }[e.info.low_vram_mode] || null,
      e.info.backend !== "torch" ? e.info.backend : null].filter(Boolean).join(", ");
    const extra = e.state === "loading" ? ` · ${progressText(e.progress) || "…"}` : "";
    label.textContent = `YuE2${mode ? ` (${mode})` : ""} · ${STATE[e.state] || e.state}${extra}`;
    label.title = e.info.yue2 ? `yue2-infer ${e.info.yue2.version}` +
      (e.info.yue2.stage_progress ? "" : " (Release-API: kein Fortschritt für Klangsynthese/Dekodieren)") : "";
  }
  $("#engine-load").hidden = !(e.engine === "yue2" && (e.state === "unloaded" || e.state === "error"));
  $("#engine-unload").hidden = !(e.engine === "yue2" && e.state === "ready");

  if (!e.installed.yue2 && e.engine === "yue2") {
    showBanner("YuE2 ist ausgewählt, aber nicht installiert. Installation: <code>./install.sh</code> " +
      "im Projektordner (siehe README) – oder unter ⚙ Einstellungen die Mock-Engine wählen.");
  } else if (!e.installed.yue2 && e.configured !== "mock") {
    showBanner("YuE2 ist nicht installiert – es läuft nur der Platzhalter-Modus. " +
      "Installation: <code>./install.sh</code> im Projektordner (siehe README).");
  } else if (e.engine === "mock") {
    showBanner("Mock-Engine aktiv: Es wird kein echtes Modell verwendet. Umstellen unter ⚙ Einstellungen → Engine.");
  } else if (e.state === "error") {
    showBanner("");
    const b = $("#banner");
    b.hidden = false;
    b.textContent = `Modell konnte nicht geladen werden: ${e.message}`;
  } else {
    showBanner("");
  }
}

// ---------------------------------------------------------------- memory widget + prediction
let prediction = null;          // prediction for the saved settings (header)
let lastEngineState = null;

function memRow(container, key, label) {
  let row = container.querySelector(`[data-key="${key}"]`);
  if (!row) {
    row = $("#memrow-template").content.firstElementChild.cloneNode(true);
    row.dataset.key = key;
    $(".memlabel", row).textContent = label;
    container.append(row);
  }
  return row;
}

function predictionTitle(pred, kind) {
  if (!pred) return "";
  const a = pred.assumptions;
  const lines = [`Prognose (${pred[kind].basis}) für einen Song bis ${Math.round(a.song_seconds / 60)} min`,
    `Modellgewichte: ${pred.weights_gib} GB${pred.file_gib ? ` (Datei ${pred.file_gib} GB)` : ""}`];
  if (kind === "vram") {
    for (const [phase, gib] of Object.entries(pred.vram.phases)) lines.push(`  ${phase}: ~${gib} GB`);
    lines.push(`CUDA-Graphs: ${a.cuda_graphs ? "ja" : "nein"}, FP8: ${a.fp8 ? "ja" : "nein"}, Offload: ${a.offload ? "ja" : "nein"}`);
  } else {
    lines.push(`Zwischen Songs: ~${pred.ram.idle_gib} GB für Music-Gen-UI`);
  }
  if (pred[kind].estimate_gib) lines.push(`(Schätzung wäre ${pred[kind].estimate_gib} GB)`);
  if (pred.calibration?.samples) {
    const f = Object.entries(pred.calibration.factors || {}).map(([k, v]) => `${k} ×${v}`).join(", ");
    lines.push(`Kalibriert mit ${pred.calibration.samples} Song(s)${f ? `: ${f}` : ""}`);
  }
  if (pred.measured_song_seconds) lines.push(`Gemessen am längsten bisherigen Song: ${Math.round(pred.measured_song_seconds)} s`);
  return lines.join("\n");
}

function drawMemory(container, status, pred) {
  if (!status) return;
  const proc = status.process || {};
  const rows = [];
  if (status.gpu?.cuda) {
    const ours = (proc.vram_gib || 0) + (proc.cuda_active ? 0.5 : 0);
    const others = Math.max(status.gpu.used_gib - ours, 0);
    const cap = pred?.vram.cap_gib;
    rows.push(["vram", "VRAM", status.gpu.total_gib, status.gpu.used_gib,
      pred ? others + pred.vram.peak_gib : null, 0.97,
      cap ? { at: others + cap + 0.5, over: pred.vram.peak_gib > cap + 0.5 } : null]);
  } else {
    rows.push(["vram", "VRAM", null, null, pred ? pred.vram.peak_gib : null, 1]);
  }
  if (status.ram) {
    rows.push(["ram", "RAM", status.ram.total_gib, status.ram.used_gib,
      pred ? Math.max(status.ram.used_gib - (proc.rss_gib || 0), 0) + pred.ram.peak_gib : null, 0.95]);
  }
  for (const [key, label, total, used, predicted, limit, capInfo] of rows) {
    const row = memRow(container, key, label);
    const bar = $(".membar", row);
    const text = $(".memtext", row);
    const basis = pred ? pred[key].basis : "";
    if (total === null) {  // no GPU visible (e.g. PyTorch not installed)
      bar.hidden = true;
      text.innerHTML = predicted !== null
        ? `keine GPU erkannt · <span class="predtext">Bedarf ~${predicted.toFixed(1)} GB (${basis})</span>` : "keine GPU erkannt";
    } else {
      bar.hidden = false;
      $(".fill", bar).style.width = `${Math.min(100, (100 * used) / total)}%`;
      const pctPred = predicted === null ? 0 : Math.min(100, (100 * predicted) / total);
      $(".pred", bar).style.width = `${pctPred}%`;
      $(".pred", bar).hidden = predicted === null;
      const over = predicted !== null && (predicted > total * limit || Boolean(capInfo?.over));
      bar.classList.toggle("over", over);
      let capLine = $(".cap", bar);
      if (capInfo) {
        if (!capLine) { capLine = document.createElement("div"); capLine.className = "cap"; bar.append(capLine); }
        capLine.style.left = `${Math.min(100, (100 * capInfo.at) / total)}%`;
        capLine.title = "Grenze für YuE2 (Gesamt-VRAM minus VRAM-Reserve)";
      } else if (capLine) {
        capLine.remove();
      }
      const why = capInfo?.over ? " – über der YuE2-Grenze (VRAM-Reserve/Offload/FP8 anpassen)" : "";
      text.innerHTML = `${used.toFixed(1)} / ${total.toFixed(0)} GB` + (predicted === null ? "" :
        ` · <span class="predtext">${over ? "⚠ " : ""}Prognose ~${predicted.toFixed(1)} GB (${basis})${why}</span>`);
    }
    row.title = predictionTitle(pred, key);
  }
}

function hintsFromSpec() {
  const spec = readSpec();
  return { plan_mode: spec.plan_mode, cfg_scale: spec.cfg_scale, audio_max_tokens: spec.audio_sampling?.max_tokens };
}

let predictTimer = null;
function schedulePrediction() {
  clearTimeout(predictTimer);
  predictTimer = setTimeout(fetchPrediction, 400);
}

async function fetchPrediction() {
  try {
    prediction = await api("/api/predict", { method: "POST", body: JSON.stringify({ spec: hintsFromSpec() }) });
  } catch {
    prediction = null;
  }
  drawMemory($("#memory"), lastStatus, prediction);
}

let dialogPredictTimer = null;
function scheduleDialogPrediction() {
  clearTimeout(dialogPredictTimer);
  dialogPredictTimer = setTimeout(async () => {
    const note = $("#memory-dialog-note");
    try {
      const pred = await api("/api/predict", { method: "POST",
        body: JSON.stringify({ settings: dialogValues(), spec: hintsFromSpec() }) });
      drawMemory($("#memory-dialog"), lastStatus, pred);
      note.textContent = `Modellgewichte im Speicher: ${pred.weights_gib} GB` +
        (pred.engine === "mock" ? " · Hinweis: Mock-Engine gewählt – die Prognose gilt für YuE2" : "") +
        ". Transparente Füllung = erwarteter Spitzenwert während eines Songs.";
    } catch (err) {
      note.textContent = `Keine Prognose: ${err.message}`;
    }
  }, 250);
}

let statusTimer = null;
let lastStatus = null;
async function refreshStatus() {
  clearTimeout(statusTimer);
  let delay = 4000;
  try {
    const s = await api("/api/status");
    renderEngine(s.engine);
    lastStatus = s;
    drawMemory($("#memory"), s, prediction);
    if (lastEngineState === "busy" && s.engine.state !== "busy") schedulePrediction();  // new measurement
    lastEngineState = s.engine.state;
    if (s.engine.state === "loading" || s.engine.state === "busy") delay = 1000;
  } catch { /* server restarting */ }
  statusTimer = setTimeout(refreshStatus, delay);
}

async function engineAction(path) {
  try {
    renderEngine(await api(path, { method: "POST" }));
  } catch (err) {
    showBanner("");
    $("#banner").hidden = false;
    $("#banner").textContent = err.message;
  }
  refreshStatus();
}

async function openSettings() {
  const { values, choices } = await api("/api/settings");
  const f = $("#settings-form").elements;
  for (const key of ["engine", "backend", "device", "memory_budget_gib", "model", "vae",
    "idle_placement", "vram_reserve_gib", "ram_guard_gib", "gguf_mode", "gpu_share"]) f[key].value = values[key];
  f.vram_mode.value = values.quantization === "fp8" ? "fp8" : "standard";
  f.offload.checked = values.offload;
  scheduleDialogPrediction();
  $("#vae-choices").innerHTML = choices.vae.map((v) => `<option value="${v}">`).join("");
  $("#settings-error").textContent = "";
  $("#settings-dialog").showModal();
}

function dialogValues() {
  const f = $("#settings-form").elements;
  const mode = f.vram_mode.value;
  return {
    engine: f.engine.value, backend: f.backend.value, device: f.device.value,
    memory_budget_gib: Number(f.memory_budget_gib.value), model: f.model.value, vae: f.vae.value,
    quantization: mode === "fp8" ? "fp8" : "none",
    offload: f.offload.checked, idle_placement: f.idle_placement.value, gguf_mode: f.gguf_mode.value,
    gpu_share: f.gpu_share.value,
    vram_reserve_gib: Number(f.vram_reserve_gib.value), ram_guard_gib: Number(f.ram_guard_gib.value),
  };
}

async function saveSettings() {
  const values = dialogValues();
  try {
    const res = await api("/api/settings", { method: "PUT", body: JSON.stringify(values) });
    renderEngine(res.engine);
    $("#settings-dialog").close();
    refreshStatus();
    loadModels();
    schedulePrediction();
  } catch (err) {
    $("#settings-error").textContent = err.message;
  }
}

function setupEngineControls() {
  $("#open-settings").addEventListener("click", openSettings);
  $("#engine-load").addEventListener("click", () => engineAction("/api/engine/load"));
  $("#engine-unload").addEventListener("click", () => engineAction("/api/engine/unload"));
  $("#settings-form").addEventListener("input", scheduleDialogPrediction);
  $("#settings-form").addEventListener("change", scheduleDialogPrediction);
  $("#settings-form").addEventListener("submit", (ev) => {
    if (ev.submitter && ev.submitter.value === "save") {
      ev.preventDefault();
      saveSettings();
    }
  });
}

// ---------------------------------------------------------------- model picker
let modelCatalog = null;

function entryText(e) {
  const bits = [e.quant, e.size_gib ? `${e.size_gib} GB` : null, e.format === "gguf" ? "GGUF" : null]
    .filter(Boolean).join(", ");
  return bits ? `${e.label} · ${bits}` : e.label;
}

function fillPicker(select, entries, selected, folder) {
  select.innerHTML = "";
  const groups = [["Lokal", entries.filter((e) => e.source === "local")],
    ["Hugging Face", entries.filter((e) => e.source === "hub")]];
  for (const [name, list] of groups) {
    const group = document.createElement("optgroup");
    group.label = name === "Lokal" ? `Lokal (${folder})` : name;
    if (!list.length && name === "Lokal") {
      const o = document.createElement("option");
      o.disabled = true; o.textContent = "– Ordner ist leer –";
      group.append(o);
    }
    for (const e of list) {
      const o = document.createElement("option");
      o.value = e.value; o.textContent = entryText(e) + (e.engine ? "" : " ⚠");
      o.title = e.note || "";
      o.disabled = !e.engine;
      group.append(o);
    }
    select.append(group);
  }
  if (![...select.options].some((o) => o.value === selected)) {
    const o = document.createElement("option");
    o.value = selected; o.textContent = `${selected} (benutzerdefiniert)`;
    select.prepend(o);
  }
  select.value = selected;
}

function showModelNote() {
  if (!modelCatalog) return;
  const notes = [];
  for (const kind of ["checkpoint", "vae"]) {
    const value = $(`#pick-${kind}`).value;
    const entry = modelCatalog[kind].find((e) => e.value === value);
    const GGUF_MODE = { bf16: "BF16 (entpackt)", balanced: "Ausgewogen", max: "Maximal (langsam)" };
    if (entry?.format === "gguf" && entry.engine && kind === "checkpoint") {
      notes.push(`${entry.label}: GGUF ${entry.quant || ""} · Modus ${GGUF_MODE[modelCatalog.gguf_mode]} (⚙ Einstellungen)`);
    }
    if (entry?.note) notes.push(`${entry.label}: ${entry.note}`);
  }
  const skipped = [...modelCatalog.checkpoint, ...modelCatalog.vae].filter((e) => !e.engine).length;
  if (skipped) notes.push(`${skipped} Datei(en) nicht nutzbar – Details per Mauszeiger auf ⚠`);
  $("#model-note").textContent = notes.join(" · ");
}

async function loadModels() {
  try {
    modelCatalog = await api("/api/models");
  } catch (err) {
    $("#model-note").textContent = `Modell-Ordner nicht lesbar: ${err.message}`;
    return;
  }
  fillPicker($("#pick-checkpoint"), modelCatalog.checkpoint, modelCatalog.selected.checkpoint, "models/yue2");
  fillPicker($("#pick-vae"), modelCatalog.vae, modelCatalog.selected.vae, "models/vae");
  $("#models-refresh").title = `Modell-Ordner neu einlesen (${modelCatalog.root})`;
  fillLoraPicker();
  showModelNote();
}

async function pickModel(kind) {
  const key = kind === "checkpoint" ? "model" : "vae";
  const select = $(`#pick-${kind}`);
  try {
    const res = await api("/api/settings", { method: "PUT", body: JSON.stringify({ [key]: select.value }) });
    renderEngine(res.engine);
    refreshStatus();
  } catch (err) {
    showBanner("");
    $("#banner").hidden = false;
    $("#banner").textContent = `Modell nicht gewechselt: ${err.message}`;
  }
  loadModels();
  schedulePrediction();
}

// ---------------------------------------------------------------- init
async function init() {
  info = await api("/api/info");
  renderEngine(info.engine);
  setupEngineControls();
  refreshStatus();
  loadModels();
  $("#pick-checkpoint").addEventListener("change", () => pickModel("checkpoint"));
  $("#pick-vae").addEventListener("change", () => pickModel("vae"));
  $("#models-refresh").addEventListener("click", loadModels);
  $("#lora-add").addEventListener("click", () => {
    const path = $("#lora-pick").value;
    if (path) addLoraRow(path, 1, { fromUser: true });
  });

  for (const name of info.sections) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip";
    const tag = name.split("-").map((p) => p[0].toUpperCase() + p.slice(1)).join("-");
    b.textContent = `[${tag}]`;
    b.addEventListener("click", () => {
      const ta = form.elements.lyrics;
      const prefix = ta.value && !ta.value.endsWith("\n\n") ? (ta.value.endsWith("\n") ? "\n" : "\n\n") : "";
      ta.value += `${prefix}[${tag}]\n`;
      ta.focus();
      schedulePreview();
    });
    $("#section-buttons").append(b);
  }

  const presetSelect = $("#preset");
  info.presets.forEach((p, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = `${p.name} (${p.source})`;
    presetSelect.append(o);
  });
  presetSelect.addEventListener("change", () => {
    const p = info.presets[presetSelect.value];
    if (p) { const { name, source, ...spec } = p; writeSpec(spec); }
  });

  addVoice();
  $("#add-voice").addEventListener("click", () => {
    addVoice({ id: nextVoiceId(), gender: "male", descriptors: [] });
    schedulePreview();
  });
  $("#dice").addEventListener("click", () => {
    form.elements.seed.value = Math.floor(Math.random() * 2 ** 31);
  });
  form.querySelectorAll("[data-enable]").forEach((t) => t.addEventListener("change", () => {
    form.elements[t.dataset.enable].disabled = !t.checked;
    updateOutputs();
  }));
  form.addEventListener("input", () => { updateOutputs(); schedulePreview(); });
  form.addEventListener("submit", (ev) => { ev.preventDefault(); submit(readSpec()); });
  updateOutputs();
  schedulePreview();
  refreshJobs();
}

// Shared with training.js
window.MGU = {
  api,
  submit: (spec) => submit(spec),
  writeSpec: (spec) => writeSpec(spec),
  reloadModels: () => loadModels(),
  showTab: (name) => showTab(name),
};

function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $("#songs-view").hidden = name !== "songs";
  $("#training-view").hidden = name !== "training";
  window.dispatchEvent(new CustomEvent("mgu:tab", { detail: name }));
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

init().catch((err) => { $("#engine-label").textContent = `Server nicht erreichbar: ${err.message}`; });
