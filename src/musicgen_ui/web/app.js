// Music-Gen-UI front end. Plain ES module, no build step.
const $ = (sel, root = document) => root.querySelector(sel);
const form = $("#song-form");
const STATUS = { queued: "wartet", running: "läuft", done: "fertig", failed: "Fehler", cancelled: "abgebrochen" };
const STAGES = { load: "Modell laden", score: "Partitur", audio: "Audio-Tokens", nar: "Klangsynthese", vae: "Dekodieren" };

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
  const stage = STAGES[p.stage] || p.stage || "";
  card.innerHTML = `
    <header><strong></strong><span class="badge">${STATUS[job.status] || job.status}</span></header>
    <p class="muted meta"></p>
    ${job.status === "running" ? `<div class="bar"><div style="width:${pct ?? 100}%" class="${pct === null ? "indeterminate" : ""}"></div></div>
      <p class="muted">${stage}${pct !== null ? ` · ${pct}%` : ""}</p>` : ""}
    ${job.status === "done" ? `<audio controls preload="none" src="/api/jobs/${job.id}/audio"></audio>` : ""}
    ${job.error ? `<p class="error"></p>` : ""}
    <div class="actions"></div>`;
  $("strong", card).textContent = job.title;
  const r = job.result || {};
  $(".meta", card).textContent = [`Seed ${job.request.seed}`, r.duration_s ? `${r.duration_s}s` : null,
    r.truncated ? "⚠ abgeschnitten" : null].filter(Boolean).join(" · ");
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

// ---------------------------------------------------------------- init
async function init() {
  info = await api("/api/info");
  const e = info.engine;
  $("#engine-info").textContent = e.engine === "mock" ? "Mock-Engine (Platzhalter-Audio)"
    : `${e.model} · ${e.device}${e.low_vram ? " · low VRAM" : ""}${e.loaded ? "" : " · lädt beim ersten Job"}`;

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

init().catch((err) => { $("#engine-info").textContent = `Server nicht erreichbar: ${err.message}`; });
