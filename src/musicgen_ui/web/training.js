// LoRA training page (artist LoRAs for YuE2's composition half).
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const api = (...args) => window.MGU.api(...args);

const PHASES = {
  idle: "bereit", start: "startet", prep: "Songs analysieren (MERT)", tokenize: "Tokens erzeugen",
  dataset: "Datensatz vorbereiten", load: "Basismodell laden", train: "Training", done: "fertig",
  stopped: "gestoppt", error: "Fehler",
};
const RUNNING = new Set(["start", "prep", "tokenize", "dataset", "load", "train"]);
const FINISHED = new Set(["done", "stopped", "error"]);
// the process may still be exiting after "done" – the phase decides what the page shows
const isRunning = (view) => RUNNING.has(view.status.phase || "idle") || (view.active && !FINISHED.has(view.status.phase));
const SETTINGS = [
  ["steps", "Schritte", "Ab ca. 1500 lernt das Modell die Songs auswendig."],
  ["rank", "LoRA-Rang", "64 wie im Original; kleiner = weniger VRAM."],
  ["lr", "Lernrate", ""],
  ["artist_fraction", "Anteil eigene Songs", "Rest kommt aus dem Regularisierungs-Paket (0.5 = halb/halb)."],
  ["checkpoint_from", "Checkpoints ab Schritt", ""],
  ["checkpoint_every", "… alle n Schritte", ""],
];

let projects = [];
let current = null;      // project view from the API
let pollTimer = null;
let visible = false;

function formatTime(seconds) {
  if (!seconds && seconds !== 0) return "";
  const m = Math.floor(seconds / 60), h = Math.floor(m / 60);
  return h ? `${h} h ${m % 60} min` : `${m} min ${Math.round(seconds % 60)} s`;
}

// ---------------------------------------------------------------- list
async function loadList() {
  const data = await api("/api/training");
  projects = data.projects;
  const box = $("#t-projects");
  box.innerHTML = projects.length ? "" : '<p class="muted">Noch keine Projekte.</p>';
  for (const p of projects) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `tproj ghost${current?.project.id === p.id ? " active" : ""}`;
    b.innerHTML = `<strong>${esc(p.name)}</strong><small>Trigger „${esc(p.trigger)}“ · ${p.songs} Songs · ${esc(PHASES[p.status] || p.status)}</small>`;
    b.addEventListener("click", () => openProject(p.id));
    box.append(b);
  }
}

// ---------------------------------------------------------------- detail
async function openProject(id) {
  current = await api(`/api/training/${id}`);
  render();
  loadList();
  schedulePoll();
}

function settingsHtml(p) {
  return SETTINGS.map(([key, label, hint]) => `
    <label>${label}<input data-setting="${key}" type="number" step="any" min="0" value="${p.settings[key]}" title="${esc(hint)}"></label>`).join("");
}

function songHtml(s, locked) {
  return `<div class="tsong" data-file="${esc(s.file)}">
    <div class="row">
      <input class="grow" data-song="title" value="${esc(s.title)}" ${locked}>
      <span class="muted">${esc(s.file)}</span>
      <button type="button" class="ghost" data-remove="${esc(s.file)}" ${locked} title="Entfernen">✕</button>
    </div>
    <input data-song="description" value="${esc(s.description)}" placeholder="Beschreibung nur für diesen Song (optional, sonst Projekt-Beschreibung)" ${locked}>
    <details ${s.lyrics.trim() ? "" : "open"}><summary>Lyrics ${s.lyrics.trim() ? "✓" : "– fehlen"}</summary>
      <textarea data-song="lyrics" rows="6" placeholder="[Verse]&#10;… vollständige Lyrics mit Abschnitten …&#10;&#10;[Chorus]&#10;…&#10;(Instrumental: [instrumental])" ${locked}>${esc(s.lyrics)}</textarea>
    </details>
  </div>`;
}

function chartSvg(status) {
  const losses = status.losses || [], evals = status.evals || [];
  if (!losses.length && !evals.length) return "";
  const steps = Math.max(status.steps || 1, ...losses.map((l) => l[0]), ...evals.map((e) => e.step));
  const values = [...losses.map((l) => l[1]), ...evals.flatMap((e) => [e.artist, e.minted_val])].filter((v) => v != null);
  const lo = Math.min(...values), hi = Math.max(...values), pad = (hi - lo) * 0.1 || 0.1;
  const x = (s) => (s / steps) * 600, y = (v) => 140 - ((v - (lo - pad)) / (hi - lo + 2 * pad)) * 130;
  const line = (pts) => pts.map(([s, v], i) => `${i ? "L" : "M"}${x(s).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const series = (key) => line(evals.filter((e) => e[key] != null).map((e) => [e.step, e[key]]));
  return `<svg class="tchart" viewBox="0 0 600 150" preserveAspectRatio="none" role="img" aria-label="Loss-Verlauf">
      <path class="train" d="${line(losses)}"/><path class="artist" d="${series("artist")}"/><path class="minted" d="${series("minted_val")}"/>
    </svg>
    <div class="tlegend"><span>— Training</span><span style="color:var(--accent)">— eigene Songs</span>
      <span style="color:var(--pred)">— Kontrolle (sollte flach bleiben)</span></div>`;
}

function trainHeadHtml() {
  const { status, active, problems } = current;
  const running = isRunning(current);
  return `${problems.length ? `<ul class="problems">${problems.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
      <div class="row">
        ${running ? '<button type="button" id="t-stop">Stoppen</button>'
          : `<button type="button" id="t-start" class="primary" ${problems.length ? "disabled" : ""}>Training starten</button>`}
        <span class="muted">${esc(status.message || "")}${status.eta_s ? ` · noch ca. ${formatTime(status.eta_s)}` : ""}</span>
      </div>`;
}

function render() {
  if (!current) return;
  const { project: p, status, active, checkpoints, problems, companion } = current;
  const phase = status.phase || "idle";
  const running = isRunning(current);
  const locked = running ? "disabled" : "";
  const progress = phase === "train" ? (100 * (status.step || 0)) / (status.steps || 1)
    : (["prep", "tokenize"].includes(phase) ? (100 * (status.song || 0)) / (status.songs || 1) : null);
  const heads = ["v9", "v8", "v5", "v4"].map((h) => `<option ${h === p.head ? "selected" : ""}>${h}</option>`).join("");
  $("#t-detail").innerHTML = `
    <div class="row"><h2 class="grow">${esc(p.name)} <span class="badge">${esc(PHASES[phase] || phase)}</span></h2>
      <button type="button" class="ghost" id="t-delete" ${locked}>Projekt löschen</button></div>
    <p class="muted">Trigger „<strong>${esc(p.trigger)}</strong>“ · trainiert den Kompositions-Teil von YuE2 auf deine Songs
      (Verfahren und Encoder von Mothersuperior, CC BY-NC – nur nicht-kommerziell).</p>

    <div class="tsec"><h3>1. Songs (${p.songs.length}) – empfohlen 10–30, vollständige Lyrics</h3>
      <label>Klang-Beschreibung für alle Songs <input id="t-description" value="${esc(p.description)}" ${locked}></label>
      <div id="t-songs">${p.songs.map((s) => songHtml(s, locked)).join("")}</div>
      <label class="check"><input type="file" id="t-upload" multiple accept=".flac,.wav,.mp3,.ogg,.aiff,.aif,audio/*" ${locked}> Audio hinzufügen (FLAC/WAV/MP3 …)</label>
      <p id="t-upload-state" class="hint"></p>
    </div>

    <div class="tsec"><h3>2. Einstellungen</h3>
      <div class="grid">${settingsHtml(p)}
        <label>Encoder / Klang-LoRA <select id="t-head" ${locked}>${heads}</select></label>
      </div>
      <label class="check"><input type="checkbox" id="t-reg" ${p.use_regularizer ? "checked" : ""} ${locked}>
        Regularisierungs-Paket nutzen (empfohlen – verhindert, dass YuE2 bei wenigen Songs „verlernt“)</label>
      <label class="check"><input type="checkbox" id="t-consent" ${p.consent ? "checked" : ""} ${locked}>
        Ich habe die Rechte an diesen Aufnahmen bzw. die Einwilligung der Künstler*innen.</label>
    </div>

    <div class="tsec"><h3>3. Training</h3>
      <div id="t-train-head">${trainHeadHtml()}</div>
      ${progress !== null ? `<div class="bar"><div style="width:${progress.toFixed(1)}%"></div></div>` : ""}
      <p class="hint">Während des Trainings ist die Song-Generierung pausiert (die GPU wird gebraucht, ca. 14–18 GB).
        Beim ersten Mal werden MERT (~1 GB), der Encoder und das Regularisierungs-Paket heruntergeladen.</p>
      ${chartSvg(status)}
      ${current.log ? `<details><summary>Protokoll</summary><pre class="tlog">${esc(current.log)}</pre></details>` : ""}
    </div>

    <div class="tsec"><h3>4. Checkpoints</h3>
      <p class="hint">Den besten Checkpoint wählst du nach Gehör: „Probe-Song“ erzeugt mit jedem Checkpoint denselben Song
        (gleicher Seed). Der Autor fand Schritt ~800 am besten.</p>
      ${companion ? `<p class="hint">Klang-LoRA ${esc(p.head)} wird mitbenutzt ✓</p>`
        : `<p class="hint">Passende Klang-LoRA (${esc(p.head)}) fehlt – sie lässt eigene Aufnahmen natürlicher klingen.
             <button type="button" class="ghost" id="t-companion">Herunterladen (~70 MB)</button></p>`}
      ${checkpoints.length ? `<button type="button" class="ghost" id="t-probe-all">Alle Checkpoints probehören</button>` : ""}
      <div>${checkpoints.map((c) => `<div class="tckpt"><span><strong>${esc(c.name)}</strong></span>
        <span class="muted">${c.size_mb} MB</span>
        <button type="button" class="ghost" data-probe="${esc(c.file)}" data-name="${esc(c.name)}">Probe-Song</button>
        <button type="button" class="ghost" data-export="${esc(c.file)}">Als LoRA übernehmen</button></div>`).join("")
        || '<p class="muted">Noch keine Checkpoints.</p>'}</div>
      <p id="t-export-state" class="hint"></p>
    </div>`;
  bind();
}

// ---------------------------------------------------------------- actions
async function save(body, { full = false } = {}) {
  try {
    current = await api(`/api/training/${current.project.id}`, { method: "PUT", body: JSON.stringify(body) });
    if (full) {
      render();
      return;
    }
    // keep focus/typing intact: only refresh hints, start button and lyrics markers
    $("#t-train-head").innerHTML = trainHeadHtml();
    bindTrainHead();
    for (const s of current.project.songs) {
      const summary = document.querySelector(`.tsong[data-file="${CSS.escape(s.file)}"] summary`);
      if (summary) summary.textContent = `Lyrics ${s.lyrics.trim() ? "✓" : "– fehlen"}`;
    }
  } catch (err) {
    alert(err.message);
  }
}

function bind() {
  const id = current.project.id;
  $("#t-description")?.addEventListener("change", (e) => save({ description: e.target.value }));
  document.querySelectorAll("[data-setting]").forEach((input) => input.addEventListener("change", () =>
    save({ settings: { [input.dataset.setting]: Number(input.value) } })));
  $("#t-head")?.addEventListener("change", (e) => save({ head: e.target.value }, { full: true }));
  $("#t-reg")?.addEventListener("change", (e) => save({ use_regularizer: e.target.checked }));
  $("#t-consent")?.addEventListener("change", (e) => save({ consent: e.target.checked }));
  document.querySelectorAll("#t-songs [data-song]").forEach((input) => input.addEventListener("change", () => {
    const file = input.closest(".tsong").dataset.file;
    save({ songs: [{ file, [input.dataset.song]: input.value }] });
  }));
  document.querySelectorAll("[data-remove]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("Song aus dem Projekt entfernen?")) return;
    current = await api(`/api/training/${id}/songs/${encodeURIComponent(b.dataset.remove)}`, { method: "DELETE" });
    render();
  }));
  $("#t-upload")?.addEventListener("change", (e) => upload([...e.target.files]));
  bindTrainHead();
  $("#t-delete")?.addEventListener("click", async () => {
    if (!confirm(`Projekt „${current.project.name}“ mit allen Songs und Checkpoints löschen?`)) return;
    await api(`/api/training/${id}`, { method: "DELETE" });
    current = null;
    $("#t-detail").innerHTML = '<p class="muted">Projekt gelöscht.</p>';
    loadList();
  });
  $("#t-companion")?.addEventListener("click", async (e) => {
    e.target.disabled = true;
    e.target.textContent = "lädt …";
    try {
      await api("/api/training/companion", { method: "POST", body: JSON.stringify({ version: current.project.head }) });
      window.MGU.reloadModels();
      await openProject(id);
    } catch (err) {
      alert(err.message);
      e.target.disabled = false;
    }
  });
  document.querySelectorAll("[data-export]").forEach((b) => b.addEventListener("click", async () => {
    const res = await api(`/api/training/${id}/export`, { method: "POST", body: JSON.stringify({ file: b.dataset.export }) });
    $("#t-export-state").textContent = `Übernommen: ${res.file} – erscheint jetzt im LoRA-Picker der Songs.`;
    window.MGU.reloadModels();
  }));
  document.querySelectorAll("[data-probe]").forEach((b) => b.addEventListener("click", () =>
    probe([{ file: b.dataset.probe, name: b.dataset.name }])));
  $("#t-probe-all")?.addEventListener("click", () =>
    probe(current.checkpoints.filter((c) => c.step !== null).map((c) => ({ file: c.file, name: c.name }))));
}

function bindTrainHead() {
  const id = current.project.id;
  $("#t-start")?.addEventListener("click", async () => {
    try {
      current = await api(`/api/training/${id}/start`, { method: "POST", body: "{}" });
      render();
      schedulePoll();
    } catch (err) {
      alert(err.message);
    }
  });
  $("#t-stop")?.addEventListener("click", async () => {
    current = await api(`/api/training/${id}/stop`, { method: "POST" });
    render();
  });
}

async function upload(files) {
  const state = $("#t-upload-state");
  for (const [i, file] of files.entries()) {
    state.textContent = `Lade hoch ${i + 1}/${files.length}: ${file.name} …`;
    const res = await fetch(`/api/training/${current.project.id}/songs?filename=${encodeURIComponent(file.name)}`,
      { method: "POST", body: file });
    if (!res.ok) {
      state.textContent = `Fehler bei ${file.name}: ${(await res.json()).detail}`;
      return;
    }
    current = await res.json();
  }
  render();
  loadList();
  $("#t-upload-state").textContent = `${files.length} Datei(en) hinzugefügt – jetzt die Lyrics eintragen.`;
}

function probe(checkpoints) {
  const p = current.project;
  const song = p.songs.find((s) => s.lyrics.trim() && s.lyrics.trim() !== "[instrumental]") || p.songs[0];
  for (const c of checkpoints) {
    const loras = [{ path: c.file, strength: 1 }];
    if (current.companion) loras.push({ path: current.companion, strength: 1 });
    // Same caption format as in training: "<trigger>, in the style of <trigger>. <description>"
    window.MGU.submit({
      title: `${p.name} – Probe ${c.name}`, language: null,
      style_override: `${p.trigger}, in the style of ${p.trigger}. ${p.description}`.trim(),
      lyrics: song?.lyrics || "[Verse]\nla la la", plan_mode: "off", seed: 12, candidates: 1, loras,
    });
  }
  window.MGU.showTab("songs");
}

// ---------------------------------------------------------------- polling
function schedulePoll() {
  clearTimeout(pollTimer);
  if (!visible || !current) return;
  const running = isRunning(current);
  pollTimer = setTimeout(async () => {
    try {
      const id = current.project.id;
      const fresh = await api(`/api/training/${id}`);
      const focused = document.activeElement?.closest("#t-detail");
      if (!focused || fresh.status.phase !== current.status.phase) {
        current = fresh;
        render();
      }
      if (!fresh.active && running) loadList();
    } catch { /* server restarting */ }
    schedulePoll();
  }, running ? 2000 : 10000);
}

$("#t-new").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target.elements;
  try {
    const view = await api("/api/training", { method: "POST", body: JSON.stringify({
      name: f.name.value, trigger: f.trigger.value, description: f.description.value }) });
    e.target.reset();
    await openProject(view.project.id);
  } catch (err) {
    alert(err.message);
  }
});

window.addEventListener("mgu:tab", (e) => {
  visible = e.detail === "training";
  if (visible) {
    loadList();
    schedulePoll();
  }
});
