# Roadmap

## Leitlinien (vom Projektinhaber, Nov. 2026)

- **Ziel:** von Suno auf lokale Generierung umsteigen. Ein Suno-Klon ist ausdrücklich nicht das Ziel.
  Weitere Referenz für Bedienkonzepte ist **Magix Music Maker** (Spuren, Clips, Timeline).
- **Stimmen (Female/Male, mehrere Stimmen):** Die Hinweise im Stil-Text reichen als Anfang. Ziel ist eine *Garantie* dafür, über LoRAs und/oder eigene Nachbearbeitung.
- **Eigene Stimme:** wird gebaut, auch wenn es nur über Umwege geht (Separation → Voice-Conversion).
- **Erweitern:** so umsetzen, wie es mit YuE2 machbar ist.
- **SFX, Intros, Überbrückungen:** über eine integrierte **Mehrspur-Timeline**.
- **LoRA:** Wir schreiben einen eigenen LoRA-Loader. Training kommt später dazu.

### Stimm-Garantie: Stufenmodell
Nichts davon allein garantiert das Geschlecht der Stimme. Kombiniert wird es aber verlässlich:

| Stufe | Mittel | Verlässlichkeit | Kosten |
|---|---|---|---|
| 1 | Stil-Hinweis („airy female vocal“) | niedrig–mittel | keine |
| 2 | Stimm-/Timbre-LoRA | mittel–hoch (abhängig vom LoRA) | LoRA finden oder trainieren |
| 3 | **Automatische Prüfung:** Gesang abtrennen, Geschlecht/Stimmlage per Klassifikator erkennen, bei Abweichung automatisch neu würfeln (Best-of-N) | hoch (prüft das Ergebnis) | zusätzliche Takes |
| 4 | Voice-Conversion auf ein Stimmprofil | sehr hoch (Klangfarbe wird ersetzt) | Stimmprofil nötig |

Stufe 3 ist die günstigste echte Garantie. Stufe 4 ist ohnehin für „eigene Stimme“ nötig und löst auch „mehrere Stimmen pro Abschnitt“.

---

## Phase 0 – Fundament ✅
- [x] Python-Paket, FastAPI-Server, Web-UI ohne Build-Schritt
- [x] Strukturierte Song-Eingabe → YuE2-Anfrage, Live-Vorschau mit Warnungen
- [x] Stimmen (Geschlecht, Klang), `@voice`-Zuordnung pro Abschnitt (gespeichert)
- [x] Seed, Stil-Treue (CFG), Kreativität (temperature), top_p, mehrere Takes
- [x] Warteschlange mit Fortschritt und Abbrechen, Bibliothek auf der Festplatte
- [x] Partitur anzeigen (abcjs) und als Vorlage übernehmen („Melodie behalten, Klang neu würfeln“)
- [x] Plugin-System: Python + Lua (Vorlagen, Makros, Hooks)
- [x] Mock-Engine + Tests

## Phase 1 – Echte Generierung stabil
- [x] Conda-Installation (`install.sh`, `start.sh`, `environment.yml`), Modell-Download vorab
- [x] YuE2 als Standard-Engine; Mock nur als Fallback/Testmodus
- [x] Einstellungen in der UI (Engine, VRAM-Modus, Backend, Modell/VAE), gespeichert in `data/settings.json`
- [x] Modell laden/entladen, Status mit Download-Fortschritt, VRAM-Anzeige
- [x] Fortschritt in echten Einheiten (Tokens ≈ Sekunden Musik), Hinweis bei VRAM voll
- [x] Erster Lauf mit YuE2 auf der RTX 4090 (Song funktioniert; danach System-OOM)
- [x] YuE2-Versionserkennung (Release vs. main)
- [x] Speicher-Schutz: Offload (optional), Modell-Platz zwischen Songs, VRAM-Reserve, RAM-Notbremse, Aufräumen nach OOM
- [x] „Abstürze“ geklärt: kein RAM/VRAM-, Strom- oder PCIe-Problem; die Klangsynthese blockierte die GPU, die auch den Desktop rendert → „Desktop flüssig halten“ (bestätigt auf RTX 4090: 81 s Musik in 144 s, Desktop bedienbar)
- [ ] Laufzeit und Speicher-Spitzen pro Phase auf der 4090 messen, Standardwerte nachjustieren
- [ ] Export: FLAC/WAV/MP3, Lautheit normalisieren

## Phase 1b – Modelle & GGUF
- [x] `models/`-Ordner (yue2, vae, lora) mit Picker in der UI, Erkennung von HF-Ordnern und GGUF (Quantisierung, Architektur)
- [x] Unpassende Dateien erkennen (YuE v1, falscher Ordner, defekt)
- [x] GGUF-Loader in der offiziellen Pipeline (HOT-Step-Layout), Modi BF16 / Ausgewogen / Maximal, offline-fähig (E10)
- [ ] GGUF auf der RTX 4090 messen: Tempo und VRAM pro Modus und Quantisierung
- [x] Checkpoints im yue2.cpp-Layout (Serveurperso/YuE2-GGUF: Originalnamen, `yue2.config_json`)
- [ ] VAE im yue2.cpp-Layout (DAC-Namen `decoder.block.N.conv_t1` …)
- [ ] Optional: zweite Engine mit schnellen GGML-Kernels – Kandidaten HOT-Step-CPP oder [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp) (CUDA/Vulkan, eigener Server)

## Phase 2 – LoRA laden ✅
- [x] `models/lora/`, Picker im Song-Formular, mehrere LoRAs mit eigener Stärke pro Song
- [x] Loader für Mothersuperior-, ComfyUI- und Starnodes-Formate, ohne das Basismodell neu zu laden, exaktes Zurücksetzen
- [x] Verwendete LoRAs werden mit dem Song gespeichert

## Phase 3 – Partitur-Workflow (YuE2s Stärke)
- [ ] „Nur planen“: Partitur erzeugen, **ohne** Audio zu rendern (schnell)
- [ ] Partitur im Browser abspielen (abcjs-Synth) → vor dem teuren Rendern hören
- [ ] ABC-Editor mit Validierung (Stimmen `Vocal`/`Ins`, Takte, Silben)
- [ ] Cover: Song hochladen → SheetSage2-Transkription (eigene Umgebung) → neuer Stil
- [ ] Takes vergleichen (A/B), Favoriten, Bewertung

## Phase 4 – Mehrspur-Timeline
Orientierung an Music Maker: Spuren untereinander, Clips per Drag & Drop.
- [ ] Projekt = mehrere Spuren; Clips aus Bibliothek (Songs, Takes), Uploads (SFX, Samples) und YuE2-Instrumental-Clips
- [ ] Clips verschieben, trimmen, Fade-in/-out, Crossfade, Lautstärke/Pan pro Clip und Spur, Mute/Solo
- [ ] Raster in Takten (Tempo aus der Partitur), Abschnittsmarken aus den Lyrics
- [ ] Stems: Songs per Separation in Gesang/Instrumental aufteilen → eigene Spuren
- [ ] Mixdown auf dem Server (numpy/ffmpeg), Lautheit normalisieren, Export
- [ ] „Erweitern“: Partitur verlängern und neu rendern, Ergebnis landet als neuer Clip; Machbarkeitstest Token-Fortsetzung

## Phase 5 – Stimmen
- [ ] Stimm-Prüfung (Stufe 3): Gesang abtrennen, Geschlecht/Stimmlage erkennen, Best-of-N automatisch
- [ ] Stimmprofile aus mehreren hochgeladenen Titeln (Separation → säubern → Datensatz), Einverständnis-Bestätigung
- [ ] Singing-Voice-Conversion (zero-shot und/oder trainiert) als Nachbearbeitung (Stufe 4)
- [ ] Stimmen-Zuordnung pro Abschnitt anwenden (`[Chorus @duo]`) über Timeline-Abschnitte

## Phase 6 – LoRA-Training (erste Version ✅)
- [x] Artist-LoRA aus eigenen Songs: MERT → Mothersuperior-Encoder → Tokens, Regularisierung, AR-LoRA (eigener Prozess)
- [x] UI: Projekte, Upload, Lyrics, Einwilligung, Fortschritt, Loss-Kurve, Checkpoints, Probe-Songs, Übernehmen
- [ ] Erster echter Lauf auf der RTX 4090 (Dauer, VRAM, Qualität nach Gehör)
- [ ] Lyric-Cursor-Hilfsaufgabe (Demucs + MMS-Alignment), optional
- [ ] Klang-LoRA (NAR) auf eigene Aufnahmen nachtrainieren (`joint.py`-Schritt)
- [ ] Ziel u.a.: Stimm-/Timbre-LoRAs für die Stimm-Garantie (Stufe 2)

## Später
- [ ] Lua-API für Batch-Workflows (`mgu.submit`)
- [ ] Fortschritt per Server-Sent Events statt Polling
- [ ] Remote-/Multi-GPU-Worker (z.B. YuE2-Turbo-Server)

## Ideen-Sammelbecken (TBA)
- **Modell-Browser:** Hugging Face nach YuE2-Modellen, GGUFs und LoRAs durchsuchen (es gibt viele: https://huggingface.co/models?search=yue2), Layout/Lizenz anzeigen, direkt nach `models/` laden (ähnlich Civitai-Browsern in Bild-UIs)
- Lyrics-Helfer: Silben pro Zeile, Reime, Abschnittsstruktur prüfen
- „Rezept“ exportieren/importieren (Anfrage + Konfiguration → exakt reproduzierbar)
- Benachrichtigung, wenn ein Song fertig ist
- Stil-Bibliothek mit Hörbeispielen
