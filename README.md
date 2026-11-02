# Music-Gen-UI

Web-Oberfläche, um mit [YuE2](https://github.com/multimodal-art-projection/YuE) (offenes Lyrics-to-Song-Modell von m-a-p, Sept. 2026) Musik zu generieren.

> **Status:** Phase 0, das Fundament steht. Mit der Mock-Engine ist alles lauffähig und getestet.
> Der YuE2-Adapter folgt der offiziellen API (`yue2-infer 0.1.6`), wurde aber **noch nicht auf einer GPU ausgeführt**.

- **Was YuE2 kann und was nicht** (Female/Male, eigene Stimme, Erweitern, LoRA …): [docs/FEATURES.md](docs/FEATURES.md)
- **Warum Python, Lua-Plugins, FastAPI …:** [docs/DECISIONS.md](docs/DECISIONS.md)
- **Was als Nächstes kommt:** [docs/ROADMAP.md](docs/ROADMAP.md)

## Schnellstart (ohne GPU, Mock-Engine)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
music-gen-ui            # → http://127.0.0.1:7860
```

Die Mock-Engine erzeugt eine Platzhalter-Melodie. Sie ist für die Arbeit an UI und Plugins gedacht.

## Mit YuE2

Voraussetzung: NVIDIA-GPU mit BF16 und ca. 24 GB VRAM. Mit `--low-vram` ist weniger möglich (experimentell, siehe [YuE2 hardware.md](https://github.com/multimodal-art-projection/YuE/blob/main/docs/hardware.md)).

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[yue2,lua]"
yue2 doctor
music-gen-ui --engine yue2
```

Die Modellgewichte (`m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae`) werden beim ersten Job von Hugging Face geladen.

### Konfiguration (Umgebungsvariablen)

| Variable | Standard | Bedeutung |
|---|---|---|
| `MGU_ENGINE` | `mock` | `mock` oder `yue2` |
| `MGU_DATA_DIR` | `data` | Bibliothek mit generierten Songs |
| `MGU_HOST` / `MGU_PORT` | `127.0.0.1` / `7860` | Server-Adresse |
| `MGU_MODEL` / `MGU_VAE` | `m-a-p/YuE2-3B` / `m-a-p/YuE2-Vae` | Checkpoints (auch lokale Pfade) |
| `MGU_DEVICE` | `auto` | `cuda`, `mps`, `xpu`, `cpu`, `auto` |
| `MGU_LOW_VRAM` | `0` | YuE2 `low_vram` (langsamer, weniger VRAM) |
| `MGU_QUANTIZATION` | `none` | `none` oder `fp8` |
| `MGU_BACKEND` | `torch` | `torch`, `torch-eager`, `vllm` |
| `MGU_PLUGIN_DIRS` | `plugins` | Plugin-Ordner (mit `:` getrennt) |

## Lyrics-Syntax

```text
[Verse @lead]
Neon on the water
Engines in the rain

[Chorus @lead,duo]
We drive into the light
```

- Abschnitte: `Intro, Verse, Pre-Chorus, Chorus, Bridge, Interlude, Outro, Instrumental`. Nummern wie `[Verse 2]` sind erlaubt.
- `@id` ordnet Abschnitte Stimmen zu. YuE2 selbst kann das nicht steuern. Die Angabe wird vor dem Senden entfernt und für die spätere Voice-Conversion gespeichert.
- Die Vorschau „Was YuE2 bekommt“ zeigt immer den exakten Text, der ans Modell geht.

## Plugins

Plugins liegen in `plugins/`, als `*.py` oder `*.lua`. Dateien mit führendem `_` werden ignoriert. Beispiele:

- [`plugins/example_presets.lua`](plugins/example_presets.lua): Vorlagen und Stil-Makros (`{lofi}`, `{epic}`)
- [`plugins/_example_plugin.py`](plugins/_example_plugin.py): Python-Variante

Verfügbare Hooks: `transform_spec`, `transform_request`, `on_complete` und Vorlagen (`presets`). Details stehen in `src/musicgen_ui/plugins/manager.py`.

## Projektstruktur

```text
src/musicgen_ui/
  models.py       Song-Struktur (Pydantic)
  prompting.py    Song-Struktur → YuE2-Anfrage (Stil-Text, Lyrics, Warnungen)
  engine/         Engine-Schnittstelle, yue2-Adapter, mock
  jobs.py         Warteschlange (1 Worker) + Bibliothek auf der Festplatte
  plugins/        Plugin-Manager, Lua-Laufzeit
  server.py       FastAPI-Routen (/api/...)
  web/            Oberfläche (HTML/CSS/JS, kein Build)
plugins/          Beispiel-Plugins
tests/            pytest
```

```bash
pytest
```

## Lizenz

Code dieses Projekts: noch nicht festgelegt.
Für die YuE2-**Gewichte** gilt eine eigene Lizenz: CC BY-NC 4.0 mit Zusatzerlaubnis für Creator ([MODEL_LICENSE](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE)).
