# Music-Gen-UI

Web-Oberfläche, um mit [YuE2](https://github.com/multimodal-art-projection/YuE) Musik zu generieren. YuE2 ist ein offenes Lyrics-to-Song-Modell von m-a-p (Sept. 2026).
Zielsystem: **Linux, NVIDIA-GPU (z.B. RTX 4090), Miniconda**.

- **Was YuE2 kann und was nicht** (Female/Male, eigene Stimme, Erweitern, LoRA …): [docs/FEATURES.md](docs/FEATURES.md)
- **Warum Python, Lua-Plugins, FastAPI …:** [docs/DECISIONS.md](docs/DECISIONS.md)
- **Was als Nächstes kommt:** [docs/ROADMAP.md](docs/ROADMAP.md)

## Installation mit Miniconda (empfohlen)

Jede WebUI bekommt ihre eigene Conda-Umgebung, hier `music-gen-ui`. Sie stört also keine anderen Installationen.

**Voraussetzungen**
- Miniconda/Anaconda/Miniforge. Das Skript findet `conda` auch, wenn es nicht im `PATH` liegt, z.B. unter `~/miniconda3`.
- Ein NVIDIA-Treiber, der CUDA 12.8 unterstützt. `nvidia-smi` muss dafür oben rechts „CUDA Version: 12.8“ oder höher zeigen (Treiber ≥ 570).
  Bei einem älteren Treiber: `./install.sh --cuda cu126`.
- `git`
- ca. 15–20 GB freier Platz: Umgebung inkl. PyTorch ca. 6 GB, Modelle ca. 8–10 GB im Hugging-Face-Cache

```bash
git clone https://github.com/Lolzen/Music-Gen-Ui.git
```
```bash
cd Music-Gen-Ui
```
```bash
./install.sh
```

`install.sh` führt diese Schritte aus:
1. Die Conda-Umgebung `music-gen-ui` anlegen (Python 3.12, git, ffmpeg) bzw. bei erneutem Aufruf aktualisieren
2. PyTorch 2.10.0 mit CUDA 12.8 installieren
3. Music-Gen-UI + YuE2 + Lua-Plugin-Unterstützung installieren
4. `yue2 doctor` ausführen, den Umgebungscheck von YuE2
5. Die Modelle `m-a-p/YuE2-3B` und `m-a-p/YuE2-Vae` vorab herunterladen

Optionen:

| Option | Wirkung |
|---|---|
| `--env NAME` | anderer Umgebungsname |
| `--fast` | zusätzlich das vLLM-Backend (schneller, nur Linux) |
| `--no-models` | Modelle nicht vorab laden; das passiert sonst beim ersten Song |
| `--cuda cu126` | andere PyTorch-CUDA-Variante |
| `--mock` | nur die Oberfläche, ohne PyTorch/YuE2 (zum Ausprobieren) |
| `--yue2-ref REF` | andere YuE2-Version statt Release `yue2-v0.1.6`, z.B. `main` (siehe unten) |

**Aktualisieren:** `git pull` und danach erneut `./install.sh`.
**Modelle woanders speichern:** vor Installation und Start `export HF_HOME=/pfad/zum/cache` setzen, wie bei anderen WebUIs auch.

## Starten

```bash
./start.sh
```

Danach http://127.0.0.1:7860 öffnen. `start.sh` aktiviert die Conda-Umgebung selbst. Alle Argumente gehen an `music-gen-ui` weiter:

| Argument | Wirkung |
|---|---|
| `--host 0.0.0.0` | im Heimnetz erreichbar machen (ohne Login, nur in vertrauenswürdigen Netzen!) |
| `--port 7861` | anderer Port, falls 7860 schon von einer anderen WebUI belegt ist |
| `--preload` | Modell direkt beim Start laden statt beim ersten Song |
| `--fp8` / `--offload` | FP8 bzw. Offload für diesen Start (siehe [Speicher & Offload](#speicher--offload)) |
| `--engine mock` | Testmodus ohne Modell |

Andere Umgebung verwenden: `MGU_CONDA_ENV=meine-env ./start.sh`

### Manuell, ohne die Skripte

```bash
conda env create -f environment.yml
```
```bash
conda activate music-gen-ui
```
```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```
```bash
pip install -e ".[yue2,lua]"
```
```bash
music-gen-ui-download
```
```bash
music-gen-ui
```

## Bedienung

**Oben rechts** steht der Modellstatus: *nicht geladen* → *lädt* (mit Download-Fortschritt beim ersten Mal) → *bereit* → *generiert*.
Daneben zeigen zwei Leisten, wie viel Grafikspeicher (VRAM) und Arbeitsspeicher (RAM) belegt sind; die RAM-Leiste wird ab 90 % rot. **Modell laden** lädt das Modell vorab, **Entladen** gibt den Grafikspeicher frei, z.B. für eine andere WebUI.

**⚙ Einstellungen** (werden in `data/settings.json` gespeichert):

| Einstellung | Empfehlung RTX 4090 |
|---|---|
| Engine | *Automatisch*: nimmt YuE2, falls installiert |
| VRAM-Modus | *Standard*. *FP8* spart VRAM (experimentell) |
| Backend | *torch*. *vllm* ist schneller (Installation mit `--fast`), aber nicht mit Offload kombinierbar |
| Speicher & Offload | siehe nächster Abschnitt |
| Modell / VAE | Standard. `YuE2-Vae-legacy` ist nur für Vergleiche mit den offiziellen Benchmarks gedacht |

Während der Generierung zeigt jeder Job seine Phase:

| Phase | Was passiert |
|---|---|
| Partitur planen | YuE2 komponiert Melodie und Akkorde als ABC-Noten |
| Audio generieren | Fortschritt in Tokens; 25 Tokens ≈ 1 Sekunde Musik |
| Klangsynthese → Dekodieren | Das fertige Audio entsteht (48 kHz Stereo, FLAC) |

Ein Song ist auf ca. 6 Minuten begrenzt. „⚠ abgeschnitten“ heißt, dass dieses Limit erreicht wurde.

## Speicher & Offload

Ein Song mit YuE2 braucht zwei Arten Speicher:
- **VRAM:** rund 20 GB in der Spitze
- **RAM:** YuE2 parkt das Modell (ca. 7,5 GB) zum Dekodieren und zwischen Songs im Arbeitsspeicher

Wird eins von beiden knapp, kann der ganze Rechner einfrieren. Das gilt besonders, wenn Linux anfängt zu swappen oder die Grafikkarte auch den Bildschirm betreibt.
Unter **⚙ Einstellungen → Speicher & Offload**:

| Einstellung | Wirkung | Kosten |
|---|---|---|
| **Offload aktivieren** | Während der Klangsynthese liegt der nicht benötigte Modellteil im RAM. Das Dekodieren läuft in kleineren Stücken. | langsamer, mehr RAM |
| **Modell zwischen Songs** | *RAM* (Standard): VRAM ist zwischen Songs frei. *VRAM*: schnellster Start, VRAM bleibt belegt. *entladen*: spart ca. 7,5 GB RAM, jeder Song lädt neu. | – |
| **VRAM-Reserve** (Standard 3 GB) | So viel VRAM bleibt für Desktop/Browser frei. YuE2 bricht lieber mit einer Fehlermeldung ab, als den Bildschirm auszuhungern. | YuE2 hat weniger VRAM |
| **RAM-Notbremse** (Standard 2 GB) | Prüft vor jedem Song, ob genug RAM frei ist. Fällt der freie RAM während des Songs unter die Grenze, wird abgebrochen und das Modell entladen. | – |

Läuft der VRAM trotzdem voll, entlädt die App das Modell automatisch. Der nächste Song startet dann sauber.

**Empfehlung, wenn das System schon einmal eingefroren ist:**
1. *RAM-Notbremse* an lassen (2 GB)
2. *VRAM-Reserve* 3–4 GB
3. Bei wenig RAM (≤ 16 GB): *Modell zwischen Songs → entladen*
4. Reicht das nicht: *Offload* einschalten, danach *FP8*

**Ursache herausfinden** (funktioniert mit runit, OpenRC, systemd …):

1. **Speicher-Protokoll der App.** Während jedes Songs schreibt die App alle 2 s RAM, Swap, VRAM und die aktuelle Phase nach `data/logs/memory-JJJJ-MM-TT.log`.
   Jede Zeile wird sofort auf die Platte geschrieben und übersteht deshalb auch einen Zwangsneustart.
   In der UI: ⚙ Einstellungen → „Speicher-Protokoll“. Im Terminal:

   ```bash
   tail -n 40 "$(ls -t data/logs/memory-*.log | head -n 1)"
   ```

   - „RAM frei“ sinkt gegen 0 und Swap steigt: Der Arbeitsspeicher war voll. *Modell zwischen Songs → entladen* wählen, RAM-Notbremse an lassen.
   - „VRAM“ steht bei voll: Der Grafikspeicher war voll. VRAM-Reserve erhöhen, *Offload* oder *FP8* wählen.
   - Die letzte Zeile zeigt, in welcher Phase es passiert ist: `score`, `audio`, `nar`, `vae` oder `danach`.

2. **Kernel-Meldungen**, falls ein Log-Dienst läuft. Unter runit ist das z.B. `socklog` (Void) oder `syslog-ng`/`metalog` (Artix):

   ```bash
   sudo grep -r -h -a -E "Out of memory: Killed process|invoked oom-killer|NVRM: Xid|NVRM: GPU at" /var/log/ 2>/dev/null | tail -n 30
   ```

   - `Out of memory: Killed process` / `invoked oom-killer`: Arbeitsspeicher voll
   - `NVRM: Xid …`: Fehler von Grafiktreiber/GPU. Die Zahl nach der PCI-Adresse ist der Fehlercode, z.B. 31 = Speicherzugriffsfehler, 79 = GPU nicht mehr erreichbar
   - Die normale Boot-Zeile `NVRM: loading NVIDIA …` ist **kein** Fehler

   Ob überhaupt Meldungen aus früheren Sitzungen gespeichert werden, zeigt dieser Befehl. Er zählt die Boot-Zeilen des Treibers; ab 2 werden ältere Sitzungen aufbewahrt:

   ```bash
   sudo grep -r -h -a -c "NVRM: loading NVIDIA" /var/log/ 2>/dev/null | paste -sd+ | bc
   ```

   Findet der Befehl nichts, werden Kernel-Meldungen nicht dauerhaft gespeichert und sind nach dem Neustart weg.
   Unter Void Linux lässt sich das so einschalten (andere Distributionen: einen Syslog-Dienst installieren):

   ```bash
   sudo xbps-install -S socklog-void
   ```
   ```bash
   sudo ln -s /etc/sv/socklog-unix /etc/sv/nanoklogd /var/service/
   ```

## Lyrics-Syntax

```text
[Verse @lead]
Neon on the water
Engines in the rain

[Chorus @lead,duo]
We drive into the light
```

- Abschnitte: `Intro, Verse, Pre-Chorus, Chorus, Bridge, Interlude, Outro, Instrumental`. Nummern wie `[Verse 2]` sind erlaubt.
- `@id` ordnet einen Abschnitt einer Stimme zu. YuE2 selbst kann das nicht steuern. Die Zuordnung wird vor dem Senden entfernt und für die spätere Voice-Conversion gespeichert.
- „Was YuE2 bekommt“ zeigt immer den exakten Text, der ans Modell geht.

## Plugins

Plugins liegen in `plugins/` als `*.py` oder `*.lua`. Dateien mit führendem `_` werden ignoriert.

- [`plugins/example_presets.lua`](plugins/example_presets.lua): Vorlagen und Stil-Makros (`{lofi}`, `{epic}`)
- [`plugins/_example_plugin.py`](plugins/_example_plugin.py): Python-Variante

Hooks: `transform_spec`, `transform_request`, `on_complete` und Vorlagen (`presets`). Details stehen in `src/musicgen_ui/plugins/manager.py`.

## Konfiguration per Umgebungsvariablen

Umgebungsvariablen überschreiben die gespeicherten Einstellungen. Das ist praktisch für Skripte.

| Variable | Standard | Bedeutung |
|---|---|---|
| `MGU_ENGINE` | `auto` | `auto`, `yue2`, `mock` |
| `MGU_DATA_DIR` | `data` | Songs und `settings.json` |
| `MGU_HOST` / `MGU_PORT` | `127.0.0.1` / `7860` | Server-Adresse |
| `MGU_MODEL` / `MGU_VAE` | `m-a-p/YuE2-3B` / `m-a-p/YuE2-Vae` | Checkpoints (auch lokale Ordner) |
| `MGU_DEVICE` | `auto` | `cuda`, `cpu`, … |
| `MGU_OFFLOAD` | `0` | Offload ein/aus (alter Name `MGU_LOW_VRAM` geht weiter) |
| `MGU_IDLE` | `ram` | Modell zwischen Songs: `ram`, `gpu`, `unload` |
| `MGU_VRAM_RESERVE_GIB` | `3` | VRAM, der für Desktop/andere Programme frei bleibt |
| `MGU_RAM_GUARD_GIB` | `2` | RAM-Notbremse (0 = aus) |
| `MGU_QUANTIZATION` | `none` | `none` oder `fp8` (braucht Compute Capability ≥ 8.9; RTX 40xx erfüllt das) |
| `MGU_BACKEND` | `torch` | `torch`, `torch-eager`, `vllm` |
| `MGU_PLUGIN_DIRS` | `plugins` | Plugin-Ordner (mit `:` getrennt) |

## Fehlerbehebung

| Problem | Lösung |
|---|---|
| Banner „YuE2 ist nicht installiert“ | `./install.sh` (ohne `--mock`) ausführen |
| „Grafikspeicher voll“ / OutOfMemory | Das Modell wird automatisch entladen. Dann *Offload* einschalten, ggf. *FP8*, andere GPU-Programme schließen |
| System friert ein / muss neu gestartet werden | Siehe [Speicher & Offload](#speicher--offload) |
| `yue2 doctor` meldet Fehler | Treiber/CUDA-Version prüfen (`nvidia-smi`), notfalls `./install.sh --cuda cu126` |
| Port belegt | `./start.sh --port 7861` |
| `unexpected keyword argument 'low_vram'` o.ä. | Behoben: Die App erkennt jetzt die installierte YuE2-Version (siehe unten) |
| Erster Song dauert lange | Beim ersten Mal werden mehrere GB heruntergeladen und geprüft. Der Fortschritt steht oben rechts |

### YuE2-Versionen

Standard ist das offizielle Release **`yue2-v0.1.6`**. Neuere Commits auf YuE2-`main` tragen dieselbe Versionsnummer, haben aber eine andere API. Die App erkennt das automatisch:

| Funktion | Release `yue2-v0.1.6` | YuE2 `main` |
|---|---|---|
| *Offload* | wird als „AR-Auslagerung“ (`offload_ar`) umgesetzt; spart weniger | echtes `low_vram` (nur der aktive Modellteil liegt im VRAM) |
| Fortschritt Klangsynthese/Dekodieren | nicht verfügbar; die Anzeige bleibt bis zum Ende bei „Audio generieren“ | mit Prozentanzeige |

Die neuere Version installierst du mit `./install.sh --yue2-ref main`. Das ist unveröffentlichter Code, also nur bei Bedarf nutzen.
Zurück zum Release geht es mit einem normalen `./install.sh`. Welche Version läuft, zeigt der Tooltip des Modellstatus oben links.

## Entwicklung

```bash
./install.sh --mock
```
```bash
conda activate music-gen-ui && pip install -e ".[dev]" && pytest
```

Die Tests prüfen den YuE2-Adapter gegen ein nachgebautes `yue2`-Paket. Dafür ist keine GPU nötig.

```text
src/musicgen_ui/
  config.py       Einstellungen (Datei/Umgebung/CLI)
  runtime.py      Engine-Verwaltung: laden, entladen, umschalten
  engine/         Engine-Schnittstelle, yue2-Adapter, mock
  models.py       Song-Struktur (Pydantic)
  prompting.py    Song-Struktur → YuE2-Anfrage
  jobs.py         Warteschlange + Bibliothek
  download.py     Modell-Download (CLI + Fortschritt)
  plugins/        Plugin-Manager, Lua-Laufzeit
  server.py       FastAPI-Routen (/api/...)
  web/            Oberfläche (HTML/CSS/JS, kein Build)
```

## Lizenz

Code dieses Projekts: vorerst keine Lizenz (alle Rechte beim Autor).
Für die YuE2-**Gewichte** gilt eine eigene Lizenz: CC BY-NC 4.0 mit Zusatzerlaubnis für Creator ([MODEL_LICENSE](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE)).
