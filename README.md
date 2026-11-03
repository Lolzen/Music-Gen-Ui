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
| `--access-log` | alle HTTP-Anfragen im Terminal anzeigen; standardmäßig werden die regelmäßigen Status-Abfragen der Oberfläche ausgeblendet |

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
Daneben zeigen zwei Leisten den Grafikspeicher (**VRAM**) und den Arbeitsspeicher (**RAM**):

- **Volle Füllung (lila):** aktuell belegt (alle Programme)
- **Transparente Füllung (orange, gestrichelte Kante):** **Prognose** für den Spitzenwert während eines Songs mit dem gewählten Modell und den aktuellen Einstellungen. Sie passt sich sofort an, wenn du Modell, GGUF-Modus, FP8, Offload, Stil-Treue oder Plan-Modus änderst.
- **Senkrechter Strich (VRAM):** Grenze für YuE2 (Gesamt-VRAM minus VRAM-Reserve). Liegt die Prognose darüber, wird sie rot und warnt.
- „geschätzt“: berechnet aus Modellgröße, KV-Cache, CUDA-Graphs, Klangsynthese und Decoder. Nach dem ersten echten Song mit derselben Konfiguration steht dort **„gemessen“** mit den tatsächlichen Spitzenwerten (gespeichert in `data/stats/`). Details zeigt der Tooltip.
- Im Einstellungsdialog zeigt dieselbe Anzeige die Prognose für die **noch nicht gespeicherten** Werte. **Modell laden** lädt das Modell vorab, **Entladen** gibt den Grafikspeicher frei, z.B. für eine andere WebUI.

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

## Modelle & Ordner

Wie bei Bildgenerierungs-UIs liegen Modelle im Ordner `models/` (anderer Ort: `--models-dir PFAD` oder `MGU_MODELS_DIR`):

```text
models/
  yue2/   Checkpoints: Hugging-Face-Ordner (config.json + *.safetensors) oder *.gguf
  vae/    Audio-Decoder (VAE): Hugging-Face-Ordner oder *.gguf
  lora/   LoRA-Adapter (*.safetensors, optional mit lora.json)
```

Oben in der Oberfläche wählst du **Modell** und **VAE** aus. 🔄 liest den Ordner neu ein. Zusätzlich stehen immer die offiziellen Hugging-Face-Modelle zur Auswahl; die liegen im Hugging-Face-Cache.
Dateien, die nicht passen, erscheinen ausgegraut mit ⚠. Der Grund steht im Tooltip. Beispiele:
- YuE-**v1**-Modelle wie `YuE-s2-1B-general` (Vorgänger, eigene Architektur)
- ein Checkpoint im `vae/`-Ordner
- beschädigte Dateien

### GGUF (quantisierte Modelle)

Unterstützt werden YuE2-GGUFs im Layout von HOT-Step-CPP, z.B. [scragnog/YuE2-GGUF](https://huggingface.co/scragnog/YuE2-GGUF).
Sie laufen in der offiziellen YuE2-Pipeline, deshalb funktionieren Offload, RAM-Notbremse und Speicher-Protokoll ganz normal. Herunterladen, z.B.:

```bash
hf download scragnog/YuE2-GGUF yue2-lm-q8_0.gguf --local-dir models/yue2
```
```bash
hf download scragnog/YuE2-GGUF yue2-vae-standard-f32.gguf --local-dir models/vae
```

- **Checkpoint:** `yue2-lm-*.gguf` gehört nach `models/yue2/`. Der Hersteller empfiehlt `q8_0` (praktisch verlustfrei). Ab Q4 abwärts sinkt die Klangqualität stärker.
- **VAE:** `yue2-vae-*.gguf` gehört nach `models/vae/`. Alternativ einfach das offizielle `m-a-p/YuE2-Vae` wählen.
- Tokenizer und Konfiguration stecken in der GGUF-Datei. GGUF-Modelle laufen also komplett offline.

**GGUF-Modus** (⚙ Einstellungen → Speicher & Offload):

| Modus | Komponieren (AR) | Klang (NAR) | Tempo | VRAM |
|---|---|---|---|---|
| **Ausgewogen** (Standard) | entpackt | bleibt gepackt | fast wie BF16 | Q8: ca. 1,3 GB weniger, Q4: ca. 2 GB weniger |
| BF16 | entpackt | entpackt | volle Geschwindigkeit | wie das Original |
| Maximal | gepackt | gepackt | Komponieren deutlich langsamer (ohne CUDA-Graphs) | ca. 3–5 GB weniger |

Ebenfalls unterstützt werden **Checkpoints im yue2.cpp-Layout**, z.B. [Serveurperso/YuE2-GGUF](https://huggingface.co/Serveurperso/YuE2-GGUF) (`YuE2-3B-Q8_0.gguf` usw.).
Deren VAE-Datei hat ein eigenes Layout und wird noch nicht unterstützt. Dafür das offizielle `m-a-p/YuE2-Vae` wählen.
Nicht unterstützt sind GGUFs in anderen Layouts (z.B. LattifAI/EdgeSpeak) und Modelle des Vorgängers YuE v1. Sie erscheinen ausgegraut, der Grund steht im Tooltip.
Die Entpack-Routinen stammen aus [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (Apache-2.0, siehe `licenses/`).

## LoRAs verwenden

LoRA-Dateien gehören nach `models/lora/` (auch in Unterordner). Im Song-Formular wählst du sie unter **LoRAs** aus, jede mit eigener Stärke.

Unterstützt werden diese Formate:
- **Mothersuperior / Music-Gen-UI:** z.B. eigene Artist-LoRAs oder `nar_lora_joint_v9`
- **ComfyUI-Layout:** Starnodes-Trainer, lym00, `*_comfyui`
- **Starnodes intern**

`.pt`-Dateien werden nur geladen, wenn sie keinen ausführbaren Code enthalten. Im Zweifel die `.safetensors`-Version nehmen.

Die Abzeichen zeigen, welchen Teil von YuE2 eine LoRA verändert:
- **Komposition:** Melodie und Aufbau, z.B. Artist-LoRAs
- **Klang:** nur den Sound

Bei Kompositions-LoRAs wird das Trigger-Wort in den Stil eingefügt und der Plan-Modus auf „off“ gestellt (so wurden sie trainiert).
LoRAs werden direkt in die Gewichte eingerechnet und beim Wechsel exakt zurückgesetzt. Mit GGUF-Modellen funktionieren sie ebenso.

## Eigene Artist-LoRA trainieren

Tab **LoRA-Training**. Die App trainiert den Kompositions-Teil von YuE2 auf deine eigenen Songs. Das Verfahren und den Audio→Token-Encoder hat [Mothersuperior](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4) veröffentlicht (CC BY-NC 4.0, nur nicht-kommerziell).

1. **Projekt anlegen:** Name, Trigger-Wort (ein Fantasiewort, z.B. `xyzq`) und eine Klang-Beschreibung
2. **Songs hochladen:** 10–30 Songs (FLAC/WAV/MP3), je Song die **vollständigen** Lyrics mit `[Verse]`/`[Chorus]` …; gekürzte Lyrics verderben die Song-Struktur
3. **Einstellungen:** Die Standardwerte entsprechen dem Original (1600 Schritte, Rang 64, halb eigene Songs, halb Regularisierung).
   Nicht länger trainieren: Ab ca. 1500 Schritten lernt das Modell die Songs auswendig.
4. **Rechte bestätigen**, dann **Training starten**
   - Die Song-Generierung pausiert so lange, weil die GPU belegt ist (ca. 14–18 GB).
   - Beim ersten Mal werden MERT (~1 GB), der Encoder und das Regularisierungs-Paket (~100 MB) heruntergeladen.
5. **Checkpoints probehören:** Ab Schritt 600 entsteht alle 200 Schritte ein Checkpoint.
   - „Alle Checkpoints probehören“ erzeugt mit jedem denselben Song (gleicher Seed).
   - Den besten übernimmst du mit **Als LoRA übernehmen**. Der Autor fand Schritt ~800 am besten.
6. **Klang-LoRA:** Die passende Klang-LoRA (`nar_lora_joint_v9`) lässt sich mit einem Klick herunterladen und wird beim Probehören automatisch mitbenutzt.



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
| System friert ein, obwohl RAM/VRAM laut Protokoll frei sind (oft beim Wechsel zu `nar`) | Wahrscheinlich Grafikkarte/Treiber statt Speicher, siehe [Absturz ohne vollen Speicher](#absturz-ohne-vollen-speicher) |
| `yue2 doctor` meldet Fehler | Treiber/CUDA-Version prüfen (`nvidia-smi`), notfalls `./install.sh --cuda cu126` |
| Port belegt | `./start.sh --port 7861` |
| `install.sh`: viele `Retrying … Failed to resolve 'pypi.ngc.nvidia.com'` | Eine nicht erreichbare Paketquelle steht in deiner globalen pip-Konfiguration. `install.sh` erkennt das inzwischen und ignoriert sie für die Installation. Dauerhaft entfernen: `python -m pip config debug` zeigt die Datei, dann `python -m pip config unset global.extra-index-url` (bzw. `user.`/`site.`), oder `PIP_EXTRA_INDEX_URL` aus der Shell-Konfiguration löschen |
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

### Desktop/Browser frieren während der Klangsynthese ein

Treibt die Grafikkarte auch den Bildschirm, kann es so aussehen, als sei das System abgestürzt:
- Browser schließen sich, Menüs und neue Fenster öffnen sich nicht.
- Der Server läuft aber weiter und lässt sich mit Strg+C beenden. Danach ist alles wieder normal, ein Neustart ist **nicht** nötig.

**Ursache:** Die offizielle YuE2-Klangsynthese rechnet die Attention über die ganze Sequenz (~24.000 Tokens) in einem Stück pro Schicht. Solange das läuft, kommt der Desktop nicht an die GPU, und Chrome/Chromium beendet sich über seinen GPU-Watchdog.

**Abhilfe:** ⚙ Einstellungen → Speicher & Offload → **Desktop flüssig halten**:

| Stufe | Wirkung |
|---|---|
| *Ausgewogen* (Standard) | Attention in Blöcken von 4.096 Abfragen, alle 4 Schichten 1 ms Pause für den Desktop |
| *Flüssig* | Blöcke von 2.048, alle 2 Schichten 3 ms Pause; die Klangsynthese wird etwas langsamer |
| *Aus* | volle Geschwindigkeit wie das Original |

Das Komponieren (CUDA-Graphs) bleibt unverändert schnell.
Wer eine Grafikeinheit in der CPU hat, kann den Monitor stattdessen ans Mainboard anschließen. Dann gehört die 4090 ganz der Musik.

Hängt eine GPU-Abfrage, schreibt das Speicher-Protokoll „GPU antwortet seit N s nicht“, die RAM-Werte laufen weiter.

### Absturz ohne vollen Speicher

Echter Absturz: Auch Strg+C reagiert nicht mehr, der Bildschirm wird schwarz oder der PC startet neu.
Bleibt im Speicher-Protokoll genug RAM und VRAM frei und friert das System trotzdem ein, liegt die Ursache fast immer unterhalb der App: beim Grafiktreiber oder der Grafikkarte.
Ein normaler Prozess kann Linux sonst nicht einfrieren. Besonders verdächtig ist der Wechsel von `audio` (Komponieren, wenig Last) zu `nar` (Klangsynthese, Volllast): An dieser Stelle zieht eine 4090 kurz deutlich mehr als 450 W.

Das Protokoll zeigt jede Sekunde Leistung, Temperatur, Takt, Last und PCIe-Verbindung der Karte (`GPU 312/450 W  61 °C … PCIe Gen4 x16  Replays 0`). Steigt der Replay-Zähler während eines Songs, ist die PCIe-Verbindung instabil (Riser-Kabel, Slot); dann im BIOS den Slot auf Gen 3 stellen. Ein abgebrochener Song nennt nach dem Neustart die Phase und die letzte Messung.

**Test: Leistungsgrenze senken.** Den gleichen Song (gleicher Seed) noch einmal erzeugen:

```bash
sudo nvidia-smi -pl 300
```

**Test ohne YuE2: GPU-Stresstest.** Er erzeugt dieselbe schlagartige BF16-Volllast wie die Klangsynthese und schreibt jede Sekunde Leistung und Temperatur nach `data/logs/stress-*.log`. Vorher offene Arbeit speichern, er kann das System einfrieren:

```bash
conda activate music-gen-ui && python -m musicgen_ui.stress --mode instant --seconds 60
```

- Friert er ebenfalls ein, liegt es an Hardware oder Stromversorgung, nicht an YuE2.
- Läuft `--mode ramp` (Last langsam steigend) durch, `instant` aber nicht, sind es Lastsprünge: Netzteil oder 12VHPWR-Stecker.
- Hilft eine niedrigere Taktgrenze (`sudo nvidia-smi -lgc 210,2400`, zurücksetzen mit `sudo nvidia-smi -rgc`), ist die Werks-Übertaktung unter dieser Last nicht stabil.

Läuft er jetzt durch, waren es Lastspitzen. Dann Netzteil und 12VHPWR-Stecker prüfen oder dauerhaft mit Limit arbeiten (bei 350 W ist die 4090 nur wenig langsamer).
Nach einem Neustart gilt wieder das Standardlimit. Zurücksetzen geht auch mit `sudo nvidia-smi -pl 450`.

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
  lora.py         LoRA-Formate lesen, einrechnen, exakt zurücksetzen
  gguf_support/   GGUF-Loader + Entpack-Kernels
  training/       Artist-LoRA-Training (Pipeline, Worker-Prozess, Projekte)
  server.py       FastAPI-Routen (/api/...)
  web/            Oberfläche (HTML/CSS/JS, kein Build)
```

## Lizenz

Code dieses Projekts: vorerst keine Lizenz (alle Rechte beim Autor).
Für die YuE2-**Gewichte** gilt eine eigene Lizenz: CC BY-NC 4.0 mit Zusatzerlaubnis für Creator ([MODEL_LICENSE](https://github.com/multimodal-art-projection/YuE/blob/main/MODEL_LICENSE)).
