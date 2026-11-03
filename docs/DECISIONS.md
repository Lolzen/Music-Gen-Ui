# Architektur-Entscheidungen

Kurz und begründet. Jede Entscheidung kann später revidiert werden; dann wird hier ein neuer Eintrag ergänzt.

## E1 – Python statt Lua als Hauptsprache

**Entscheidung:** Backend in Python.

- YuE2 ist ein Python-Paket (PyTorch, transformers, optional vLLM). Dasselbe gilt für alles, was später dazukommt: Stem-Separation, Voice-Conversion, SheetSage2, Audio-Verarbeitung.
- Ein Lua-Backend müsste YuE2 trotzdem über einen Python-Prozess ansprechen. Man hätte zwei Sprachen, eine Prozessgrenze und Serialisierung, aber keinen Vorteil.
- Lua glänzt als **eingebettete Skriptsprache**. Genau dafür wird es eingesetzt, siehe E2.

## E2 – Lua-Plugins: ja, als Skriptschicht

**Entscheidung:** Plugins in Python *oder* Lua, Lua über [`lupa`](https://pypi.org/project/lupa/), optional installierbar.

- Lua-Plugins geben eine Tabelle zurück: `presets`, `transform_spec`, `transform_request`, `on_complete` (siehe `plugins/example_presets.lua`).
- Jedes Lua-Plugin bekommt einen eigenen Lua-State mit reduzierter Standardbibliothek (kein `io`, `os.execute`, `require`, `load` …). Daten gehen nur als Tabellen über die Grenze, nie als Python-Objekte.
- Das schützt vor Versehen, ist aber **keine Sicherheitsgrenze**. Plugins gelten als vertrauenswürdiger lokaler Code.
- Grenze: Lua-Plugins sind für Logik gedacht (Vorlagen, Makros, Lyrics-Umformungen, später Batch-Workflows), nicht für Audio- oder ML-Arbeit. Das bleibt bei Python-Plugins.
- Später: eine kleine, kontrollierte Lua-API (z.B. `mgu.submit(spec)` für Batch-Jobs, `mgu.syllables(line)`).

## E3 – FastAPI + eigene Web-Oberfläche statt Gradio

**Entscheidung:** FastAPI-JSON-API plus eine statische Single-Page-UI (Vanilla JS als ES-Modul, **ohne Build-Schritt**).

- Gradio ist für ML-Demos der schnellste Weg. Geplant sind aber Arrangement-Timeline, Partitur-Editor/-Player, Stimmen-Zuordnung pro Abschnitt und eine Bibliothek, und dabei stößt Gradio schnell an Grenzen.
- Die saubere API erlaubt später andere Clients (CLI, ComfyUI-Node, Skripte).
- Ohne Build-Schritt bleibt die Einstiegshürde niedrig. Wird die UI zu groß, kann man auf Svelte/Preact umsteigen, ohne die API zu ändern.

## E4 – Engine-Abstraktion mit Mock-Engine

**Entscheidung:** `Engine`-Protokoll (`generate(request, out_dir, progress, cancelled)`) mit zwei Implementierungen:
`yue2` (echtes Modell) und `mock` (erzeugt eine Platzhalter-Melodie und läuft ohne GPU).

- UI und Tests laufen auf jedem Rechner. Das Modell braucht ca. 24 GB VRAM.
- Weitere Engines (YuE2-Turbo-Server, Remote-GPU, andere Modelle) lassen sich anstecken.

## E5 – Eine Warteschlange, ein Worker

YuE2 verarbeitet pro Pipeline genau eine Anfrage gleichzeitig. Deshalb laufen alle Jobs durch einen Worker-Thread.
Ergebnisse liegen als Ordner in `data/library/<job-id>/` (`job.json` + YuE2-Artefakte) und überleben einen Neustart.

## E6 – Eigene Song-Struktur, kompiliert zur YuE2-Anfrage

Die UI speichert mehr als YuE2 versteht: Stimmen, Zuordnung pro Abschnitt, Sprache, BPM. `prompting.py` kompiliert das zu `style` + `lyrics`.
Die Vorschau zeigt **genau** den Text, den das Modell bekommt. Nur so lassen sich spätere Stufen (Voice-Conversion, Arrangement) mit Informationen füttern, die YuE2 selbst nicht verarbeitet.

## E7 – Eigene Stimme über Nachbearbeitung, nicht im Modell

YuE2 akzeptiert kein Referenz-Audio. Stimmen werden deshalb über *Stem-Separation → Singing-Voice-Conversion → Remix* angepasst (Phase 4). Details in [FEATURES.md](FEATURES.md).

## E8 – LoRA-Training in der App (revidiert)

Ursprünglich war das Training als eigenes Projekt geplant, weil es keinen offiziellen Trainingscode und keinen Audio→Token-Encoder gab.
Seit Mothersuperior beides veröffentlicht hat, ist das Artist-LoRA-Training in die App eingebaut (siehe E11).

## E9 – Installation über Conda

Viele WebUIs auf einem Rechner brauchen getrennte Umgebungen. `install.sh` legt deshalb eine eigene Conda-Umgebung an (Python 3.12, git, ffmpeg).
PyTorch kommt per pip aus dem offiziellen CUDA-Index, wie es YuE2 vorgibt. conda-forge-PyTorch-Builds würden von den YuE2-Pins abweichen.
`start.sh` aktiviert die Umgebung selbst, man muss sie also nicht vorher aktivieren. Ohne die Skripte geht es mit `environment.yml` und den manuellen Schritten in der README.

## E10 – GGUF über eigenen Loader in der offiziellen Pipeline

GGUF-Dateien (HOT-Step-Layout) werden in die offizielle PyTorch-Pipeline geladen, statt ein fremdes Programm fernzusteuern.

- Die Tensor-Namen bilden 1:1 auf `YuE2ForCausalLM`/`YuE2VAE` ab; im VAE wird die gefaltete Weight-Norm exakt zurückgerechnet.
  Tokenizer und Konfiguration werden aus der GGUF erzeugt. YuE2 bekommt dafür ein kleines Gerüst-Verzeichnis, die Gewichte werden direkt eingesetzt.
- Quantisierte Gewichte bleiben auf Wunsch gepackt (`QuantLinear`) und werden pro Aufruf entpackt. Die Kernels stammen aus ComfyUI-GGUF (Apache-2.0).
- YuE2s CUDA-Graphs akzeptieren nur normale BF16-Schichten im Kompositionsteil (AR). Daher bleibt der AR-Teil im Standardmodus „Ausgewogen“ entpackt und nur der Klangteil (NAR) gepackt.
  Der Modus „Maximal“ packt alles und erzwingt `torch-eager`.
- Vorteil: Alle Funktionen der App (Offload, RAM-Notbremse, Protokoll, später LoRA) gelten auch für GGUF, und es ist nichts zusätzlich zu installieren.
- HOT-Step-CPP (schnelle GGML-Kernels, eigene LoRA-Unterstützung) bleibt als optionale zweite Engine geplant, falls die Geschwindigkeit nicht reicht.

## E11 – Artist-LoRA-Training: nachgebaut, eigener Prozess, sicheres Laden

- Die Skripte von Mothersuperior sind Forschungscode mit fest verdrahteten Pfaden und stehen unter CC BY-NC. Sie werden **nicht** übernommen.
  Das Verfahren ist in `training/pipeline.py` nachgebaut; Encoder und Regularisierungs-Paket werden zur Laufzeit von Hugging Face geladen.
- Das Training läuft als **eigener Prozess**. Er belegt die GPU exklusiv (die Generierung wartet), lässt sich sauber stoppen und gibt danach allen Speicher frei.
- `.pt`-Dateien (Regularisierungs-Paket, LoRA-Checkpoints) werden nur mit `weights_only` geladen; zusätzlich erlaubt sind ausschließlich numpy-Typen. Pickles mit Code werden abgelehnt.
- Vor dem Start ist eine Bestätigung nötig: Rechte an den Aufnahmen bzw. Einwilligung der Künstler*innen.

## E12 – „Desktop flüssig halten“ standardmäßig an

Auf einer GPU, die auch den Bildschirm betreibt, blockierte YuE2s Klangsynthese den Desktop so lange, dass Browser abstürzten und Menüs nicht mehr aufgingen. Das sah wie ein Systemabsturz aus.
Eingegrenzt wurde es Schritt für Schritt mit Speicher-Protokoll, Telemetrie, ECC/PCIe-Prüfung und Tests mit und ohne Offload.

- Die Attention der Klangsynthese läuft in Blöcken (YuE2s eigenes `query_chunk_size`), und alle paar NAR-Schichten gibt es eine kurze Pause.
- Das Komponieren mit CUDA-Graphs bleibt unverändert.
- Standard ist „Ausgewogen“. Wer die GPU nur für Musik nutzt (Monitor an der iGPU), kann auf „Aus“ stellen.

