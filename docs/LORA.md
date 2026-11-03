# LoRA für YuE2

Stand: 2. Nov. 2026. Die Angaben stammen aus dem Quellcode der Community-Trainer und den Model-Cards. Die Ökosysteme sind jung, es bewegt sich viel.

## Lage

- **Offiziell (m-a-p):** kein LoRA-Loader, kein Trainingscode, kein Audio→Token-Encoder (`yue2-infer 0.1.6`).
- **Community:** Trainer, ein nachgebauter Audio→Token-Encoder und erste LoRAs existieren.
  Es gibt aber **kein einheitliches Dateiformat**. Fast alles zielt auf ComfyUIs natives YuE2-Layout.

### Zwei Hälften des Modells

YuE2 hat zwei Hälften. Ein LoRA kann eine oder beide verändern.

| Hälfte | Aufgabe | Module im m-a-p-Code | ComfyUI-Slot | Schlüssel-Präfix |
|---|---|---|---|---|
| **Planer (AR)** | Partitur und Audio-Tokens: *was* gespielt/gesungen wird | `model.layers.N.self_attn.*`, `mlp.*` | CLIP | `text_encoders.` |
| **Akustik (NAR)** | Klang aus Tokens: *wie* es klingt (Timbre, Mix) | `model.layers.N.nar_self_attn.*`, `nar_mlp.*`, `llm2vae`, `vae2llm`, `time_embedder` | MODEL | `diffusion_model.` |

Ein reines Akustik-LoRA ändert nur den Klang und nicht die Melodie. Der Planer ist davon unberührt, weil AR-Positionen die NAR-Positionen nicht sehen.

## Dateiformate

| Quelle | Schlüssel (Beispiel) | Hälfte | alpha | Skalierung |
|---|---|---|---|---|
| ComfyUI-nativ (Starnodes-Trainer, Mothersuperior, lym00) | `diffusion_model.model.layers.0.self_attn.qkv_proj.lora_down.weight` / `.lora_up.weight` | NAR, bei lym00 auch AR (`text_encoders.…`) | Starnodes: `.alpha` = fusionierter Rang (→ Faktor 1.0); lym00: kein alpha, Faktor in `lora_down` eingerechnet | `alpha / rang(lora_down)` |
| Starnodes intern `yue2-lora-v1` | `model.layers.0.nar_self_attn.q_proj.lora_down.weight` | NAR | im Header | `alpha / rank` |
| YuE2-Studio-Rohformat | `yue2.blk.0.attn_q.lora_A` (auch LoKr) | AR + NAR (`nar_`-Präfix) | – | – |
| Atomtan „Sound & Vision“-Bundles | nicht veröffentlicht | AR + NAR | ? | eigener Loader |
| goldhub RS-Finetunes | **keine LoRAs**, komplette Gewichte (Merges) | – | – | – |

**Fusionierte Gewichte:** ComfyUI fasst `q/k/v` zu `qkv_proj` (Zeilen 2048 | 1024 | 1024) und `gate/up` zu `gate_up_proj` (6144 | 6144) zusammen. Die LoRA-Matrizen sind dort blockdiagonal.
Der m-a-p-Code hat getrennte Layer. Der Loader rechnet deshalb `ΔW = up · down · scale` und teilt die Zeilen auf `q/k/v` bzw. `gate/up` auf.

Weitere Header-Felder, die wir nutzen: `trigger_word` / `modelspec.trigger_phrase`, `base_model`, `source_rank`, `source_steps`.
Atomtan liefert zusätzlich eine `lora.json` mit Trigger-Wort und empfohlenen Generierungs-Einstellungen, z.B. `cot: off`, `cfg_scale: 1`.

**Abhängigkeiten:** Manche LoRAs setzen ein *Begleit-LoRA* voraus. Die lym00-LoRAs brauchen z.B. `nar_lora_joint_v9_comfyui` von Mothersuperior mit Stärke 1.0. Der Loader muss deshalb mehrere LoRAs gleichzeitig können.

## Loader (umgesetzt: `src/musicgen_ui/lora.py`)

- **Formate:** Mothersuperior/Music-Gen-UI (`layers.N.[nar_]self_attn.q_proj.lora_A|B`, NAR mit vollem `vae2llm`/`llm2vae`; `.pt` nur über sicheres Laden), ComfyUI-Layout (fusionierte `qkv_proj`/`gate_up_proj` werden anhand der Modellmaße zerlegt, `alpha`, `.diff`/`.diff_b`) und Starnodes intern.
- **Anwenden:** Deltas werden vor jedem Song direkt in die Gewichte eingerechnet, aber nur, wenn sich die Auswahl geändert hat. Das ist kompatibel mit CUDA-Graphs, FP8 (wird pro Song aus den aktuellen Gewichten gebaut) und Offload.
- **Entfernen:** Die betroffenen Tensoren werden bit-genau aus der Modellquelle neu gelesen (HF-safetensors per mmap oder GGUF). Es werden keine RAM-Kopien vorgehalten.
- **GGUF:** Gepackte Schichten (`QuantLinear`) bekommen die LoRA als Seitenpfad (`x·Aᵀ·Bᵀ`).
- **Pro Song:** Gewählte LoRAs und Stärken werden mit dem Song gespeichert. Ein Trigger-Wort und empfohlene Einstellungen kommen aus dem Header oder aus `lora.json` / `<datei>.json`.

## Training (umgesetzt: `src/musicgen_ui/training/`)

Das Verfahren folgt dem von Mothersuperior veröffentlichten Ablauf (nachgebaut, nicht kopiert):

1. Audio → MERT-v2-FullSong, Schicht 20, auf 25 Hz interpoliert
2. → Audio→Token-Encoder (8-Layer-Transformer, Versionen v4/v5/v8/v9) → YuE2-Semantik-Tokens
3. Datensatz: Stil `"<trigger>, in the style of <trigger>. <Beschreibung>"` + vollständige Lyrics, ohne Partitur (`cot=off`), 50/50 gemischt mit dem Regularisierungs-Paket (4.732 YuE2-Songs)
4. LoRA Rang 64 auf den AR-Projektionen (q/k/v/o, gate/up/down); AdamW (0.9/0.95), 50 Schritte Warm-up, Cosine bis 0,2·LR, Gradient-Checkpointing, max. 12.288 Tokens pro Sequenz
5. Checkpoints im Mothersuperior-Layout. Der eigene Loader und `ckpt_io.py` des Autors lesen sie beide.

Die Lyric-Cursor-Hilfsaufgabe des Originals (Demucs + MMS-Alignment) ist noch nicht eingebaut. Sie ist laut Autor optional.
Das Training läuft als eigener Prozess. Ein Stopp speichert den letzten Stand; danach ist die GPU wieder vollständig frei.

## Bekannte LoRAs (Auswahl)

| LoRA | Inhalt | Hälften | Hinweise |
|---|---|---|---|
| [atomtanstudio/lora-library](https://huggingface.co/atomtanstudio/lora-library) DreamPop v2 | Stil | AR + NAR | Trigger `sv_dreampop`, eigenes Bundle-Format |
| ebd. Old School Hip-Hop | Stil (Boom-Bap) | AR + NAR | Trigger `sv_oldschoolhiphop` |
| ebd. Industrial v1 | Stil, experimentell | AR + NAR | laut Autor teils dünner Klang |
| Mothersuperior `nar_lora_joint` | Akustik-Verbesserung | NAR | Grundlage/Begleitung für andere LoRAs |
| Mothersuperior `ar_lora_inst` | Instrumental | AR | braucht `cot=full` |
| [lym00/comfyui_yue2_lora](https://huggingface.co/lym00/comfyui_yue2_lora) | Künstler-Stile | AR + NAR | je 1,9 GB, braucht Begleit-LoRA |

**Für die Stimm-Garantie** gibt es noch keine passenden LoRAs, etwa „immer weibliche Stimme“. Mit dem eingebauten Training lassen sie sich jetzt selbst erstellen, z.B. aus Songs mit nur weiblichem Gesang.

## Andere Trainer

- [Starnodes ComfyUI-YuE2-Trainer](https://github.com/Starnodes2024/ComfyUI-YuE2-Trainer) (MIT): nur Akustik
  - 5–30 Songs, 24 GB VRAM empfohlen, Rang 32, ca. 1–3 s pro Schritt auf einer 4090, 3000 Schritte
  - Hinweis: Er läuft als ComfyUI-Node. Der Kern (`trainer_core`) ist aber reines PyTorch und könnte als Basis dienen.
- Planer-Training braucht den **Audio→Token-Encoder** der Community: [Mothersuperior realaudio-tokenizer](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4).
  Er bildet MERT-v2-Features auf YuE2s 25-Hz-Tokens ab. Die Trefferquote ist laut Autor mäßig (16 % exakt), reicht aber offenbar fürs Training. Die Lizenz ist noch ungeprüft.
- Eine Stimm-LoRA müsste vor allem die **Akustik** trainieren (Timbre). Das ist genau der Teil, den der Starnodes-Ansatz schon abdeckt.
