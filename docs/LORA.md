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

## Loader-Design (Roadmap Phase 2)

1. **Erkennen:** Header und Schlüssel lesen und das Format bestimmen. Unbekannte Formate werden mit Begründung abgelehnt, nicht stillschweigend ignoriert.
2. **Umrechnen:** native Namen → m-a-p-Modulnamen, fusionierte Matrizen aufteilen, Skalierung je Format.
3. **Anwenden:** Die Deltas werden *in place* in die Gewichte addiert, mit getrennter Stärke für Planer und Akustik.
   - Die Originale der betroffenen Tensoren werden auf der CPU gesichert. Entfernen stellt sie exakt wieder her, ohne zu subtrahieren und ohne Rundungsfehler.
   - In-place bleibt kompatibel mit CUDA-Graphs und Low-VRAM-Auslagerung, weil die Tensoren dieselben bleiben.
   - Das Basismodell wird nicht neu geladen. Ein Wechsel kostet Sekunden.
4. **Pro Song:** Gewählte LoRAs und Stärken werden im Job gespeichert. Der Worker setzt sie vor dem Song und nur, wenn sie sich ändern.
5. **UI:** Die Dateien liegen in `data/loras/`. Die UI zeigt pro LoRA Hälften, Rang, Trigger-Wort und Basis-Revision.
   Sie fügt das Trigger-Wort in den Stil ein und bietet empfohlene Einstellungen aus `lora.json` an.

**Risiken / Einschränkungen**
- YuE2 lädt das Modell intern und lazy (`pipe._model`). Der Loader hängt sich an diese private Stelle. Deshalb pinnen wir die YuE2-Version (0.1.6) und haben einen Test, der bei Änderungen anschlägt.
- **FP8-Modus:** Er baut die Planer-Layer aus den BF16-Originalen neu auf. Planer-LoRAs und FP8 vertragen sich daher zunächst nicht, und die UI warnt. Akustik-LoRAs sind unbetroffen.
- **Basis-Revision:** Die LoRAs wurden auf `m-a-p/YuE2-3B` Revision `1a96eca` trainiert. Weicht die Revision ab, kommt eine Warnung.
- **Lizenzen:** Die meisten LoRAs sind CC BY-NC (nicht kommerziell). Einige Community-LoRAs bilden reale Künstler nach. Das ist rechtlich heikel, und wir liefern keine solchen LoRAs mit.

## Bekannte LoRAs (Auswahl)

| LoRA | Inhalt | Hälften | Hinweise |
|---|---|---|---|
| [atomtanstudio/lora-library](https://huggingface.co/atomtanstudio/lora-library) DreamPop v2 | Stil | AR + NAR | Trigger `sv_dreampop`, eigenes Bundle-Format |
| ebd. Old School Hip-Hop | Stil (Boom-Bap) | AR + NAR | Trigger `sv_oldschoolhiphop` |
| ebd. Industrial v1 | Stil, experimentell | AR + NAR | laut Autor teils dünner Klang |
| Mothersuperior `nar_lora_joint` | Akustik-Verbesserung | NAR | Grundlage/Begleitung für andere LoRAs |
| Mothersuperior `ar_lora_inst` | Instrumental | AR | braucht `cot=full` |
| [lym00/comfyui_yue2_lora](https://huggingface.co/lym00/comfyui_yue2_lora) | Künstler-Stile | AR + NAR | je 1,9 GB, braucht Begleit-LoRA |

**Für die Stimm-Garantie** gibt es noch keine passenden LoRAs, etwa „immer weibliche Stimme“. Die müssten wir selbst trainieren (Phase 6).

## Training (Roadmap Phase 6)

- [Starnodes ComfyUI-YuE2-Trainer](https://github.com/Starnodes2024/ComfyUI-YuE2-Trainer) (MIT): nur Akustik
  - 5–30 Songs, 24 GB VRAM empfohlen, Rang 32, ca. 1–3 s pro Schritt auf einer 4090, 3000 Schritte
  - Hinweis: Er läuft als ComfyUI-Node. Der Kern (`trainer_core`) ist aber reines PyTorch und könnte als Basis dienen.
- Planer-Training braucht den **Audio→Token-Encoder** der Community: [Mothersuperior realaudio-tokenizer](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4).
  Er bildet MERT-v2-Features auf YuE2s 25-Hz-Tokens ab. Die Trefferquote ist laut Autor mäßig (16 % exakt), reicht aber offenbar fürs Training. Die Lizenz ist noch ungeprüft.
- Eine Stimm-LoRA müsste vor allem die **Akustik** trainieren (Timbre). Das ist genau der Teil, den der Starnodes-Ansatz schon abdeckt.
