# LoRA für YuE2

Stand: Nov. 2026. Wird mit den Ergebnissen der Format-Recherche ergänzt.

## Lage
- **Offiziell (m-a-p):** kein LoRA-Loader, kein Trainingscode, kein Audio→Token-Encoder veröffentlicht (`yue2-infer 0.1.6`).
- **Community:** Es gibt bereits Trainer und erste LoRAs, alles noch jung und experimentell:
  - [Starnodes2024/ComfyUI-YuE2-Trainer](https://github.com/Starnodes2024/ComfyUI-YuE2-Trainer): trainiert Stil/Timbre (AR-Zweig eingefroren)
  - [speedyrulz/ComfyUI-YuE2-Trainer](https://github.com/speedyrulz/ComfyUI-YuE2-Trainer): akustisches MODEL- und Planer-CLIP-LoRA, kann beide zusammenführen
  - [Audio→Token-Encoder der Community](https://comfyui-wiki.com/en/news/2026-09-16-yue2-realaudio-encoder): fehlender Baustein, damit man auf eigenen Songs trainieren kann
  - LoRA-Sammlungen: [atomtanstudio/lora-library](https://huggingface.co/atomtanstudio/lora-library), [lym00/comfyui_yue2_lora](https://huggingface.co/lym00/comfyui_yue2_lora)
- Die Projekte unterscheiden sich darin, ob nur der akustische Teil (NAR) oder auch der Planer (AR) trainiert wird. Der Loader muss beides können.

## Plan
1. Loader für die verbreiteten Dateiformate (Schlüssel-Namen auf YuE2-Module abbilden), Stärke pro LoRA, getrennt nach Planer/Akustik
2. Anwenden ohne das Basismodell neu zu laden (Gewichte nur temporär verändern oder Adapter einhängen)
3. Später: Training aus der UI (Roadmap Phase 6)
