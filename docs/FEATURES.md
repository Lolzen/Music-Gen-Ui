# Wunschliste vs. was YuE2 wirklich kann

Stand: YuE2 `yue2-infer 0.1.6` (Release 9. Sep. 2026). Grundlage sind der Quellcode (`src/yue2/protocol.py`, `pipeline.py`) und die offiziellen Docs, nicht Marketing-Texte.

**Wichtigster Unterschied zu Suno:** Suno ist ein geschlossener Dienst mit vielen Nachbearbeitungsfunktionen (Extend, Personas, Cover, Stems …). YuE2 ist *ein Modell* mit einer schmalen Eingabe:

```text
style      Freitext: Genre, Instrumente, Stimmcharakter, Sprache, Tempo
lyrics     Text mit Abschnitts-Tags [Verse] [Chorus] ...
cot        "full" | "melody" | "off"   → plant das Modell erst eine Partitur?
abc        optional: eigene Partitur (ABC-Notation) statt Planung
seed, cfg_scale, Sampling (temperature, top_p, top_k, repetition_penalty, max_tokens)
```

Alles, was darüber hinausgeht, muss **diese App** bauen, meist als Vor- oder Nachbearbeitung.
Dafür hat YuE2 etwas, das Suno nicht offenlegt: eine **editierbare Partitur** (ABC), die vor dem Audio entsteht.

| Wunsch | YuE2 nativ? | Umsetzung hier | Status |
|---|---|---|---|
| Musikstil-Prompt | ✅ `style` | strukturiert (Stil, Sprache, BPM, Stimmen) → zu einem Stil-Text kompiliert, Vorschau zeigt den exakten Text | **fertig** |
| Lyrics | ✅ `lyrics` | Abschnitts-Buttons, Normalisierung (`[Verse 2]` → `[Verse]`), Warnungen | **fertig** |
| Female/Male | ⚠️ nur als Text-Hinweis | wird in den Stil eingebaut („airy female vocal“), **keine Garantie** | **fertig** |
| Mehrere Stimmen | ⚠️ nur „duet …“ im Stil | jetzt: Stil-Hinweis + `[Chorus @lead,duo]`-Zuordnung wird gespeichert. Später: Voice-Conversion pro Abschnitt | teilweise |
| Stimmen individualisieren | ⚠️ Text-Hinweis | Klang-Beschreibungen pro Stimme; echte Kontrolle erst über Voice-Conversion | teilweise |
| Eigene/kopierte Stimme | ❌ kein Referenz-Audio in YuE2 | Nachbearbeitung: Gesang abtrennen → Voice-Conversion mit Stimmprofil | Phase 4 |
| Randomness | ✅ seed + Sampling | Seed, „Zufall/Kreativität“ (temperature), top_p, CFG, mehrere Takes | **fertig** |
| Titel erweitern | ❌ keine Fortsetzung | 3 Wege, siehe unten | Phase 3 |
| SFX/Clips als Intro/Bridge | ❌ | Arrangement-Timeline in der App, Übergänge per YuE2-Instrumental | Phase 3 |
| Weights (Prompt-Gewichtung wie `(piano:1.3)`) | ❌ | entfällt, siehe unten | gestrichen |
| LoRA | ❌ kein Adapter-Code, kein Trainingscode veröffentlicht | Engine-Schnittstelle offen halten; Training als eigenes Projekt | später |

---

## Details und kritische Punkte

### Female/Male und Stimmcharakter
YuE2 hat **kein** Feld für Geschlecht, Sänger oder Stimme. Die Docs sagen ausdrücklich, dass Stimmbeschreibungen im Stil „die Generierung lenken, ohne einen bestimmten Sänger zu garantieren“. Die UI behandelt das deshalb als *Hinweis* und sagt das auch.
Wer verlässlich eine bestimmte Stimme will, braucht die Voice-Conversion-Stufe (Phase 4).

### Mehrere Stimmen
Laut YuE2-Docs gibt es keine Sängerangabe pro Abschnitt, und Sprecher-Markierungen in den Lyrics erzeugen keinen eigenen Steuerkanal.
Deshalb gibt es hier eine eigene Syntax: `[Chorus @lead,duo]`. Sie wird **vor** YuE2 entfernt und als Metadatum gespeichert.
Später geht der Ablauf so: Gesang abtrennen → Abschnitte zeitlich zuordnen (Partitur-Takte × Tempo oder Forced Alignment) → jeden Abschnitt mit dem Stimmprofil der zugeordneten Stimme konvertieren → wieder zusammenmischen.

Grenze: Singen zwei Stimmen *gleichzeitig* (Harmonien), lässt sich das aus einer Gesangsspur nicht sauber trennen.

### Eigene Stimme / Stimme kopieren
YuE v1 hatte ein Referenz-Audio-Feature (In-Context-Learning, eigener Checkpoint). **YuE2 hat das nicht mehr.** Die Docs sagen: „no audio-reference … argument“.
Der realistische Weg ist eine Nachbearbeitung:

1. Aus mehreren hochgeladenen Titeln den Gesang abtrennen (Stem-Separation, z.B. Demucs / BS-RoFormer via `audio-separator`).
2. Den Gesang säubern, in Stücke schneiden und als Datensatz für ein *Stimmprofil* speichern.
3. Mit einem Singing-Voice-Conversion-Modell die YuE2-Gesangsspur auf das Profil umwandeln. Es gibt zwei Ansätze:
   - zero-shot mit einem Referenz-Clip (z.B. Seed-VC), sofort nutzbar
   - trainiertes Modell (z.B. RVC), braucht ca. 10+ min sauberen Gesang und liefert meist die bessere Qualität
4. Mit dem Instrumental neu mischen.

Die genannten Tools sind Kandidaten. Bei der Umsetzung prüfen wir den aktuellen Stand und die Lizenzen.
**Rechtliches:** Die eigene Stimme ist unproblematisch. Die Stimme realer Künstler zu kopieren berührt Persönlichkeitsrechte, und die YuE2-Modelllizenz verbietet täuschende Nutzung. Die UI soll beim Anlegen eines Profils eine Einverständnis-Bestätigung verlangen.

### Randomness: zwei Arten von Zufall
YuE2 arbeitet in zwei Phasen, und jede hat eigene Sampling-Werte:

| Phase | Default | Wirkung |
|---|---|---|
| **Partitur** (`abc_sampling`) | temp 0.7, top_p 0.9, top_k 30 | *Komposition*: Melodie, Akkorde, Form |
| **Audio** (`semantic_sampling`) | temp 1.0, top_p 0.95, top_k 100, rep. penalty 1.2 | *Aufführung und Klang* |

Daraus ergibt sich ein Workflow, den Suno so nicht bietet: **„Partitur übernehmen“ + neuer Seed** behält die Melodie und würfelt nur den Klang neu. Das ist schon eingebaut.
Die Suno-Regler „Style Influence“ und „Weirdness“ entsprechen grob `cfg_scale` (Stil-Treue) und `temperature`.

### Titel erweitern
YuE2 hat keine Fortsetzungsfunktion. `--resume` prüft nur fertige Ergebnisse. Eine Generierung ist auf **max. 9000 Audio-Tokens bei 25 Hz ≈ 6 Minuten** begrenzt. Es gibt drei mögliche Wege:

1. **Partitur verlängern, dann komplett neu rendern** (offiziell unterstützt): Man hängt neue Abschnitte und Lyrics an die ABC-Partitur an und rendert mit gleichem Stil und Seed.
   Musikalisch ist das stimmig, aber das bisherige Audio klingt danach nicht identisch.
2. **Token-Fortsetzung** (experimentell): Die bereits erzeugten semantischen Tokens werden als Präfix vorgegeben, und das Modell generiert nur den Rest.
   So bleibt der Anfang fast erhalten. Dafür braucht es die internen APIs (`generate_tokens`), also zuerst einen Machbarkeitstest.
3. **Audio anhängen**: Ein neuer Teil wird separat generiert und in der Arrangement-Timeline mit Crossfade angesetzt. Das ist einfach, aber die Naht ist hörbar.

### SFX / Clips als Intro oder Überbrückung
Das ist Audio-Postproduktion und nichts, was YuE2 lösen muss: eine **Arrangement-Timeline** (Song-Teile, hochgeladene Clips, SFX, Crossfades, Lautheit) und ein Renderer mit numpy/ffmpeg.
YuE2 kann zusätzlich kurze **Instrumental-Clips** erzeugen (offizielles `instrumental_clip.py`), passend zum Stil für Übergänge.
Einschränkung: Der Song „weiß“ nichts vom SFX. Ein Bruch im Song selbst entsteht nur, wenn man in der Partitur eine Pause oder ein `[Interlude]` vorsieht.

### Weights (gestrichen)
Gemeint war die Gewichtung einzelner Prompt-Begriffe wie `(piano:1.3)` aus Stable Diffusion. YuE2 kennt das nicht: Der Stil geht als einfacher Text ins Modell, und `cfg_scale` wirkt nur auf den Prompt als Ganzes.
Eine Nachbildung über Wortwiederholung o.ä. wäre unzuverlässig. Der Punkt ist deshalb gestrichen.
Was es weiterhin gibt:
- **Modell-Auswahl:** Checkpoint, VAE (`YuE2-Vae` / `YuE2-Vae-legacy`), FP8 (per Konfiguration)
- **Gesamt-Regler:** Stil-Treue und Kreativität

### LoRA
Im YuE2-Code gibt es weder Adapter-Laden noch Trainingsskripte (Stand 0.1.6). LoRA bräuchte zwei Dinge:
- ein Trainingsverfahren für das Mixture-of-Transformers-Modell (Datenvorbereitung: Audio → semantische Tokens + Partitur, das ist aufwendig)
- einen Adapter-Loader in der Inferenz

**Empfehlung:** Das Training wird ein eigenes Projekt. Diese App bekommt nur einen „Adapter laden“-Punkt in der Engine-Schnittstelle, sobald es ein Format gibt (offiziell oder aus der Community).

### Lizenz
- Code von YuE2: Apache 2.0
- Gewichte: CC BY-NC 4.0 mit Zusatzerlaubnis für Creator. Privatpersonen und Musiker dürfen Ergebnisse monetarisieren, Firmen brauchen eine kommerzielle Lizenz.
