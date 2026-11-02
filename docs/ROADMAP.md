# Roadmap

## Phase 0 – Fundament ✅ (dieser Stand)
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
- [ ] Erster Lauf mit YuE2 auf der RTX 4090, Adapter anpassen; Laufzeit und VRAM-Spitzen messen
- [ ] Fortschritt per Server-Sent Events statt Polling
- [ ] Export: FLAC/WAV/MP3, Lautheit normalisieren

## Phase 2 – Partitur-Workflow (YuE2s Stärke)
- [ ] „Nur planen“: Partitur erzeugen, **ohne** Audio zu rendern (schnell)
- [ ] Partitur im Browser abspielen (abcjs-Synth) → vor dem teuren Rendern hören
- [ ] ABC-Editor mit Validierung (Stimmen `Vocal`/`Ins`, Takte, Silben)
- [ ] Cover: Song hochladen → SheetSage2-Transkription (eigene Python-Umgebung) → neuer Stil
- [ ] Takes vergleichen (A/B), Favoriten, Bewertung

## Phase 3 – Arrangement
- [ ] Timeline: Song-Teile, hochgeladene Clips, SFX, Crossfades
- [ ] Übergänge/Intros als YuE2-Instrumental-Clips erzeugen
- [ ] „Erweitern“: Partitur verlängern und neu rendern; Machbarkeitstest Token-Fortsetzung
- [ ] Stems (Gesang/Instrumental) per Separation

## Phase 4 – Stimmen
- [ ] Stimmprofile aus hochgeladenen Titeln (Separation → Datensatz), Einverständnis-Bestätigung
- [ ] Singing-Voice-Conversion (zero-shot und/oder trainiert) als Nachbearbeitung
- [ ] Stimmen-Zuordnung pro Abschnitt anwenden (`[Chorus @duo]`)

## Später / eigenes Projekt
- [ ] LoRA: Adapter laden, sobald es ein Format gibt; Training als eigenes Projekt
- [ ] Lua-API für Batch-Workflows (`mgu.submit`)
- [ ] Remote-/Multi-GPU-Worker (z.B. YuE2-Turbo-Server)

## Ideen-Sammelbecken (TBA)
- Lyrics-Helfer: Silben pro Zeile, Reime, Abschnittsstruktur prüfen
- „Rezept“ exportieren/importieren (Anfrage + Konfiguration → exakt reproduzierbar)
- Benachrichtigung, wenn ein Song fertig ist
- Stil-Bibliothek mit Hörbeispielen
