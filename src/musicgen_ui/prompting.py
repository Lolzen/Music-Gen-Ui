"""Compile a structured :class:`SongSpec` into a native YuE2 request.

Lyrics syntax (superset of YuE2):

    [Verse]              plain YuE2 section tag
    [Verse 2]            numbering is accepted and normalised to [Verse]
    [Chorus @lead,duo]   "@voice ids" mark who sings; stripped before YuE2
                         sees the text (YuE2 has no per-section singer input),
                         kept as metadata for the voice-conversion stage.
"""
from __future__ import annotations

import re
import secrets
from dataclasses import asdict, dataclass, field

from .models import SongSpec, VoiceSpec

# Section names used by the YuE2 docs and its instrumental tooling.
KNOWN_SECTIONS = ("intro", "verse", "pre-chorus", "chorus", "bridge", "interlude", "outro", "instrumental")
DEFAULT_INSTRUMENTAL_FORM = ["Intro", "Verse", "Chorus", "Outro"]

_TAG = re.compile(r"^\s*\[([^\]@]+?)\s*(?:@([^\]]*))?\]\s*$")
_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class Section:
    name: str  # canonical tag, e.g. "Pre-Chorus"
    voices: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


@dataclass
class CompiledRequest:
    """Everything needed to call the engine, plus UI-side metadata."""

    id: str
    style: str
    lyrics: str
    cot: str
    seed: int
    abc: str | None = None
    cfg_scale: float | None = None
    abc_sampling: dict | None = None
    semantic_sampling: dict | None = None
    sections: list[Section] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def engine_kwargs(self) -> dict:
        """Keyword arguments for ``YuE2Pipeline.__call__``."""
        kwargs = {"id": self.id, "style": self.style, "lyrics": self.lyrics,
                  "cot": self.cot, "seed": self.seed}
        if self.abc is not None:
            kwargs["abc"] = self.abc
        if self.cfg_scale is not None:
            kwargs["cfg_scale"] = self.cfg_scale
        if self.abc_sampling:
            kwargs["abc_sampling"] = self.abc_sampling
        if self.semantic_sampling:
            kwargs["semantic_sampling"] = self.semantic_sampling
        return kwargs

    def to_dict(self) -> dict:
        return asdict(self)


def _canonical_section(raw: str) -> tuple[str, bool]:
    name = re.sub(r"\s+\d+$", "", raw.strip())  # "Verse 2" -> "Verse"
    key = name.lower().replace(" ", "-")
    if key == "prechorus":
        key = "pre-chorus"
    if key in KNOWN_SECTIONS:
        return "-".join(p.capitalize() for p in key.split("-")), True
    return name, False


def parse_lyrics(text: str) -> tuple[list[Section], list[str]]:
    sections: list[Section] = []
    warnings: list[str] = []
    current: Section | None = None
    for line in text.replace("\r\n", "\n").split("\n"):
        match = _TAG.match(line)
        if match:
            name, known = _canonical_section(match.group(1))
            if not known:
                warnings.append(f"Unbekannter Abschnitt [{name}] – YuE2 kennt: {', '.join(KNOWN_SECTIONS)}")
            voices = [v.strip() for v in (match.group(2) or "").split(",") if v.strip()]
            current = Section(name, voices)
            sections.append(current)
            continue
        if current is None:
            if not line.strip():
                continue
            current = Section("Verse")
            sections.append(current)
            warnings.append("Lyrics beginnen ohne Abschnitts-Tag – [Verse] wurde angenommen")
        current.lines.append(line.rstrip())
    for section in sections:  # trim blank lines at section ends
        while section.lines and not section.lines[-1].strip():
            section.lines.pop()
        while section.lines and not section.lines[0].strip():
            section.lines.pop(0)
    return sections, warnings


def render_lyrics(sections: list[Section], *, instrumental: bool) -> str:
    """Render sections in native YuE2 form (no voice markers)."""
    if instrumental:
        names = [s.name for s in sections] or DEFAULT_INSTRUMENTAL_FORM
        return "\n\n".join(f"[{n}]" for n in names) + "\n"
    return "\n\n".join(f"[{s.name}]\n" + "\n".join(s.lines) for s in sections).strip()


def voice_phrase(voice: VoiceSpec) -> str:
    words = ", ".join(d.strip() for d in voice.descriptors if d.strip())
    gender = "" if voice.gender == "any" else voice.gender
    return " ".join(p for p in (words, gender, "vocal") if p)


def build_style(spec: SongSpec) -> str:
    if spec.style_override and spec.style_override.strip():
        return spec.style_override.strip()
    parts: list[str] = []
    if spec.language:
        parts.append(spec.language.strip())
    if spec.style.strip():
        parts.append(spec.style.strip().rstrip(","))
    if spec.instrumental or not spec.voices:
        parts.append("instrumental, no vocals")
    elif len(spec.voices) == 1:
        parts.append(voice_phrase(spec.voices[0]))
    else:
        label = "duet" if len(spec.voices) == 2 else "group vocals"
        parts.append(f"{label}: " + " and ".join(voice_phrase(v) for v in spec.voices))
    if spec.bpm:
        parts.append(f"{spec.bpm} BPM")
    return ", ".join(parts)


def make_id(title: str) -> str:
    slug = _ID_SAFE.sub("-", title.strip()).strip("-.")[:60] or "song"
    if not slug[0].isalnum():
        slug = "s" + slug
    return f"{slug}-{secrets.token_hex(3)}"


def compile_spec(spec: SongSpec, *, seed: int | None = None, request_id: str | None = None) -> CompiledRequest:
    if spec.abc is not None and spec.abc.strip() and spec.plan_mode == "off":
        raise ValueError("Eine eigene ABC-Partitur braucht plan_mode 'full' oder 'melody'")

    sections, warnings = parse_lyrics(spec.lyrics)
    voice_ids = {v.id for v in spec.voices}
    for section in sections:
        unknown = [v for v in section.voices if v not in voice_ids]
        if unknown:
            warnings.append(f"[{section.name}] nutzt unbekannte Stimme(n): {', '.join(unknown)}")
    if any(s.voices for s in sections):
        warnings.append("Stimmen-Zuordnung pro Abschnitt wird von YuE2 nicht direkt gesteuert; "
                        "sie ist für die spätere Voice-Conversion-Stufe gespeichert")

    instrumental = spec.instrumental or not spec.voices
    if not instrumental and not any(line.strip() for s in sections for line in s.lines):
        warnings.append("Keine Lyrics angegeben – das Ergebnis wird vermutlich instrumental")
    if spec.abc and spec.bpm:
        warnings.append("Bei eigener Partitur bestimmt das Tempo im ABC-Kopf (Q:) das Timing; BPM im Stil nur als Hinweis")

    return CompiledRequest(
        id=request_id or make_id(spec.title),
        style=build_style(spec),
        lyrics=render_lyrics(sections, instrumental=instrumental),
        cot=spec.plan_mode,
        seed=seed if seed is not None else (spec.seed if spec.seed is not None else secrets.randbelow(2**31)),
        abc=spec.abc if spec.abc and spec.abc.strip() else None,
        cfg_scale=spec.cfg_scale,
        abc_sampling=spec.score_sampling.overrides(),
        semantic_sampling=spec.audio_sampling.overrides(),
        sections=sections,
        warnings=warnings,
    )
