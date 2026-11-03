"""Data model of the UI. This is deliberately richer than the YuE2 request.

YuE2 itself only accepts ``style``, ``lyrics``, ``cot``, ``seed``, ``abc``,
``cfg_scale`` (plus sampling overrides). Everything else here (voices,
language, BPM, per-section singers ...) is *our* structure, which
``musicgen_ui.prompting`` compiles down into those fields. Keeping the
structure lets later post-processing stages (voice conversion, arrangement)
use information YuE2 cannot take as input.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Gender = Literal["female", "male", "any"]
PlanMode = Literal["full", "melody", "off"]

# Mirrors yue2.protocol.GenerationConfig defaults (yue2-infer 0.1.6).
# Only used to show defaults in the UI. ``None`` in a request always means
# "let YuE2 use its own default", so upstream changes take effect.
YUE2_DEFAULTS = {
    "score": {"temperature": 0.7, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.005,
              "max_tokens": 4096},
    "audio": {"temperature": 1.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2,
              "max_tokens": 9000},
    "semantic_token_rate_hz": 25,
}


class VoiceSpec(BaseModel):
    id: str = Field("lead", pattern=r"^[A-Za-z0-9_-]{1,32}$")
    gender: Gender = "female"
    # Free descriptors such as "breathy", "raspy", "powerful", "soft".
    descriptors: list[str] = Field(default_factory=list, max_length=8)
    # Reserved for the voice-conversion stage (own/cloned voice profile id).
    voice_profile: str | None = None


class SamplingSettings(BaseModel):
    """Overrides for one YuE2 sampling phase; unset fields keep YuE2 defaults."""

    temperature: float | None = Field(None, ge=0, le=5)
    top_p: float | None = Field(None, gt=0, le=1)
    top_k: int | None = Field(None, ge=1, le=10000)
    repetition_penalty: float | None = Field(None, gt=0, le=5)
    max_tokens: int | None = Field(None, ge=1, le=24000)

    def overrides(self) -> dict | None:
        data = self.model_dump(exclude_none=True)
        return data or None


class LoraRef(BaseModel):
    path: str = Field(..., max_length=2000)   # file in models/lora (or any path)
    strength: float = Field(1.0, ge=0, le=2)


class SongSpec(BaseModel):
    title: str = Field("Untitled", max_length=120)
    style: str = Field("", max_length=1000)
    language: str | None = Field("English", max_length=40)
    bpm: int | None = Field(None, ge=30, le=300)
    instrumental: bool = False
    voices: list[VoiceSpec] = Field(default_factory=lambda: [VoiceSpec()], max_length=6)
    lyrics: str = Field("", max_length=20000)
    # Expert mode: send this style string verbatim instead of the compiled one.
    style_override: str | None = Field(None, max_length=2000)

    plan_mode: PlanMode = "full"
    abc: str | None = Field(None, max_length=200000)
    seed: int | None = Field(None, ge=0, lt=2**63)
    cfg_scale: float | None = Field(None, ge=0, le=20)
    score_sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    audio_sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    # Generate several takes with consecutive seeds and pick the best one.
    candidates: int = Field(1, ge=1, le=8)
    loras: list[LoraRef] = Field(default_factory=list, max_length=8)
