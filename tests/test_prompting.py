import pytest

from musicgen_ui.models import SongSpec, VoiceSpec
from musicgen_ui.prompting import build_style, compile_spec, parse_lyrics, render_lyrics


def test_style_matches_yue2_example_shape():
    spec = SongSpec(style="warm piano pop, light drums", language="English", bpm=88,
                    voices=[VoiceSpec(gender="female", descriptors=["expressive"])])
    assert build_style(spec) == "English, warm piano pop, light drums, expressive female vocal, 88 BPM"


def test_duet_and_instrumental_style():
    duet = SongSpec(style="synthwave", language=None,
                    voices=[VoiceSpec(id="a", gender="male"), VoiceSpec(id="b", gender="female", descriptors=["airy"])])
    assert build_style(duet) == "synthwave, duet: male vocal and airy female vocal"
    inst = SongSpec(style="lo-fi", language=None, instrumental=True)
    assert build_style(inst) == "lo-fi, instrumental, no vocals"


def test_style_override_wins():
    assert build_style(SongSpec(style="x", style_override="  exact text ")) == "exact text"


def test_parse_lyrics_voice_markers_and_normalisation():
    sections, warnings = parse_lyrics("[Verse 1 @lead]\nla\n\n[prechorus]\nhm\n[Chorus @lead, duo]\noh\n")
    assert [(s.name, s.voices) for s in sections] == [("Verse", ["lead"]), ("Pre-Chorus", []),
                                                      ("Chorus", ["lead", "duo"])]
    assert warnings == []
    assert render_lyrics(sections, instrumental=False) == "[Verse]\nla\n\n[Pre-Chorus]\nhm\n\n[Chorus]\noh"


def test_unknown_section_and_untagged_start_warn():
    sections, warnings = parse_lyrics("first line\n[Rap]\nyo")
    assert [s.name for s in sections] == ["Verse", "Rap"]
    assert len(warnings) == 2


def test_instrumental_lyrics_are_tags_only():
    sections, _ = parse_lyrics("[Intro]\nla la\n[Outro]\n")
    assert render_lyrics(sections, instrumental=True) == "[Intro]\n\n[Outro]\n"
    assert render_lyrics([], instrumental=True) == "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n"


def test_compile_engine_kwargs_only_contains_set_overrides():
    spec = SongSpec(title="My Song!", style="pop", lyrics="[Verse]\nhi", seed=5)
    req = compile_spec(spec)
    kwargs = req.engine_kwargs()
    assert set(kwargs) == {"id", "style", "lyrics", "cot", "seed"}
    assert kwargs["seed"] == 5 and kwargs["id"].startswith("My-Song")

    spec = SongSpec(style="pop", lyrics="[Verse]\nhi", cfg_scale=2.0,
                    audio_sampling={"temperature": 1.3})
    kwargs = compile_spec(spec).engine_kwargs()
    assert kwargs["cfg_scale"] == 2.0
    assert kwargs["semantic_sampling"] == {"temperature": 1.3}


def test_compile_warnings():
    spec = SongSpec(lyrics="[Chorus @ghost]\nla", voices=[VoiceSpec(id="lead")])
    warnings = compile_spec(spec).warnings
    assert any("ghost" in w for w in warnings)
    assert compile_spec(SongSpec(lyrics="")).warnings  # empty lyrics warns


def test_abc_requires_plan():
    with pytest.raises(ValueError):
        compile_spec(SongSpec(abc="X:1\nK:C\nC|", plan_mode="off"))
