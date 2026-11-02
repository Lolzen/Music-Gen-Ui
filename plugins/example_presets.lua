-- Beispiel-Plugin (Lua). Wird automatisch aus ./plugins geladen, wenn 'lupa' installiert ist.
-- Ein Lua-Plugin gibt eine Tabelle zurück. Alle Felder sind optional.

local plugin = {}

-- Vorlagen erscheinen im "Vorlage"-Dropdown. Felder entsprechen dem SongSpec.
plugin.presets = {
  {
    name = "Piano-Pop Ballade",
    style = "warm piano pop, acoustic piano, rounded bass and light drums, lyrical memorable melody",
    language = "English",
    bpm = 88,
    voices = { { id = "lead", gender = "female", descriptors = { "expressive" } } },
  },
  {
    name = "Synthwave Duett",
    style = "80s synthwave, analog synth pads, gated reverb drums, driving bassline",
    bpm = 110,
    voices = {
      { id = "lead", gender = "male", descriptors = { "warm", "baritone" } },
      { id = "duo",  gender = "female", descriptors = { "airy" } },
    },
  },
  {
    name = "Lo-Fi Instrumental",
    style = "lo-fi hip hop, dusty drums, mellow rhodes, vinyl crackle",
    bpm = 75,
    instrumental = true,
  },
}

-- Stil-Makros: "{lofi}" im Stil-Feld wird durch den hinterlegten Text ersetzt.
local macros = {
  lofi = "lo-fi, tape saturation, vinyl crackle",
  epic = "cinematic, orchestral strings, big taiko drums, choir",
}

function plugin.transform_spec(spec)
  spec.style = (spec.style or ""):gsub("{(%w+)}", function(key)
    return macros[key] -- nil lässt unbekannte Makros stehen
  end)
  return spec
end

return plugin
