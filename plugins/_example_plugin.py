"""Beispiel-Plugin (Python). Dateien mit führendem '_' werden ignoriert –
zum Aktivieren umbenennen, z.B. in 'tidy_lyrics.py'."""


def register(api):
    api.add_preset({"name": "Akustik-Folk", "style": "acoustic folk, fingerpicked guitar, soft brushes",
                    "voices": [{"id": "lead", "gender": "male", "descriptors": ["gentle"]}]})

    def tidy(request):
        # Mehrfache Leerzeilen in den Lyrics zusammenfassen.
        lines, out = request["lyrics"].split("\n"), []
        for line in lines:
            if line.strip() or (out and out[-1].strip()):
                out.append(line)
        return {"lyrics": "\n".join(out)}

    api.on("transform_request", tidy)
