# Akupanelen woonkamer

Interactieve demo die houten akoestische panelen (latten op vilt) over een foto van de woonkamer tekent. Kies hout, vilt, hoeveel van de wand bedekt wordt en de maat van de latten, en vergelijk voor en na.

## Bekijken

De demo staat in `docs/` en is een statische site (HTML, CSS en WebGL, geen build-stap).

Lokaal:

```bash
python3 -m http.server 8765 --directory docs
```

en open http://localhost:8765.

## GitHub Pages

1. Push deze repo naar GitHub.
2. Ga naar **Settings → Pages**.
3. Kies bij **Source** "Deploy from a branch", branch `main`, map `/docs`.

## Assets opnieuw maken

`docs/assets/` (foto, masker, lichtkaart en `scene.json`) wordt gemaakt met:

```bash
python3 -m venv .venv
.venv/bin/pip install opencv-python-headless numpy onnxruntime
.venv/bin/python tools/build_assets.py
```

Voor de optie "Opruimen" schildert het script speelgoed en andere spullen weg met het LaMa-inpaintmodel (Apache-2.0). Dat wordt de eerste keer gedownload naar `.cache/` (ongeveer 200 MB) en komt niet in git.

Voor de optie "Muur" staat in `PAINT_SURFACES` waar behang zit. De verf krijgt per muurvlak de belichting van het behang dat er zat.

De hoeken van de wand en de omtrek van alles wat ervoor staat zijn met de hand ingemeten in `tools/build_assets.py`.
