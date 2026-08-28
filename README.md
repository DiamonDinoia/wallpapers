# wallpapers

Generative math and physics wallpapers (3840×2160), procedurally generated with
numpy and rendered as SVG (line art) or PNG (fields), with LaTeX-style equations
typeset in Computer Modern.

Full-resolution PNGs, SVG sources and gallery previews are attached to the
[latest release](https://github.com/DiamonDinoia/wallpapers/releases/latest);
the git repository carries text only — the generator, this README, LICENSE.

## Gallery

Click a preview for the full 4K PNG. (`releases/latest/download/<name>` links.)

| preview | name | motif |
|---|---|---|
| [![chebyshev](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/preview-chebyshev.jpg)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/chebyshev.png) | `chebyshev` | Polar roses of Tₙ(cos θ) = cos(n θ), n = 1…8 |
| [![dejong](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/preview-dejong.jpg)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/dejong.png) | `dejong` | De Jong attractor cloud; hue = flow direction, brightness = density |
| [![lorenz](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/preview-lorenz.jpg)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/lorenz.png) | `lorenz` | Lorenz attractor, x–z projection, time-coded color |
| [![streamlines](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/preview-streamlines.jpg)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/streamlines.png) | `streamlines` | Phase portrait of a vortex + saddle field, RK2 streamlines |
| [![torus](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/preview-torus.jpg)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/torus.png) | `torus` | Perspective wireframe torus with a modal deformation |
| [![wavefield](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/preview-wavefield.jpg)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/wavefield.png) | `wavefield` | Interference \|Σ exp(i·k·r)/√r\|² of 42 point sources |

## Regenerate

Requires `python3` with `numpy`, `matplotlib`, `pillow`, plus `inkscape` to
rasterize the SVG line art:

```
python3 gen_wallpapers.py            # writes *.svg and field *.png in place
for s in lorenz streamlines torus chebyshev; do
    inkscape "$s.svg" -o "$s.png" -w 3840 -h 2160
done
python3 -c 'import gen_wallpapers as g; g.apply_overlays()'   # equations on PNGs
```

Everything regenerates deterministically from constants at the top of the
script (papersizes, palette, seeds); tweak and re-run.

## License

CC0 1.0 — public domain, see [LICENSE](LICENSE).
