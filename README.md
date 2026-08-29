# wallpapers

Generative math and physics wallpapers (3840×2160), procedurally generated with
numpy and rendered as SVG (line art) or PNG (fields), with LaTeX-style equations
typeset in Computer Modern.

Full-resolution PNGs and SVG sources are attached to the
[latest release](https://github.com/DiamonDinoia/wallpapers/releases/latest);
the git repository carries text only: the generator, this README, LICENSE.

## Gallery

Thumbnails link to the full 4K PNG; the wallpaper SVG line art is in the same
release.

| image | name | motif |
|---|---|---|
| [![chebyshev](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/chebyshev-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/chebyshev.png) | `chebyshev` | Polar roses of Tₙ(cosθ) = cos nθ, n = 1…8 |
| [![dejong](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/dejong-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/dejong.png) | `dejong` | De Jong attractor cloud; hue = flow direction, brightness = density |
| [![gamma](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/gamma-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/gamma.png) | `gamma` | Domain colouring of Γ(z); hue = argΓ, one brightness band per octave of |Γ| |
| [![helmholtz](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/helmholtz-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/helmholtz.png) | `helmholtz` | Plane wave on a sound-soft cylinder, Mie series, ka = 12 |
| [![littlewood](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/littlewood-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/littlewood.png) | `littlewood` | Roots of all 2²⁰ Littlewood polynomials Σ aₖzᵏ, aₖ = ±1, a₂₀ = 1 |
| [![lorenz](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/lorenz-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/lorenz.png) | `lorenz` | Lorenz attractor, x–z projection, time-coded color |
| [![pseudospectra](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/pseudospectra-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/pseudospectra.png) | `pseudospectra` | ε-pseudospectra of Grcar(32): level sets of σₘᵢₙ(zI − A) |
| [![spread](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/spread-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/spread.png) | `spread` | Type-1 NUFFT spreading: ES-kernel decade level sets on the σ = 2 grid |
| [![stability](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/stability-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/stability.png) | `stability` | Absolute stability regions |Rₚ(z)| = 1, Rₚ = Σ zᵏ/k!, p = 1…8 |
| [![streamlines](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/streamlines-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/streamlines.png) | `streamlines` | Phase portrait of a vortex + saddle field, RK2 streamlines |
| [![talbot](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/talbot-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/talbot.png) | `talbot` | Talbot carpet: Fresnel self-imaging of a duty-½ grating |
| [![torus](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/torus-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/torus.png) | `torus` | Perspective wireframe torus with a modal deformation |
| [![wavefield](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/wavefield-thumb.png)](https://github.com/DiamonDinoia/wallpapers/releases/latest/download/wavefield.png) | `wavefield` | Interference |Σⱼ exp(i·kⱼ·rⱼ)/√rⱼ|² of 42 free-space sources |

## Regenerate

Requires `python3` with `numpy`, `scipy`, `matplotlib`, `pillow`, plus `inkscape`
to rasterize the SVG line art:

```
python3 gen_wallpapers.py
```

One command runs the whole pipeline: the fields and the SVGs render into `raw/`,
`rasterize` calls inkscape on the line art, `selfcheck` asserts the kernel level
sets, the emptiness test and the framing, `apply_overlays` places each caption,
and `thumbs` writes `thumbs/*-thumb.png` for the gallery above.

All thirteen plates stay full-bleed 4K, with the equation set where the artwork
leaves room. The placement is measured, not guessed: luminance is
max-pooled onto an 8 px grid, and a summed-area table then rejects every window
holding a single pixel above the background. The four inset corners are tried
first, then any window anywhere in the plate, scored by its distance to the
nearest ink, which puts the caption in the middle of the largest hole rather than
against its edge. `littlewood` lands in the eye of the root annulus and
`helmholtz` in the acoustic shadow behind the cylinder.

Several plates are drawn off-centre to open that room, the attractor moved aside
for the equation. Three name their position, where faint structure covers the
plate but is not what the eye reads: `streamlines` takes the sparse streaks top
left, while `spread` and `talbot` name only a band and let the caption slide
along it to the emptiest x. The slide is worth it: 3.4% ink against 5.0% centred
for `spread`, 1.4% against 52.4% for `talbot`, whose carpet leaves no window
clear anywhere. Placement always reads `raw/` and writes the top level, so
re-running it is a no-op.

Everything regenerates deterministically from constants at the top of the
script (papersizes, palette, seeds); tweak and re-run.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Use them anywhere,
including commercially, with attribution. See [LICENSE](LICENSE).
