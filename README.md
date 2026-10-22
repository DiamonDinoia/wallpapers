# wallpapers

Generative math and physics wallpapers (3840×2160), procedurally generated with
numpy and rendered as SVG (line art) or PNG (fields), with LaTeX-style equations
typeset in Computer Modern.

Browse them at [diamondinoia.com/wallpapers](https://diamondinoia.com/wallpapers/).
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

Every plate is signed in the same Computer Modern, in whichever corner holds the
least artwork and is not already taken by the equation: the name, then the
licence under it at 0.62 of its type size.

Everything regenerates deterministically from constants at the top of the
script (papersizes, palette, seeds); tweak and re-run.

## Videos

`gen_videos.py` animates each plate into a 30 s, 1920x1080, 60 fps AV1 loop for
the Flatiron viswall, and writes the viswall captions to `videos/captions.txt`.
It composites on the plates in `raw/`, so run `gen_wallpapers.py` first, and it
needs `ffmpeg`:

```
python3 gen_videos.py [names...] [--cpu]   # av1_nvenc by default, libsvtav1 with --cpu
python3 gen_videos.py NAME --part K/N      # only frame range K of N, as NAME.partK.webm
python3 gen_videos.py --selfcheck          # assert every loop closes
```

Each run first checks that ffmpeg converts colours with the bt709 matrix the
files declare. On every push to main, CI renders the thirteen videos in parallel
with `--cpu`. It splits the two slowest into frame ranges and joins the parts by
stream copy. Then it puts the videos in the latest release with the plates, as
`<name>.webm` and `captions.txt`.

## Screensaver

`screensaver/package` is the spread plate as an endless KDE Plasma 6 wallpaper
plugin. Each drop lands at a new random place every cycle and the grid drifts,
so no pixel holds a static image. Each drop is its own quad, so a pixel shades
only the few drops that reach it. Install it, then select "Spread" as the
lock screen wallpaper (System Settings, Screen Locking, Appearance):

```
kpackagetool6 -t Plasma/Wallpaper -i screensaver/package    # -u to upgrade
```

The shaders ship baked as `.qsb`. After an edit, bake them again with the
command at the top of each `.frag`. `screensaver/check.py` renders two hours of
the plugin. It asserts that every pixel lights up, the time-mean image is flat
and the grid draws. The spread video is the positive control: its static
equation overlay fails the flatness bound.

```
uv run --with pyside6==6.11.2 --with numpy --with pillow python screensaver/check.py videos/spread.webm
```

## Debian package

`diamondinoia-wallpapers` installs the plates in `/usr/share/wallpapers/DiamonDinoia`
and the Spread wallpaper plugin. At the next login of each user, Plasma runs its
update script once: the desktop becomes a 5 min slideshow of the plates and the lock
screen shows Spread. The package does not change when the screen locks. Users can
change both afterwards. The package is in the
[DiamonDinoia/apt](https://github.com/DiamonDinoia/apt) repository:

```
sudo apt-get install diamondinoia-wallpapers
```

`deb/test.sh` passes on Debian sid (Plasma 6.7, Qt 6.11) and trixie (Plasma 6.3, Qt 6.8).
The script runs once per user, so a later version does not reset a user's choice.
Removing the package leaves each user's Plasma settings in place: a desktop that
still shows the slideshow then has no plates, and the lock screen turns black.
Plasma shows a black screen with no panel for a desktop wallpaper plugin that is
not installed, so switch any desktop that shows Spread to another wallpaper before
the removal.

`deb/build.sh` builds it from rendered plates. `deb/test.sh` installs it in a
clean `debian:sid` container, or the image in `IMAGE`, and runs Plasma and the
lock screen on Xvfb with Mesa.
It checks the default wallpaper before the install, then the slideshow and lock
screen after it, also for a user created after the install. It also checks that
the slideshow shows a plate pixel by pixel, that a user's later choice survives the
next login, that Spread renders on the desktop and the lock screen, and that both
start without Spread after a purge:

```
deb/build.sh out ~/Pictures/wallpapers/*.png && deb/test.sh out/*.deb
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Use them anywhere,
including commercially, keeping the credit each plate carries. See
[LICENSE](LICENSE).
