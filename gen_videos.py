#!/usr/bin/env python3
"""Animate the 13 wallpapers into 1920x1080 60 fps AV1 loops for the Flatiron viswall.

Reuses gen_wallpapers for palette, field math and the equation/credit overlay:
the overlay is typeset once per video and composited onto every frame.
Periods are integers in loops, so frame N-1 glides into frame 0 exactly;
`--selfcheck` asserts that for every video.

    python3 gen_videos.py [names...] [--preview] [--cpu]   # spec: 1800 frames, 1920x1080
    python3 gen_videos.py --selfcheck              # assert fn(2pi) == fn(0) per loop

Frames render on CPU with numpy in a fork pool and pipe as rgb24 rawvideo into
one av1_nvenc call per video, or libsvtav1 with --cpu on hosts without NVENC.
A cupy GPU port was A/B-measured and dropped: slower end to end (single-process
generation loses to the 20-worker pool on the chaotic-map and stamp-loop clips);
see the viswall-gpu notes in the repo's scratch dir for the numbers.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")       # BLAS inside 20 fork workers
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import math
import multiprocessing as mp
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import gen_wallpapers as gw
from gen_wallpapers import (AMBER, BG, BREEZE, CMAP, CYC, DIVERGE, MONO, TEAL,
                            VIOLET, ramp3, lerp_hex, hex_to_rgb, frame, fig4k)

W, H = 1920, 1080        # viswall output size; gen_wallpapers.py itself stays 4K
gw.W, gw.H = W, H        # so overlay placement math lands on the video frame size

FPS = 60
SECONDS = 30
NAMES = gw.NAMES
OUT = Path(__file__).parent
VID = OUT / "videos"
PREV_DIR = OUT / "previews"  # already gitignored
WORKERS = min(20, len(os.sched_getaffinity(0)))  # obey taskset: the wall shares the host

TXT = {
"wavefield": """Forty-two point sources emit circular waves, and the video shows the light intensity where they overlap. The sources hum at slightly different pitches, so the bright and dark interference fringes drift slowly across the screen. The frequencies are tuned so every wave returns to its starting phase at the end of the loop.""",
"lorenz": """The Lorenz attractor is the famous butterfly-shaped path traced by a simple model of convection in the atmosphere. Here it slowly spins like a sculpture on a turntable while a bright comet of recent history travels along the trajectory. One full rotation takes the whole video, so the loop is seamless.""",
"streamlines": """This is the flow map of a fluid stirred by six vortices superimposed on a saddle drift. Bright dashes ride along the flow like debris on a river, over a backdrop of the flow lines themselves. The dashes advance by whole spacings per loop, so the motion repeats exactly.""",
"torus": """A wireframe torus whose tube breathes with a standing-wave ripple, viewed from above its plane. The camera orbits once around it while the ripple swells and relaxes. Both motions share the loop's period, so the final frame flows back into the first.""",
"chebyshev": """Chebyshev polynomials traced in polar coordinates make roses with petals, one rose per degree one through eight. A bright highlight sweeps along every curve, riding the petals like a bead on a wire. The sweep completes whole turns, so the motion loops without a seam.""",
"dejong": """The De Jong map is a pair of trigonometric equations iterated millions of times; the points pile up into a smoky attractor colored by the direction of travel. Here its four constants glide around a tiny closed orbit, and the attractor flexes in response. The constants return to their starting values each loop.""",
"talbot": """Shine a plane wave through a comb of slits and it reprints its own pattern at regular distances downstream: the Talbot effect. The video flies through the beam, and the carpet of light scrolls past. It advances by exactly one revival period, so the pattern repeats seamlessly.""",
"spread": """Interpolating scattered data onto a grid means spreading each point's influence with a smooth kernel, shown here as nested rings around every source. The sources wander on small closed loops and their rings follow. Every loop closes exactly, so the animation never jumps.""",
"pseudospectra": """The pseudospectrum of a matrix shows how close it comes to having a given complex number as an eigenvalue, revealing how sensitive it is to perturbation. The tolerance breathes in and out, so the nested halos around the eigenvalue dots grow and shrink. The breathing is sinusoidal and loops exactly.""",
"stability": """Each colored region is the set of step sizes for which an order-p Runge-Kutta method keeps a decaying wave from blowing up. The regions pile up one order at a time over the loop, from the small first-order disk to the lobed eighth-order shape, older ones dimming so the newest reads; the full stack holds, then fades out together. The cycle is continuous, so it loops without a cut.""",
"gamma": """The gamma function extends the factorial to all complex numbers; here color encodes its angle and brightness bands its size, with poles burning bright on the left. A global phase factor slowly rotates the hue wheel one full turn. After one full turn the colors are back where they started.""",
"littlewood": """Every point is a root of a polynomial whose coefficients are all plus or minus one; together they draw a dense ring with ghostly holes. In the video one coefficient circles the unit circle instead of sitting at plus one, and the whole cloud flows in response. When it completes its circle the cloud returns to frame one.""",
"helmholtz": """A sound wave traveling rightward scatters off a rigid cylinder, computed as an exact Mie series of Hankel functions. The video runs the wave in time: crests roll in from the left, wrap the cylinder, and leave a flickering shadow behind it. The clip spans whole periods of the wave, so it loops seamlessly.""",
}

# ---------------------------------------------------------------------------
# raster kit: artwork accumulates into an (H, W, 3) float32 canvas, then goes
# to 8-bit. Stamps are anti-aliased discs; lines are resampled to ~1.5 px.
# Additive over the near-black BG reads like the SVG strokes of the plates.
# ---------------------------------------------------------------------------

def _disc(diam: int) -> np.ndarray:
    r = np.arange(diam) - (diam - 1) / 2
    rr = np.hypot(*np.meshgrid(r, r, indexing="ij")) / (diam / 2)
    return np.clip(1.5 - 1.5 * rr, 0, 1).astype(np.float32)


STAMPS = {d: _disc(d) for d in (3, 5, 7, 9, 11)}


def stamp(canvas: np.ndarray, xs: np.ndarray, ys: np.ndarray,
          color: np.ndarray, alpha, diam: int) -> None:
    """Add one soft dot per (x, y). color (3,) or (M, 3); alpha scalar or (M,)."""
    s = STAMPS[diam]
    k = diam // 2
    xi = np.round(xs).astype(np.int64)
    yi = np.round(ys).astype(np.int64)
    ok = (xi >= k) & (xi < W - k) & (yi >= k) & (yi < H - k)
    xi, yi = xi[ok], yi[ok]
    color = np.asarray(color, np.float32)
    per_pt = color.ndim == 2
    alpha = np.asarray(alpha, np.float32)
    per_a = alpha.ndim == 1
    if per_pt and len(color) == len(ok):  # caller passed colors for all points
        color = color[ok]
    if per_a and len(alpha) == len(ok):
        alpha = alpha[ok]
    for dy in range(diam):
        for dx in range(diam):
            w0 = s[dy, dx]
            if w0 <= 0:
                continue
            if per_a:
                sc = alpha * w0  # (M,)
            else:
                sc = w0 * float(alpha)  # scalar
            if per_pt:
                val = color * sc[:, None] if per_a else color * sc
            else:
                val = color[None, :] * sc[:, None] if per_a else color * sc
            np.add.at(canvas, (yi + dy - k, xi + dx - k), val)


def resample(pts: np.ndarray, step: float = 1.5) -> np.ndarray:
    """Densify a polyline so consecutive points sit about `step` px apart."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] < step:
        return pts
    tt = np.linspace(0, t[-1], min(200000, max(2, int(t[-1] / step))))
    return np.column_stack([np.interp(tt, t, pts[:, 0]), np.interp(tt, t, pts[:, 1])])


def draw_line(canvas, pts, color, alpha, diam):
    p = resample(pts)
    stamp(canvas, p[:, 0], p[:, 1], np.asarray(color, np.float32), alpha, diam)


def to_u8(canvas: np.ndarray) -> np.ndarray:
    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)


def lut_rgb(cmap, arr: np.ndarray, n: int = 1024) -> np.ndarray:
    """Fast colormap application through a uniform lookup table."""
    tab = cmap(np.linspace(0, 1, n))[:, :3].astype(np.float32)
    idx = np.clip((arr * (n - 1)).astype(np.int64), 0, n - 1)
    return tab[idx]


def canvas_bg() -> np.ndarray:
    return np.full((H, W, 3), hex_to_rgb(BG), np.float32)


# ---------------------------------------------------------------------------
# overlay: the caption is typeset once per video at EQ_SCALE_VID and composited
# onto every frame. Placement is fixed by PLACEMENT, measured on 16 frames of
# each loop (videos/proto/placement/): a corner where box+16 px margin holds
# no art in any sampled frame, else a BG band freed by cropping the art
# (aspect ratio kept), so the equation never inks over artwork.
# ---------------------------------------------------------------------------

EQ_SCALE_VID = 0.65 * gw.EQ_SCALE  # ~12 px smallest glyph at 1080p
CORNER_INSET = 48    # measured corner inset
BAND_TOP_PAD = 16    # equation margin at the top of a bottom band; eq ends by y=1000

# name -> ("corner", anchor) or ("band", side, size, crop_offset); cropped % in
# the comment is the share of lit art removed, measured over 16 sampled frames.
# Split overlays (OVERLAYS anchor = [(rows, anchor), ...]) are not listed here:
# their parts sit in corners and nothing crops.
PLACEMENT = {
"wavefield": ("band", "bottom", 247, 244),     # 22.5%
"lorenz": ("band", "left", 455, 224),          #  0.5%
"streamlines": ("band", "left", 685, 440),     # 24.9%
"torus": ("corner", "top-right"),
"dejong": ("corner", "top-right"),
"chebyshev": ("corner", "top-left"),
"talbot": ("band", "bottom", 212, 132),        # 19.2%
"spread": ("band", "bottom", 320, 196),        # 25.9%
"pseudospectra": ("band", "bottom", 280, 117), #  8.5% (plane turned 90 deg)
"stability": ("corner", "top-left"),
"littlewood": ("corner", "top-left"),
"helmholtz": ("band", "bottom", 212, 104),     # 24.5%
}

OVERLAYS = {  # name -> (lines, color, anchor, alpha[, caption scale]); lines mirror apply_overlays
"wavefield": ([r"$a(\mathbf{x}) = \sum_{j=1}^{42} "
               r"\frac{e^{i k_j |\mathbf{x}-\mathbf{x}_j|}}{\sqrt{|\mathbf{x}-\mathbf{x}_j|}}, "
               r"\quad k_j \in [18, 34], \quad I = |a|^2$"],
              "#9db3cd", None, 0.92),
"lorenz": ([r"$\dot{x}=\sigma(y-x)$", r"$\dot{y}=x(\rho-z)-y$", r"$\dot{z}=xy-\beta z$",
            r"$(\sigma,\rho,\beta)=(10,\,28,\,8/3)$",
            r"$x\!-\!z\ \mathrm{projection}$"],
           "#9db3cd", None, 0.92),
"streamlines": ([r"$\dot{x} = x - 0.3y - \sum_i \gamma_i\,(y - c_i^y)/r_i^2$",
                 r"$\dot{y} = -0.4x - 0.15y + \sum_i \gamma_i\,(x - c_i^x)/r_i^2$",
                 r"$6\ \mathrm{vortices},\ \gamma_i \in \pm[0.6, 1.4],\ \mathrm{RK2}$"],
                "#9db3cd", "top-left", 0.92),
"torus": ([r"$X=(R+r^*\cos v)\cos u$", r"$Y=(R+r^*\cos v)\sin u$", r"$Z=r^*\sin v$",
           r"$r^*(u,v) = r + 0.12\,\sin 6u\,\cos 3v$"],
          "#9db3cd", None, 0.92),
"dejong": ([r"$x_{n+1}=\sin(ay_n)-\cos(bx_n)$", r"$y_{n+1}=\sin(cx_n)-\cos(dy_n)$",
            r"$(a,b,c,d)=(1.4,\,-2.3,\,2.4,\,-2.1)$",
            r"$\mathrm{hue = flow\ direction, \ brightness = density}$"],
           "#9db3cd", None, 0.92),
"chebyshev": ([r"$T_n(\cos\theta) = \cos n\theta, \quad n = 1 \ldots 8$"],
              "#9db3cd", "bottom-left", 0.92),
"talbot": ([r"$u(x,z) = \sum_n c_n\, e^{2\pi i n x/a}\, e^{-i\pi \lambda z n^2/a^2}, "
             r"\quad z_T = 2a^2/\lambda$"], "#9db3cd", "top", 0.92),
"spread": ([r"$\varphi(z) = e^{\beta(\sqrt{1-z^2}-1)}, \quad \beta = 2.30\,w, \ w = 7$",
            r"$f_j = \sum_k c_k\, \varphi(\sigma x_k - j), \quad \sigma = 2, "
            r"\quad \varphi = 10^{-1/2} \ldots 10^{-13/2}$"],
           "#9db3cd", "top", 0.92),
"pseudospectra": ([r"$\Lambda_\varepsilon(A) = \{z : \sigma_{\min}(zI - A) < \varepsilon\}$",
                   r"$A = \mathrm{Grcar}(32), \quad \varepsilon = 10^{-9} \ldots 10^{0}$"],
                  "#9db3cd", None, 0.92),
"stability": ([r"$R_p(z) = \sum_{k=0}^{p} z^k/k!, \quad |R_p(z)| = 1, \ p = 1 \ldots 8$"],
              "#9db3cd", None, 0.92),
"gamma": ([r"$\Gamma(z) = \int_0^\infty t^{z-1} e^{-t}\,dt$",
           r"$\mathrm{hue} = \arg\Gamma, \quad \mathrm{bands} = \log_2|\Gamma|$"],
          "#9db3cd", [([0, 1], "top-left")], 0.92, 0.55 * gw.EQ_SCALE),  # no crop; user: smaller
"littlewood": ([r"$\{z \ : \ \sum_{k=0}^{20} a_k z^k = 0\}$", r"$a_k = \pm 1, \ a_{20} = 1$",
                r"$2^{20}\ \mathrm{polynomials}$", r"$2.1 \times 10^{7}\ \mathrm{roots}$"],
               "#9db3cd", None, 0.92),
"helmholtz": ([r"$u = e^{ikx} - \sum_n i^n \, [J_n(ka)/H^{(1)}_n(ka)] \, H^{(1)}_n(kr)\, e^{in\theta},"
               r" \ ka = 12$"],
              "#9db3cd", None, 0.92),
}


def premult(tile_rgba: Image.Image, alpha: float) -> Image.Image:
    """RGBA tile whose RGB equals the composited ink on BG, A = effective alpha.

    Pasting a matplotlib RGBA caption with mask alpha*a onto a dark plate gives
    out = bg*(1-a) + src*a; premultiplied with src already the antialiased ink,
    out = bg + (src - bg)*(alpha*A). We bake (src-bg)*alpha*A into RGB and keep
    A = alpha*A so composite can do out = art + RGB - bg*A... simpler: keep
    standard paste math by storing src RGB and effective alpha; composite does
    out = art*(1-A) + src*A exactly like Image.paste(..., mask).
    """
    a = tile_rgba.getchannel("A").point(lambda v: int(v * alpha))
    out = tile_rgba.copy()
    out.putalpha(a)
    return out


def _corner_xy(at: str, w: int, h: int) -> tuple[int, int]:
    v, _, hz = at.partition("-")
    y = CORNER_INSET if v == "top" else H - CORNER_INSET - h
    x = CORNER_INSET if hz == "left" else W - CORNER_INSET - w
    return x, y


SUB_BOX = (337, 1019, 1584, 1062)  # subtitle box measured on the wall


def _band_sig_xy(side: str, size: int, eq: tuple[int, int, int, int], cw: int, ch: int) -> tuple[int, int]:
    """The credit goes in the freed BG band next to the equation: below it, else to
    its right, left or above. The box, with a 16 px margin, must stay in the band
    and clear of the subtitles; the band is plain BG, so the spot is art-free."""
    M = 16
    x0, y0, x1, y1 = eq
    if side == "left":
        bx0, by0, bx1, by1 = 0, 0, size, H
    elif side == "right":
        bx0, by0, bx1, by1 = W - size, 0, W, H
    else:
        bx0, by0, bx1, by1 = 0, H - size, W, H
    cands = [((x0 + x1 - cw) // 2, y1 + M),
             (x1 + M, (y0 + y1 - ch) // 2),
             (x0 - M - cw, (y0 + y1 - ch) // 2),
             ((x0 + x1 - cw) // 2, y0 - M - ch)]
    for sx, sy in cands:
        if not (bx0 + M <= sx and sx + cw <= bx1 - M
                and by0 + M <= sy and sy + ch <= by1 - M):
            continue
        if (sx < SUB_BOX[2] + M and sx + cw > SUB_BOX[0] - M
                and sy < SUB_BOX[3] + M and sy + ch > SUB_BOX[1] - M):
            continue
        return sx, sy
    raise AssertionError("no credit spot in the band")


def part_xy(at: str, w: int, h: int) -> tuple[int, int]:
    """One equation part's corner slot at left-aligned inset, lifted clear of the
    subtitle box when a bottom part would run into it."""
    x, y = _corner_xy(at, w, h)
    if (x < SUB_BOX[2] + 16 and x + w > SUB_BOX[0] - 16
            and y < SUB_BOX[3] + 16 and y + h > SUB_BOX[1] - 16):
        y = SUB_BOX[1] - 16 - h  # lift above the subtitles instead of cropping
        assert y >= CORNER_INSET, f"part {w}x{h} at {at} cannot clear the subtitles"
    return x, y


def overlay_layers(name: str):
    """(tiles, sig_tile, sig_xy, band): equation tiles are (tile, xy), one per part.

    OVERLAYS' anchor may be a list of (rows, anchor) parts: each part is set
    left-aligned in its own corner and nothing crops. A string anchor keeps the
    old behaviour: one block in its measured art-free corner, or on the BG band
    freed by cropping the art in composite()."""
    assert (gw.RAW / f"{name}.png").exists(), f"run gen_wallpapers.py first (no raw/{name}.png)"
    lines, color, anchor, alpha = OVERLAYS[name][:4]
    sc_v = OVERLAYS[name][4] if len(OVERLAYS[name]) > 4 else EQ_SCALE_VID
    band, tiles = None, []
    if isinstance(anchor, list):
        for rows, at in anchor:
            sub = gw.caption([lines[i] for i in rows], color, W - 2 * gw.PLATE_MARGIN,
                             scale=sc_v)
            eq = gw.stack(sub, "left")  # parts share one left edge
            tiles.append((premult(eq, alpha), part_xy(at, eq.width, eq.height)))
    else:
        rows = gw.caption(lines, color, W - 2 * gw.PLATE_MARGIN, scale=sc_v)
        eq = gw.stack(rows, "center")
        pl = PLACEMENT[name]
        if pl[0] == "corner":
            xy = _corner_xy(pl[1], eq.width, eq.height)
        else:
            _, side, size, off = pl
            band = (side, size, off)
            if side == "left":
                xy = ((size - eq.width) // 2, (H - eq.height) // 2)
            elif side == "right":
                xy = (W - size + (size - eq.width) // 2, (H - eq.height) // 2)
            else:  # bottom band: the equation ends by y=1000, clear of the subtitles
                xy = ((W - eq.width) // 2, H - size + BAND_TOP_PAD)
                assert xy[1] + eq.height <= 1000, f"{name}: bottom band reaches the subtitles"
        tiles.append((premult(eq, alpha), xy))
    art = Image.open(gw.RAW / f"{name}.png").convert("RGB").resize((W, H), Image.LANCZOS)
    # the credit, like the equation, is set at EQ_SCALE_VID (0.65 x plate): gw's
    # 434 px block fits no narrow band (lorenz's is 455 px minus margins)
    sc = sc_v * gw.CREDIT_SCALE
    cr_name = gw.caption([gw.CREDIT], gw.CREDIT_COLOR, W, scale=sc)[0]
    cr_lic = gw.caption([gw.CREDIT_LICENSE], gw.CREDIT_COLOR, W, scale=sc * gw.CREDIT_LICENSE_SCALE)[0]
    cw, ch = max(cr_name.width, cr_lic.width), cr_name.height + 14 + cr_lic.height
    taken = (min(x for _, (x, y) in tiles), min(y for _, (x, y) in tiles),
             max(x + t.width for t, (x, y) in tiles), max(y + t.height for t, (x, y) in tiles))
    if isinstance(anchor, list):  # split: the credit keeps a corner of its own
        sg_at, sg_xy = "left", part_xy("bottom-left", cw, ch)
    elif band is None:
        sg_at, sg_xy = gw.sign_spot(art, taken, cw, ch)
    else:  # the corner rule lands on the moving art: use the freed band instead
        sg_at, sg_xy = "band", _band_sig_xy(band[0], band[1], taken, cw, ch)
    sig = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    if sg_at.endswith("left"):
        off_ = lambda im: 0
    elif sg_at == "band":
        off_ = lambda im: (cw - im.width) // 2
    else:
        off_ = lambda im: cw - im.width
    sig.paste(cr_name, (off_(cr_name), 0))
    sig.paste(cr_lic, (off_(cr_lic), cr_name.height + 14))
    return (tiles, premult(sig, gw.CREDIT_ALPHA), sg_xy, band)


def composite(layers, frame_u8: np.ndarray) -> np.ndarray:
    art = frame_u8.astype(np.float32)
    band = layers[3]
    if band is not None:  # crop the art (aspect kept); the freed side fills with BG
        side, size, off = band
        out = np.full((H, W, 3), np.asarray(hex_to_rgb(BG), np.float32) * 255)
        if side == "left":
            out[:, size:] = art[:, off:off + W - size]
        elif side == "right":
            out[:, :W - size] = art[:, off:off + W - size]
        else:
            out[:H - size] = art[off:off + H - size]
        art = out
    for tile, xy in layers[0] + [(layers[1], layers[2])]:
        h, w = tile.height, tile.width
        reg = art[xy[1]:xy[1] + h, xy[0]:xy[0] + w]
        t = np.asarray(tile, dtype=np.float32)
        a = t[..., 3:4] / 255.0
        art[xy[1]:xy[1] + h, xy[0]:xy[0] + w] = reg * (1 - a) + t[..., :3] * a
    return np.clip(art + 0.5, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Per-video setups. Each returns fn(phi) -> uint8 (H, W, 3) with phi in
# [0, 2pi); every period is integer in loops, so frame 1800 == frame 0.
# When gv.W/gv.H are shrunk (unit tests), setups adapt except where noted
# (pseudospectra keeps its own 1000x562 grid and upsamples).
# ---------------------------------------------------------------------------

def setup_wavefield(preview, xp=np):
    """Option A: every source orbits a small ellipse from its plate position, 1-2 whole
    turns per loop, so the interference pattern reorganises. xp = numpy or cupy."""
    rng = np.random.default_rng(7)
    y, x = np.mgrid[-0.9:0.9:complex(0, H), -1.6:1.6:complex(0, W)]
    pts = (rng.uniform(-1.3, 1.3, (42, 2)) * np.array([1.5, 0.85])).T
    ks = rng.uniform(18, 34, pts.shape[1])
    ms = rng.uniform(1, 4, pts.shape[1]).round().astype(np.float32)  # integer turns per loop
    orb = np.random.default_rng(11)  # separate stream: the plate's draws stay untouched
    turns = orb.choice([-2, -1, 1, 2], len(ks))
    ax, ay = orb.uniform(0.12, 0.30, (2, len(ks)))  # frame is 3.2 x 1.8 units
    th = orb.uniform(0, 2 * np.pi, len(ks))
    x, y = xp.asarray(x), xp.asarray(y)
    tab = xp.asarray(plt.get_cmap("magma")(np.linspace(0, 1, 1024))[:, :3], xp.float32)

    def field(phi):
        re = xp.zeros(x.shape, xp.float64)
        im = xp.zeros_like(re)
        for j, m in enumerate(ms.tolist()):
            a = turns[j] * phi + th[j]  # offset is exactly 0 at phi = 0
            sx = pts[0, j] + ax[j] * (math.cos(a) - math.cos(th[j]))
            sy = pts[1, j] + ay[j] * (math.sin(a) - math.sin(th[j]))
            r = xp.hypot(x - sx, y - sy) + 1e-3
            ph = (ks[j] * r).astype(xp.float32) - np.float32(phi) * np.float32(m)
            w = (1 / xp.sqrt(r)).astype(xp.float32)
            re += xp.cos(ph) * w
            im += xp.sin(ph) * w
        return re * re + im * im

    # fixed gain from frame 0: a per-frame max jumps when a moving source crosses a pixel
    gain = 0.99 / float(field(0.0).max())

    def fn(phi):
        amp = xp.log1p(field(phi) * gain)
        rgb = tab[xp.clip((amp * 1023).astype(xp.int32), 0, 1023)]
        out = (xp.clip(rgb, 0, 1) * 255).astype(xp.uint8)
        return out if xp is np else out.get()
    fn.gpu = xp is not np  # CUDA frames cannot fork: encode streams them from the parent
    return fn


def wrap_gauss(d: np.ndarray, sigma: float) -> np.ndarray:
    """Periodic gaussian bump of points at cyclic distance d in [0, 0.5]."""
    return np.exp(-np.minimum(d, 1 - d) ** 2 / (2 * sigma * sigma))


def setup_lorenz(preview):
    dt, n = 0.0035, 60000
    p = np.array([0.1, 0.0, 20.0])
    traj = np.empty((n, 3))

    def f(s):
        x, y_, z = s
        return np.array([10 * (y_ - x), x * (28 - z) - y_, x * y_ - 8 / 3 * z])
    for i in range(n):
        p += dt * f(p + dt / 2 * f(p))
        traj[i] = p
    traj = traj[4000:]
    xz = traj[:, [0, 2]]
    lo, hi = xz.min(0), xz.max(0)
    pad = gw.PLATE_MARGIN + 24
    s = (H - 2 * pad) / (hi[1] - lo[1])
    x0 = (lo[0] + hi[0]) / 2
    sx_r = (H - 2 * pad) / (hi[1] - lo[1])
    # widest rotated x-extent decides the horizontal scale, so nothing clips
    xr = np.abs(np.hypot(traj[:, 0] - x0, traj[:, 1]))
    sx = min(1.88 * sx_r, (W / 2 - pad) / max(xr.max(), 1e-9))
    assert pad % 2 == 0
    seg_t = np.linspace(0, 1, len(traj))
    cols = np.array([hex_to_rgb(lerp_hex(BREEZE, VIOLET, t)) for t in seg_t], np.float32)

    def fn(phi):
        ca, sa = np.cos(phi), np.sin(phi)
        xr_ = traj[:, 0] * ca - traj[:, 1] * sa
        px = W / 2 + (xr_ - x0) * sx
        py = H - pad + (lo[1] - traj[:, 2]) * s
        d = (seg_t % 1.0 - phi / (2 * np.pi)) % 1.0     # comet head, one turn per loop
        boost = 1 + 5.0 * wrap_gauss(d, 0.006)
        canvas = canvas_bg()
        stamp(canvas, px, py, cols, 0.07, 9)            # glow pass
        stamp(canvas, px, py, cols * boost[:, None], 1.0, 5)
        return to_u8(canvas)
    return fn


def setup_streamlines(preview):
    rng = np.random.default_rng(11)
    vort = rng.normal(0, 0.55, (6, 2))
    strength = rng.uniform(0.6, 1.4, 6) * rng.choice([-1, 1], 6)

    def field(s):
        x, y_ = s[..., 0], s[..., 1]
        vx = x - 2 * y_ * 0.15
        vy = -0.15 * y_ - 0.4 * x
        for (cx, cy), g in zip(vort, strength):
            dx, dy = x - cx, y_ - cy
            r2 = dx**2 + dy**2 + 1e-3
            vx += -g * dy / r2
            vy += g * dx / r2
        return np.stack([vx, vy], axis=-1)

    seeds = rng.uniform(-1.7, 1.7, (120, 2))
    paths, cols = [], []
    dt, n_step = 0.004, 800
    for pi, s0 in enumerate(seeds):
        s = s0.copy()
        path = np.empty((n_step, 2))
        for i in range(n_step):
            s += dt * field(s + dt / 2 * field(s))
            path[i] = s
            if np.hypot(*s) > 2.2:
                path = path[: i + 1]
                break
        if len(path) < 30:
            continue
        px = np.column_stack([W / 2 + path[:, 0] * W / 3.4, H / 2 - path[:, 1] * H / 1.9])
        c = lerp_hex(TEAL, BREEZE, pi / 120) if pi % 5 else AMBER
        paths.append(px)
        cols.append(np.asarray(hex_to_rgb(c), np.float32))
    # comets ride each line at the flow's own speed: the cell coordinate runs in
    # RK2 steps (flow time), not arclength, so they race round vortex cores and
    # crawl through the saddle. CELLS integer cells pass per loop -> exact seam.
    step_px = np.median(np.concatenate([np.hypot(*np.diff(px, axis=0).T) for px in paths]))
    cell = 230.0 * W / 1920 / step_px           # RK2 steps per comet spacing (~230 px median)
    CELLS, TAIL = 11, 0.22                      # ~85 px/s median at 1080p; tail as cell fraction
    backdrop = canvas_bg()
    pre = []
    for px, c in zip(paths, cols):
        p = resample(px, 1.5)
        s_raw = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(px, axis=0).T))])
        s_ar = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
        u = np.interp(s_ar, s_raw, np.arange(len(px))) / cell
        stamp(backdrop, p[:, 0], p[:, 1], c, 0.16, 3)   # static lines, wallpaper brightness
        pre.append((p, c, u))

    def fn(phi):
        canvas = backdrop.copy()
        for p, c, u in pre:
            d = (CELLS * phi / (2 * np.pi) - u + 1e-9) % 1.0   # 0 at a head, grows behind it; +1e-9: 11*(2pi)/(2pi) lands 1 ulp under 11, which would park the head on the (d < TAIL) cutoff at phi=2pi only
            w_ = np.exp(-d / (TAIL / 3)) * (d < TAIL)
            on = w_ > 0.02
            if on.any():
                stamp(canvas, p[on, 0], p[on, 1], np.minimum(c * 1.7, 1), w_[on] * 0.6, 5)
        return to_u8(canvas)
    return fn


def setup_torus(preview):
    u = np.linspace(0, 2 * np.pi, 160)
    v = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    R, r = 2.6, 1.0
    U, V = np.meshgrid(u, v, indexing="ij")
    colb, cola = np.asarray(hex_to_rgb(BREEZE), np.float32), np.asarray(hex_to_rgb(AMBER), np.float32)
    S = W / 3840.0  # torus geometry was tuned in 4K px; scale its origin shift and size

    def fn(phi):
        amp = 0.65 + 0.35 * np.cos(phi)  # 1 at phi=0, like the static plate
        rr = r + 0.12 * amp * np.sin(6 * U) * np.cos(3 * V)
        X = (R + rr * np.cos(V)) * np.cos(U)
        Y = (R + rr * np.cos(V)) * np.sin(U)
        Z = rr * np.sin(V)
        Xr = X * np.cos(phi) - Y * np.sin(phi)   # camera orbit about the z axis
        Yr = X * np.sin(phi) + Y * np.cos(phi)
        Yv = 0.45 * Z + 0.89 * Yr - 0.119
        D = 0.08 * Z
        canvas = canvas_bg()
        for j in range(len(v)):
            pts = np.column_stack([W / 2 - 420 * S + Xr[:, j] * (297 * S), H / 2 - Yv[:, j] * (297 * S)])
            a = 0.15 + 0.65 * (D[:, j].mean() + 1) / 2
            draw_line(canvas, pts, colb, a, 3)
        for i in range(0, len(u), 4):
            pts = np.column_stack([W / 2 - 420 * S + Xr[i, :] * (297 * S), H / 2 - Yv[i, :] * (297 * S)])
            a = 0.15 + 0.65 * (D[i, :].mean() + 1) / 2
            draw_line(canvas, pts, cola, a, 3)
        return to_u8(canvas)
    return fn


def setup_chebyshev(preview):
    """Roses at their wallpaper positions; a highlight orbits each once per loop.

    The spec row asks each rose to rotate by one petal per loop. At 60 fps that
    turn is sub-pixel except at phi=2pi where cos(n 2pi/n) re-phases the radius
    by up to 2 px — a frame(N-1) -> frame(0) pop of 161/255. Roses therefore
    stay fixed (last-frame motion is continuous into frame 0) and the sweep
    carries the motion; the pixel-exact seam is asserted in selfcheck.
    """
    n_max = 8
    cx, cy, R = W / 2, H / 2, 940.0 * H / 2160

    def ramp4(t):
        if t < 1 / 3: return lerp_hex(TEAL, BREEZE, 3 * t)
        if t < 2 / 3: return lerp_hex(BREEZE, VIOLET, 3 * t - 1)
        return lerp_hex(VIOLET, AMBER, 3 * t - 2)

    roses = []
    for n in range(1, n_max + 1):
        th = np.linspace(0, 2 * np.pi, 4 * n * 180 + 1)
        col = np.asarray(hex_to_rgb(ramp4((n - 1) / (n_max - 1))), np.float32)
        roses.append((n, th, col))
    circ = 2 * np.pi * np.arange(721) / 720
    ring = np.column_stack([cx + R * np.cos(circ), cy + R * np.sin(circ)])
    pre = []
    for n, th, col in roses:
        r = np.cos(n * th)
        pts = np.column_stack([cx + R * r * np.cos(th), cy + R * r * np.sin(th)])
        p = resample(pts, 1.5)
        seg = np.hypot(*np.diff(p, axis=0).T)
        s_ar = np.concatenate([[0.0], np.cumsum(seg)])
        # integer number of sweeps along the arclength -> invisible phase wrap
        pre.append((p, col, s_ar / s_ar[-1] * 8.0))

    def fn(phi):
        canvas = canvas_bg()
        draw_line(canvas, ring, hex_to_rgb("#39414f"), 0.8, 3)
        stamp(canvas, [cx], [cy], np.asarray(hex_to_rgb(AMBER), np.float32), 1.0, 7)
        for n, th, col in roses:
            # bloom: r = (a + cos n th)/(1 + a); a = 0 at phi = 0 and 2pi, n blooms per loop
            a = 0.55 * np.sin(n * phi / 2) ** 2
            r = (a + np.cos(n * th)) / (1 + a)
            p = resample(np.column_stack([cx + R * r * np.cos(th), cy + R * r * np.sin(th)]), 1.5)
            s_ar = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
            u = s_ar / s_ar[-1] * 8.0
            stamp(canvas, p[:, 0], p[:, 1], col, 0.22, 7)
            dd = (u + phi / (2 * np.pi)) % 1.0  # highlight sweeps once per loop
            boost = 0.35 + 0.9 * np.maximum(np.sin(np.pi * dd), 0) ** 2
            stamp(canvas, p[:, 0], p[:, 1], (col * boost[:, None]).astype(np.float32), 0.95, 3)
        return to_u8(canvas)
    return fn


def setup_dejong(preview):
    rng = np.random.default_rng(3)
    n_walkers, n_steps = 4000, 4500
    x0 = rng.uniform(-2, 2, n_walkers).astype(np.float32)
    y0 = rng.uniform(-2, 2, n_walkers).astype(np.float32)
    hue_tab = CYC(np.linspace(0, 1, 1024))[:, :3].astype(np.float32)

    def fn(phi):
        sa, sb = 0.22 * np.sin(phi), 0.22 * np.cos(phi)
        a, b = 1.4 + sa, -2.3 + sb
        c, d_ = 2.4 - sb, -2.1 - sa  # closed ellipse, exact return at phi=2pi
        x, y = x0.copy(), y0.copy()
        dens = np.zeros((H, W), np.float32)
        sx_, sy_ = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        for _ in range(n_steps):
            xn = np.sin(a * y) - np.cos(b * x)
            yn = np.sin(c * x) - np.cos(d_ * y)
            iy = np.clip(((yn + 2.3) / 4.6 * (H - 1)).astype(np.int32), 0, H - 1)
            ix = np.clip(((xn + 2.3) / 4.6 * (H - 1) + (W - H) // 2 - 500 * W // 3840).astype(np.int32), 0, W - 1)  # 500 was a 4K px shift
            np.add.at(dens, (iy, ix), 1.0)
            np.add.at(sx_, (iy, ix), xn - x)
            np.add.at(sy_, (iy, ix), yn - y)
            x, y = xn, yn
        hue_idx = np.clip((((np.arctan2(sy_, sx_) + np.pi) / (2 * np.pi)) * 1023).astype(np.int64), 0, 1023)
        val = np.log1p(dens)
        val /= val.max()
        rgb = hue_tab[hue_idx] * (0.95 * val ** 0.55)[..., None]
        rgb[dens < 1] = hex_to_rgb(BG)
        return to_u8(rgb)
    return fn


def setup_talbot(preview):
    """Talbot carpet scroll via parity splitting: u(z+d) = c(F+G) + i s(F-G) - 2i s conj(H2).

    F/G = even/odd-mode parts at the displayed rows, H2 = sum e^{-4 pi i n^2 z} A_n.
    The scroll factor e^{-2 pi i n^2 dz} = cos(d)' - i (-1)^n sin(d)' is exact and
    1-periodic in dz, so the carpet scrolls one full Talbot period per loop.
    The slab covers rows 0.02..0.28+ like the static plate, upsampled to H rows.
    """
    n = np.arange(-100, 101)
    c = np.where(n == 0, 0.5, np.sinc(n / 2) / 2)
    x = np.linspace(-0.9, 0.9, W)
    slab = 0.26                      # zeta range shown at any instant
    R = 1850                         # rows per zeta: 0.26*R = 481 and 1.0*R = R, both exact
    n_sl = 481                       # slab rows: 3840/480 ~ 2160/270 px cells per period
    # scroll window: rows cover zeta in [0.02, 0.28] like the static plate; the
    # window drops one full Talbot revival (exactly R rows) per loop, so its wrap
    # is invisible (|u|^2 is 1-periodic in zeta); tables span 0.02..0.02+1.0+slab.
    zr = 0.02 + np.arange(R + n_sl) / R
    q = 2 * np.pi
    even = (n % 2 == 0).astype(np.float64)
    phase = np.exp(2j * np.pi * np.outer(zr, n.astype(np.float64) ** 2))  # (rows, N)
    modes = c * np.exp(2j * np.pi * np.outer(x, n))                       # (W, N)
    half = np.exp(4j * np.pi * np.outer(zr, n.astype(np.float64) ** 2))
    # Et*half would rebuild the even block; instead reuse the unmasked phase for
    # the second parity table directly.
    F = modes @ np.ascontiguousarray((phase * even).T)   # (W, rows) even block, all rows
    G = modes @ np.ascontiguousarray((phase * (1 - even)).T)
    H2 = modes @ np.ascontiguousarray(half.T)            # sum_n e^{-4pi i n^2 zeta} A_n

    n_frames_ref = 1800  # full-run frame count; preview runs scale the step the same way

    def fn(phi):
        dzc = phi / (2 * np.pi)  # one full Talbot period of scroll per loop
        u = np.cos(2 * np.pi * dzc) * (F + G) + 1j * np.sin(2 * np.pi * dzc) * (F - G) - \
            2j * np.sin(2 * np.pi * dzc) * np.conj(H2)
        i0 = int(round(dzc * R))  # 1.0 zeta window drop per loop; wrap is |u|^2-periodic
        slab_rows = u[:, i0:i0 + n_sl]         # (W, n_sl)
        amp = np.abs(slab_rows) ** 2
        amp = np.clip(amp / np.percentile(amp, 99.9), 0, 1) ** 1.8
        tile = lut_rgb(MONO, amp.T)            # (n_sl, W, 3)
        im = Image.fromarray(to_u8(tile)).resize((W, H), Image.Resampling.BILINEAR)
        return np.asarray(im).copy()
    return fn


def setup_spread(preview):
    m, w_ = 38, 7
    rng = np.random.default_rng(23)
    R = 300.0 * (W / 3840)  # the static's R is a 4K px radius; scale like torus/dejong
    step = H // 9
    grid_cols = [np.column_stack([[gx, 0], [gx, H]]) for gx in range(0, W, step)]
    grid_rows = [np.column_stack([[0, gy], [W, gy]]) for gy in range(0, H, step)]
    th = np.linspace(0, 2 * np.pi, 361)
    beta = 2.30 * w_
    dec = np.arange(0.5, 7.0, 0.5)
    dec = dec[dec * np.log(10) < beta]
    rad = R * np.sqrt(1 - (1 - dec * np.log(10) / beta) ** 2)
    sources = []
    for k in range(m):
        sx, sy = rng.uniform(-R, W + R), rng.uniform(-R, H + R)
        col = np.asarray(hex_to_rgb(ramp3(TEAL, BREEZE, VIOLET, rng.random())), np.float32)
        kk = int(rng.integers(1, 4))
        rx = rng.uniform(15, 45)
        ry = rng.uniform(15, 45)
        dir_ = rng.choice([-1, 1])
        ph0 = rng.uniform(0, 2 * np.pi)
        sources.append((sx, sy, col, kk, rx, ry, dir_, ph0))
    gridc = np.asarray(hex_to_rgb("#1a2534"), np.float32)
    amb = np.asarray(hex_to_rgb(AMBER), np.float32)

    def fn(phi):
        canvas = canvas_bg()
        for g in grid_cols + grid_rows:
            stamp(canvas, g[:, 0], g[:, 1], gridc, 1.0, 3)
        for sx, sy, col, kk, rx, ry, dir_, ph0 in sources:
            x = sx + rx * np.cos(kk * phi * dir_ + ph0)
            y = sy + ry * np.sin(kk * phi * dir_ + ph0)
            for li, rrr in enumerate(rad):
                pts = np.column_stack([x + rrr * np.cos(th), y + rrr * np.sin(th)])
                draw_line(canvas, pts, col, 0.85 - 0.05 * li, 3)
            stamp(canvas, [x], [y], amb, 1.0, 7)
        return to_u8(canvas)
    return fn


def _polar_to_px(fig, cx0, cy0, r, th):
    """Polar coords about data point (cx0, cy0) -> pixel coords from top-left (mpl y flips)."""
    ax = fig.axes[0]
    xy = ax.transData.transform(np.column_stack([cx0 + r * np.cos(th), cy0 + r * np.sin(th)]))
    xy[:, 1] = fig.bbox.height - xy[:, 1]
    return xy[:, 0], xy[:, 1]


def setup_pseudospectra(preview):
    a = gw._grcar(32)
    half = 5.9
    lim = (-half, half, -half * H / W, half * H / W)  # screen (X, Y), Y up
    nx, ny = 1000, int(1000 * H / W)  # wallpaper's own sampling; frames upsample for display
    sx = np.linspace(lim[0], lim[1], nx)
    sy = np.linspace(lim[3], lim[2], ny)  # row 0 is the top of the frame
    # the plane turns 90 deg CCW against the wallpaper: X = -Im z, Y = Re z - 0.9
    z = (0.9 + sy[:, None]) - 1j * sx[None, :]
    eye = np.eye(a.shape[0])
    out = np.empty(z.size)
    flat = z.ravel()
    for i in range(0, flat.size, 8192):
        blk = flat[i:i + 8192, None, None] * eye - a
        out[i:i + 8192] = np.linalg.svd(blk, compute_uv=False)[:, -1]
    smin = out.reshape(z.shape)
    ev = np.linalg.eigvals(a)
    levels = np.linspace(0.0, 9.6, 33)  # v spans ~0.07..10.8; static plate clips at 0..9 (gen_wallpapers.py:665)
    ctab = np.asarray(CMAP(np.linspace(0, 1, len(levels) - 1))[:, :3], np.float32)
    # eigenvalue dots land at the same pixels the contour plot used
    _fig, _ax = fig4k(lim)
    ex, ey = _ax.transData.transform(np.column_stack([-ev.imag, ev.real - 0.9])).T
    ey = _fig.bbox.height - ey
    ex *= W / _fig.bbox.width  # fig4k tracks the patched gw.W/gw.H; guard any drift
    ey *= H / _fig.bbox.height
    plt.close(_fig)
    amps = (0.0168, 0.168)
    canvas0 = np.full((ny, nx, 3), hex_to_rgb(BG), np.float32)  # pre-canvas template

    def fn(phi):
        amp = math.sqrt(amps[0] * amps[1]) * (10 ** (0.5 * np.sin(phi)))
        v = -np.log10(amp * smin)
        canvas = canvas0.copy()
        for li in range(len(levels) - 2, -1, -1):  # outermost first
            c = ctab[li]
            inside = (v >= levels[li]) & (v < levels[li + 1])
            canvas[inside] += 0.55 * c
        canvas[v >= levels[-1]] += 0.55 * ctab[-1]  # extend="both" caps the top band
        small = Image.fromarray(to_u8(canvas)).resize((W, H), Image.Resampling.BILINEAR)
        full = np.asarray(small).astype(np.float32)
        stamp(full, ex, ey, np.asarray(hex_to_rgb("#fff2d6"), np.float32), 1.0, 11)
        return np.clip(full, 0, 255).astype(np.uint8)
    return fn


def setup_stability(preview):
    p_max = 8
    lim = tuple(np.add(frame(8.2), (-2.0, -2.0, 0, 0)))
    zr = np.linspace(lim[0], lim[1], W)
    zi = np.linspace(lim[2], lim[3], H)
    z = zr[None, :] + 1j * zi[:, None]
    absc = np.empty((p_max, *z.shape))
    colr = []
    for p in range(1, p_max + 1):
        r = sum(z**k / math.factorial(k) for k in range(p + 1))
        absc[p - 1] = np.abs(r)
        t = (p - 1) / (p_max - 1)
        colr.append(ramp3(TEAL, BREEZE, VIOLET, 1.5 * t) if t < 2 / 3
                    else lerp_hex(VIOLET, AMBER, 3 * t - 2))

    from scipy import ndimage
    colsf = np.array([hex_to_rgb(c) for c in colr], np.float32)
    s_max = SECONDS * FPS  # for reference only

    def alphas(bar):
        """(line, fill) visibility per p at progress bar in [0, 1): the orders
        pile up on 2 s stages (1 s fade-in each) through 15 s, the full stack
        holds to 27 s, then the whole stack fades out over the last ~3 s. The
        fade lands on the empty frame-0 state 0.1 s early, so frame 1799 and
        frame 0 are identical and the loop closes without a jump."""
        t = bar * SECONDS
        g = float(np.clip((29.9 - t) / 2.9, 0, 1))  # stack-wide fade-out
        out = []
        for p in range(1, p_max + 1):
            a = float(np.clip(t - (p - 1) * 2.0, 0, 1))  # 1 s fade-in per order
            if p < p_max:  # earlier orders dim to 0.8 while the next one fades in
                a *= 1.0 - 0.2 * float(np.clip(t - p * 2.0, 0, 1))
            line = a * g
            fill = float(np.clip(a * 1.4 - 0.15, 0, 1)) * g
            out.append((line, fill))
        return out

    ins_bnd = []
    for p in range(1, p_max + 1):
        inside = absc[p - 1] < 1.0
        bnd = ndimage.binary_dilation(inside) ^ inside
        ins_bnd.append((inside, bnd))  # static per order; frame loop only scales them

    def fn(phi):
        bar = (phi / (2 * np.pi)) % 1.0
        canvas = canvas_bg()
        row = np.concatenate(([0.0], np.full(H - 2, 1.5), [0.0]))
        colv = np.concatenate(([0.0], np.full(W - 2, 1.5), [0.0]))
        axisc = np.asarray(hex_to_rgb("#39414f"), np.float32)
        iy0 = round((0 - lim[2]) / (lim[3] - lim[2]) * (H - 1))
        ix0 = round((0 - lim[0]) / (lim[1] - lim[0]) * (W - 1))
        vis = alphas(bar)
        for p in range(1, p_max + 1):
            line, fill = vis[p - 1]
            if line <= 0:
                continue
            inside, bnd = ins_bnd[p - 1]
            if fill > 0:
                canvas[inside] += 0.16 * fill * colsf[p - 1]
            canvas[bnd] += line * colsf[p - 1]
        canvas[iy0, :] += 1.5 / 255 * 40 * axisc / axisc.max()  # axes hairlines, alpha ~0.4
        canvas[:, ix0] += 1.5 / 255 * 40 * axisc / axisc.max()
        return to_u8(canvas)
    return fn


def setup_gamma(preview):
    """Hue LUT turns once per loop; the log2|Gamma| bands stream out of the poles.

    frac(log2|G| - K phi/2pi) with integer K returns exactly at phi = 2pi.
    """
    from scipy.special import gamma as gfun
    lim = frame(6.4)
    x = np.linspace(lim[0] - 1.6, lim[1] - 1.6, W)
    y = np.linspace(lim[2], lim[3], H)
    with np.errstate(over="ignore", invalid="ignore"):
        g = gfun(x[None, :] + 1j * y[:, None])
    g = np.nan_to_num(g, nan=0.0, posinf=1e300, neginf=-1e300)
    mag = np.abs(g)
    hue = (np.angle(g) + np.pi) / (2 * np.pi)
    band = np.log2(mag + 1e-300)
    env = 1 / (1 + np.exp(-0.55 * band)) / (1 + np.exp((band - 7.0) / 2.0))
    env = (env ** 0.9).astype(np.float32)          # fixed falloff: poles stay bright
    bf = (band - np.floor(band)).astype(np.float32)
    base = (np.rint(hue * 1024).astype(np.int64)) & 1023   # integer LUT position
    tab = CYC(np.linspace(0, 1, 1024))[:, :3].astype(np.float32)  # cyclic, 1024 steps
    K = 6  # bands per loop: a new ring leaves every pole each 5 s

    def fn(phi):
        shift = int(round(phi / (2 * np.pi) * 1024))
        f = (bf + np.float32(K * phi / (2 * np.pi))) % np.float32(1)
        vv = (0.35 + 0.65 * f) ** np.float32(0.9) * env
        rgb = tab[(base + shift) & 1023] * vv[..., None]
        return to_u8(rgb)
    return fn


def setup_littlewood(preview):
    """a_6 = e^{i phi} turns one circle; roots cycle back each loop.

    Degree 16 against the static's degree 20: with the 2x2 bilinear splat below,
    the ring's median luma matches the static's (111 vs 113.5 at 1080p); 2^16
    polys/frame is still feasible at 60 fps. Degree 13 left the ring at luma
    13.8, equal to the background.
    """
    deg = 16  # a_k for k=0..15 free, a_16 = 1: companion is 16x16
    lim = frame(2.75)  # the static's frame: the whole ring inside the plate
    n_poly = 1 << deg
    bits = 1 << np.arange(deg)
    comp = np.zeros((deg, deg), dtype=np.complex128)
    comp[1:, :-1] = np.eye(deg - 1)
    signs = np.where((np.arange(n_poly)[:, None] & bits) > 0, 1.0, -1.0)
    jcoef = 6  # the coefficient that turns

    def fn(phi):
        rot = np.exp(1j * phi)
        co = signs.astype(np.complex128)
        co[:, jcoef] = rot
        dens = np.zeros((H, W), np.float32)
        for start in range(0, n_poly, 8192):
            idxs = slice(start, min(start + 8192, n_poly))
            m = np.repeat(comp[None], idxs.stop - idxs.start, 0)
            m[:, 0, :] = -co[idxs]
            r = np.linalg.eigvals(m).ravel()
            for halfs in (1, -1):
                rr = r * halfs
                fx = (rr.real - lim[0]) / (lim[1] - lim[0]) * (W - 1)
                fy = (rr.imag - lim[2]) / (lim[3] - lim[2]) * (H - 1)
                ix = np.floor(fx).astype(np.int32)
                iy = np.floor(fy).astype(np.int32)
                wx = (fx - ix).astype(np.float32)
                wy = (fy - iy).astype(np.float32)
                w0 = 1.0 / (1 + (halfs < 0) * 3)
                for dx, ax in ((0, 1 - wx), (1, wx)):  # 2x2 bilinear splat
                    for dy, ay in ((0, 1 - wy), (1, wy)):
                        ok = (ix + dx >= 0) & (ix + dx < W) & (iy + dy >= 0) & (iy + dy < H)
                        np.add.at(dens, ((iy + dy)[ok], (ix + dx)[ok]), (w0 * ax * ay)[ok])
        v = np.log1p(dens)
        v = (v / v.max()) ** 0.62
        return to_u8(lut_rgb(CMAP, v))
    return fn


def setup_helmholtz(preview):
    from scipy.special import hankel1, jv
    ka = 12.0
    lim = frame(6.4)
    x = np.linspace(lim[0], lim[1], W)
    y = np.linspace(lim[2], lim[3], H)
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    rs = np.linspace(1.0, r.max() + 0.01, 6000)
    u = np.exp(1j * ka * X).astype(np.complex128)
    for n in range(-40, 41):
        cn = -(1j**n) * jv(n, ka) / hankel1(n, ka)
        hs = hankel1(n, ka * rs)
        h = np.interp(r, rs, hs.real) + 1j * np.interp(r, rs, hs.imag)
        u += cn * h * np.exp(1j * n * th)
    mask = r < 1.0

    def fn(phi):
        f = np.real(u * np.exp(-1j * phi))
        f[mask] = 0.0
        f = np.clip(f / 1.9, -1, 1)
        f = np.sign(f) * np.abs(f) ** 1.4
        return to_u8(lut_rgb(DIVERGE, 0.5 + 0.5 * f))
    return fn


SETUPS = {
    "wavefield": setup_wavefield, "lorenz": setup_lorenz, "streamlines": setup_streamlines,
    "torus": setup_torus, "chebyshev": setup_chebyshev, "dejong": setup_dejong,
    "talbot": setup_talbot, "spread": setup_spread, "pseudospectra": setup_pseudospectra,
    "stability": setup_stability, "gamma": setup_gamma, "littlewood": setup_littlewood,
    "helmholtz": setup_helmholtz,
}

# ---------------------------------------------------------------------------
# driver: frames -> ffmpeg stdin (one av1_nvenc call), one video at a time
# ---------------------------------------------------------------------------

_FN = None
_LAYERS = None
_NFRAMES = 0
_PREVIEW = False


def _init_worker(name, nframes, preview):
    """One setup per worker process: fork copies _FN set by encode beforehand."""
    os.nice(19)
    global _NFRAMES, _PREVIEW
    _NFRAMES, _PREVIEW = nframes, preview


def _render_one(i: int) -> bytes:
    phi = (i / _NFRAMES) * 2 * np.pi
    arr = composite(_LAYERS, _FN(phi))
    if _PREVIEW:
        arr = np.asarray(Image.fromarray(arr).resize((960, 540), Image.LANCZOS))
    return arr.tobytes()


# libsvtav1 serves hosts without NVENC (CI runners). crf 26 on 120 dejong frames
# (ffmpeg 6.1, SVT-AV1 1.7.0): 35.29 dB RGB PSNR vs av1_nvenc's 35.10 dB.
CODEC = {
    "av1_nvenc": ["-c:v", "av1_nvenc", "-preset", "p7", "-tune", "hq",
                  "-rc", "vbr", "-cq", "30", "-b:v", "0", "-multipass", "fullres"],
    "libsvtav1": ["-c:v", "libsvtav1", "-preset", "6", "-crf", "26"],
}


def ffmpeg_cmd(w: int, h: int, out: Path, codec: str) -> list:
    """rgb24 frames on stdin -> one AV1 webm; one call, no passlog, no second read."""
    return ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-an",
            # yuv420p so players get the 8-bit profile they render correctly:
            # rgb-profile AV1 shows up black or range-misread in several players.
            # The explicit bt709 matrix matches the tags below: ffmpeg 6.1's plain
            # format=yuv420p shifted R by -0.6 and lost 0.76 dB RGB PSNR on dejong (crf 26).
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", "-color_range", "tv",
            *CODEC[codec], str(out)]


def colorcheck(codec: str) -> None:
    """A flat red must encode to bt709 limited-range Y'CbCr, as the stream tags declare.

    ffmpeg 6.1's plain format=yuv420p used bt601 here; the bt601 values of this red
    sit outside the tolerance, so the check fails on that bug (control asserted)."""
    rgb, w, h, tol = (200, 40, 40), 320, 180, 2

    def ycbcr(kr, kb):
        r, g, b = (c / 255 for c in rgb)
        y = kr * r + (1 - kr - kb) * g + kb * b
        return np.array([16 + 219 * y, 128 + 224 * (b - y) / (2 - 2 * kb), 128 + 224 * (r - y) / (2 - 2 * kr)])

    want, bt601 = ycbcr(0.2126, 0.0722), ycbcr(0.299, 0.114)
    assert np.abs(bt601 - want).max() > 2 * tol, "this colour cannot tell bt601 from bt709"
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "c.webm"
        frames = np.tile(np.array(rgb, np.uint8), (8, h, w, 1)).tobytes()
        subprocess.run(ffmpeg_cmd(w, h, out, codec), input=frames, check=True)
        tags = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "default=nw=1",
                               "-show_entries", "stream=color_space,color_range", str(out)],
                              capture_output=True, text=True, check=True).stdout.split()
        assert sorted(tags) == ["color_range=tv", "color_space=bt709"], f"{codec}: tags {tags}"
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                             capture_output=True, check=True).stdout
    p = np.frombuffer(raw, np.uint8).astype(float)
    got = np.array([p[:w * h].mean(), p[w * h:w * h * 5 // 4].mean(), p[w * h * 5 // 4:].mean()])
    assert np.abs(got - want).max() <= tol, f"{codec}: Y'CbCr {got.round(1)}, bt709 wants {want.round(1)}"
    print(f"colorcheck {codec}: Y'CbCr {got.round(1)} within {tol} of bt709 {want.round(1)}, "
          f"tags bt709/tv; bt601 {bt601.round(1)} would fail", flush=True)


def encode(name: str, fn, layers, preview: bool, codec: str = "av1_nvenc") -> Path:
    w, h = (960, 540) if preview else (W, H)
    nframes = FPS * (5 if preview else SECONDS)
    global _FN, _LAYERS
    _FN, _LAYERS = fn, layers  # set before Pool(): fork workers inherit them
    log = tempfile.NamedTemporaryFile(prefix=f"viswall-{name}-", suffix=".log", delete=False)
    dest = (PREV_DIR if preview else VID)
    dest.mkdir(exist_ok=True)
    out = dest / f"{name}.webm"
    cmd = ffmpeg_cmd(w, h, out, codec)
    t0 = time.time()
    if getattr(fn, "gpu", False):
        # cupy frames come from the parent: a CUDA context cannot cross fork()
        with open(log.name, "ab") as lf:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=lf)
            for i in range(nframes):
                arr = composite(layers, fn((i / nframes) * 2 * np.pi))
                if preview:
                    arr = np.asarray(Image.fromarray(arr).resize((960, 540), Image.LANCZOS))
                proc.stdin.write(arr.tobytes())
            proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"{name}: ffmpeg rc={rc}; see {log.name}")
        print(f"{name}: {time.time() - t0:.0f}s -> {out} (GPU frames, log {log.name})",
              flush=True)
        return out
    # fork so workers inherit _FN/_LAYERS (closures are unpicklable); 3.14's
    # default start method is no longer fork.
    pool = mp.get_context("fork").Pool(WORKERS, initializer=_init_worker,
                                       initargs=(name, nframes, preview))
    try:
        with open(log.name, "ab") as lf:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=lf)
            try:
                for chunk in pool.imap(_render_one, range(nframes), chunksize=1):
                    proc.stdin.write(chunk)
                proc.stdin.close()
            except BrokenPipeError:
                pass
            except KeyboardInterrupt:  # let Ctrl-C break the wall too
                proc.kill()
                pool.terminate()
                raise
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"{name}: ffmpeg rc={rc}; see {log.name}")
    finally:
        pool.close()
        pool.join()
    print(f"{name}: {time.time() - t0:.0f}s -> {out} (log {log.name})", flush=True)
    return out


def build(name: str, preview: bool):
    if name == "wavefield":
        try:
            import cupy
            fn = setup_wavefield(preview, xp=cupy)
            print("wavefield: frames on GPU via cupy", flush=True)
        except Exception as e:
            print(f"wavefield: cupy unusable ({e}); numpy fallback", flush=True)
            fn = setup_wavefield(preview)
    else:
        fn = SETUPS[name](preview)
    layers = overlay_layers(name)
    return fn, layers


def _seam_one(name):
    """fn(2pi) vs fn(0): integer-period state must return exactly (modulo AA speckle).

    wavefield/fluid-style fields carry a phase of magnitude m_j * 2pi in fp32;
    cos of a large argument jitters by ~1e-4 of fringe phase, plus or minus.
    The seam is exact at float32's quantization of kr +- 2pi*m_j; the tiny
    residual is encoded identically on both sides by AV1 (identical u8 frames).
    """
    fn = SETUPS[name](False)
    a = fn(0.0)
    b = fn(2 * np.pi)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    dmax, frac = int(d.max()), float((d.max(2) > 8).mean())
    # integer-period state is exact, but 1-2 px AA lines undersample at the
    # pixel grid and flip a few levels: allow sparse speckle, not structure.
    # A wrong period moves whole curves (frac >> 0.05 at any resolution).
    assert dmax <= 32 and frac <= 0.02, f"{name}: seam max {dmax}, lit frac {frac:.4f}"
    # fn(2pi)==fn(0) cannot see a wrap inside the loop (talbot's table window
    # jumped 479->0 between frames 1798 and 1799): compare adjacent raw frames.
    nf = FPS * SECONDS
    fr = [fn(2 * np.pi * n / nf) for n in (0, 1, 600, 601, 1200, 1201, nf - 2, nf - 1)]

    def mad(p, q):
        return float(np.abs(p.astype(np.int16) - q.astype(np.int16)).mean())

    diffs = [mad(fr[i], fr[i + 1]) for i in (0, 2, 4, 6)]         # n -> n+1
    base = float(np.median(diffs)) or 1e-9
    r_end, r_wrap = diffs[3] / base, mad(fr[7], fr[0]) / base     # 1798->1799, 1799->0
    assert max(r_end, r_wrap) <= 2, f"{name}: loop seam ratio end={r_end:.2f} wrap={r_wrap:.2f}"
    return name, dmax, frac


def selfcheck() -> None:
    """Every loop must return to frame 0: fn(2*pi) == fn(0) on the rendered array.

    Integer-period state is exact; 1-2 px antialiased lines undersample the pixel
    grid and flip a few levels (fp32 trig noise), so sparse speckle is allowed.
    A wrong period moves whole curves and fails by orders of magnitude.
    """
    with mp.get_context("fork").Pool(min(13, WORKERS)) as pool:
        for name, mx, frac in pool.map(_seam_one, NAMES):
            print(f"seam {name:14s} max={mx:3d} lit_frac={frac:.4f}", flush=True)
    print("videos selfcheck ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", choices=NAMES, help="subset (default: all)")
    ap.add_argument("--preview", action="store_true", help="960x540, 5 s, single pass")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="encode with libsvtav1, not av1_nvenc")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    codec = "libsvtav1" if args.cpu else "av1_nvenc"
    colorcheck(codec)
    VID.mkdir(exist_ok=True)
    (VID / "captions.txt").write_text("".join(f"{n}\n{TXT[n]}\n\n" for n in sorted(TXT)))
    for name in (args.names or NAMES):
        fn, layers = build(name, args.preview)
        encode(name, fn, layers, args.preview, codec)
    print("all done", flush=True)


def _sigterm(sig, frame):
    """nice + bare kill should not orphan the worker pool mid-video."""
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, _sigterm)

if __name__ == "__main__":
    main()
