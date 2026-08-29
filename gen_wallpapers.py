#!/usr/bin/env python3
"""Generate math/engineering wallpapers: 4K PNGs, SVGs for the line art.

inkscape converts the SVGs. Re-run after tweaking any constant.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

W, H = 3840, 2160
OUT = Path(__file__).parent
RAW = OUT / "raw"  # unframed renders; apply_overlays always reads from here
BG = "#0b0e14"  # near-black, matches dark KDE schemes

BREEZE = "#1d99f3"  # KDE blue
AMBER = "#f5a211"
TEAL = "#1abc9c"
VIOLET = "#9b59b6"

CMAP = LinearSegmentedColormap.from_list(
    "breeze", [BG, "#132133", VIOLET, BREEZE, TEAL, AMBER, "#fff2d6"]
)
CYC = LinearSegmentedColormap.from_list(  # cyclic, for phase
    "breeze_cyc", [TEAL, BREEZE, VIOLET, AMBER, TEAL]
)
MONO = LinearSegmentedColormap.from_list(
    "breeze_m", [BG, "#0d2136", "#155a92", BREEZE, "#a9dcff", "#eaf6ff"]
)
DIVERGE = LinearSegmentedColormap.from_list(
    "breeze_d", [TEAL, "#0d2a30", BG, "#33240c", AMBER]
)


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def lerp_hex(a: str, b: str, t: float) -> str:
    pa, pb = hex_to_rgb(a), hex_to_rgb(b)
    c = [int(255 * (pa[i] + (pb[i] - pa[i]) * t)) for i in range(3)]
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def svg_header() -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
    ]


def write_svg(name: str, body: list[str]) -> Path:
    p = OUT / f"{name}.svg"
    p.write_text("\n".join(svg_header() + body + ["</svg>"]) + "\n")
    print("wrote", p)
    return p


def save_field(name: str, rgb: np.ndarray) -> Path:
    RAW.mkdir(exist_ok=True)
    p = RAW / f"{name}.png"
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(p)
    print("wrote", p)
    return p


def wavefield() -> Path:
    """Interference |sum_j e^{ik_j r}/sqrt(r)|^2: 2D cylindrical waves, 42 free-space sources."""
    rng = np.random.default_rng(7)
    y, x = np.mgrid[-0.9:0.9:complex(0, H), -1.6:1.6:complex(0, W)]
    pts = (rng.uniform(-1.3, 1.3, (42, 2)) * np.array([1.5, 0.85])).T
    ks = rng.uniform(18, 34, pts.shape[1])
    field = np.zeros((H, W), dtype=np.complex128)
    for (sx, sy), k in zip(pts.T, ks):
        r = np.hypot(x - sx, y - sy) + 1e-3
        field += np.exp(1j * k * r) / np.sqrt(r)
    amp = np.abs(field) ** 2
    amp = np.log1p(amp * (0.99 / amp.max()))
    cmap = plt.get_cmap("magma")
    return save_field("wavefield", cmap(amp)[..., :3])


def _segments(things, color_a, color_b, width, glow) -> list[str]:
    """Split one long path into per-segment strokes with a color ramp."""
    body = []
    n_seg = 2200
    idx = np.linspace(0, len(things[0]) - 1, n_seg + 1).astype(int)
    for si in range(n_seg):
        i0, i1 = idx[si], idx[si + 1]
        for pts, w, a in ((things[0], width[0], 1.0),) + tuple(
            (g, glow[0], glow[1]) for g in things[1:]
        ):
            d = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts[i0 : i1 + 1])
            c = lerp_hex(color_a, color_b, (i0 / len(pts) + si / n_seg) / 2)
            body.append(
                f'<polyline points="{d}" fill="none" stroke="{c}" '
                f'stroke-width="{w}" stroke-opacity="{a}" stroke-linecap="round"/>'
            )
    return body


def lorenz() -> Path:
    """Lorenz attractor, x-z projection, time-coded color ramp."""
    dt, n = 0.0035, 60000
    p = np.array([0.1, 0.0, 20.0])
    traj = np.empty((n, 3))

    def f(s):
        x, y_, z = s
        return np.array([10 * (y_ - x), x * (28 - z) - y_, x * y_ - 8 / 3 * z])

    for i in range(n):
        k1, k2 = f(p), f(p + dt / 2 * f(p))
        p += dt * k2  # midpoint RK2
        traj[i] = p
    traj = traj[4000:]  # drop transient
    lo, hi = traj[:, [0, 2]].min(0), traj[:, [0, 2]].max(0)
    pad = PLATE_MARGIN + 24  # the glow bleeds well past the stroke
    s = (H - 2 * pad) / (hi[1] - lo[1])  # height binds; 1.88 widens the wings
    px = (W / 2 + (traj[:, 0] - (lo[0] + hi[0]) / 2) * 1.88 * s,
          H - pad + (lo[1] - traj[:, 2]) * s)
    pts = np.column_stack(px)
    body = _segments(
        (pts, pts), BREEZE, VIOLET, width=(1.6,), glow=(9.0, 0.07)
    )
    return write_svg("lorenz", body)


def streamlines() -> Path:
    """Phase portrait: RK2 streamlines of a vortex+saddle vector field."""
    rng = np.random.default_rng(11)
    vort = rng.normal(0, 0.55, (6, 2))
    strength = rng.uniform(0.6, 1.4, 6) * rng.choice([-1, 1], 6)

    def field(s):
        x, y_ = s[..., 0], s[..., 1]
        vx, vy = x - 2 * y_ * 0.15, -0.15 * y_ - 0.4 * x  # saddle drift
        for (cx, cy), g in zip(vort, strength):
            dx, dy = x - cx, y_ - cy
            r2 = dx**2 + dy**2 + 1e-3
            vx += -g * dy / r2
            vy += g * dx / r2
        return np.stack([vx, vy], axis=-1)

    body, n_paths = [], 120
    seeds = rng.uniform(-1.7, 1.7, (n_paths, 2))
    dt, n_step = 0.004, 800
    for pi, s0 in enumerate(seeds):
        s = s0.copy()
        path = np.empty((n_step, 2))
        for i in range(n_step):
            k1 = field(s)
            k2 = field(s + dt / 2 * k1)
            s += dt * k2
            path[i] = s
            if np.hypot(*s) > 2.2:
                path = path[: i + 1]
                break
        if len(path) < 30:
            continue
        px = np.column_stack(
            [W / 2 + path[:, 0] * W / 3.4, H / 2 - path[:, 1] * H / 1.9]
        )
        c = lerp_hex(TEAL, BREEZE, pi / n_paths) if pi % 5 else AMBER
        d = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in px)
        body.append(
            f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="1.8" '
            f'stroke-opacity="0.75" stroke-linecap="round"/>'
        )
    return write_svg("streamlines", body)


def torus() -> Path:
    """Perspective wireframe torus with a modal deformation."""
    u = np.linspace(0, 2 * np.pi, 160)
    v = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    R, r = 2.6, 1.0
    U, V = np.meshgrid(u, v, indexing="ij")
    rr = r + 0.12 * np.sin(6 * U) * np.cos(3 * V)  # superimposed mode
    X, Y, Z = (R + rr * np.cos(V)) * np.cos(U), (R + rr * np.cos(V)) * np.sin(U), rr * np.sin(V)
    Yv = 0.45 * Z + 0.89 * Y - 0.119  # orthographic tilt, 26.8 deg above the xy-plane
    Xv = X
    D = 0.08 * Z  # depth cue from height

    def project(a, b, d):
        s = 297.0
        return np.column_stack([W / 2 - 420 + a * s, H / 2 - b * s]), d

    body = []
    for j in range(len(v)):  # parallels
        pts, d = project(Xv[:, j], Yv[:, j], D[:, j])
        line = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts)
        a = 0.15 + 0.65 * (d.mean() + 1) / 2
        body.append(
            f'<polyline points="{line}" fill="none" stroke="{BREEZE}" stroke-width="1.4" stroke-opacity="{a:.2f}"/>'
        )
    for i in range(0, len(u), 4):  # meridians
        pts, d = project(Xv[i, :], Yv[i, :], D[i, :])
        line = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts)
        a = 0.15 + 0.65 * (d.mean() + 1) / 2
        body.append(
            f'<polyline points="{line}" fill="none" stroke="{AMBER}" stroke-width="1.2" stroke-opacity="{a:.2f}"/>'
        )
    return write_svg("torus", body)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text_block(x, y, lines, size=56, color="#c8d3e0", anchor="start", opacity=0.85):
    """Multi-line SVG text; one tspan per line, dy = size*1.35."""
    parts = [
        f'<text x="{x}" y="{y}" font-family="DejaVu Sans" font-size="{size}" '
        f'fill="{color}" fill-opacity="{opacity}" text-anchor="{anchor}">'
    ]
    for i, line in enumerate(lines):
        parts.append(f'<tspan x="{x}" dy="{0 if i == 0 else size * 1.35}">{esc(line)}</tspan>')
    return ["\n".join(parts) + "</text>"]


def poly(pts, color, width, opacity):
    d = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts)
    return (
        f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-opacity="{opacity}" stroke-linecap="round"/>'
    )


def ramp3(a: str, b: str, c: str, t: float) -> str:
    """Two-stop color ramp a->b->c over t in [0,1]."""
    return lerp_hex(a, b, 2 * t) if t < 0.5 else lerp_hex(b, c, 2 * t - 1)


def chebyshev(n_max: int = 8) -> Path:
    """Rose bouquet: T_n(cos th) = cos(n th) traced in polar coordinates."""
    cx, cy, R = W / 2, H / 2, 940.0
    def ramp4(t):
        if t < 1 / 3: return lerp_hex(TEAL, BREEZE, 3 * t)
        if t < 2 / 3: return lerp_hex(BREEZE, VIOLET, 3 * t - 1)
        return lerp_hex(VIOLET, AMBER, 3 * t - 2)

    body = []
    circ = 2 * np.pi * np.arange(721) / 720
    body.append(poly(np.column_stack([cx + R * np.cos(circ), cy + R * np.sin(circ)]), "#39414f", 2, 0.8))
    body.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{AMBER}"/>')
    for n in range(1, n_max + 1):
        th = np.linspace(0, 2 * np.pi, 4 * n * 180 + 1)
        r = np.cos(n * th)
        pts = np.column_stack([cx + R * r * np.cos(th), cy + R * r * np.sin(th)])
        col = ramp4((n - 1) / (n_max - 1))
        body.append(poly(pts, col, 12, 0.10))
        body.append(poly(pts, col, 5, 0.25))
        body.append(poly(pts, col, 1.8, 0.95))
    return write_svg("chebyshev", body)





def dejong(n_walkers: int = 4000, n_steps: int = 4500) -> Path:
    """De Jong attractor point cloud; brightness = density, hue = step direction."""
    a, b, c, d = 1.4, -2.3, 2.4, -2.1
    rng = np.random.default_rng(3)
    x = rng.uniform(-2, 2, n_walkers)
    y = rng.uniform(-2, 2, n_walkers)
    dens = np.zeros((H, W), dtype=np.float32)
    sx = np.zeros_like(dens)
    sy = np.zeros_like(dens)

    def idx(xv, yv):
        iy = ((yv + 2.3) / 4.6 * (H - 1)).astype(np.int32)
        ix = ((xv + 2.3) / 4.6 * (H - 1) + (W - H) // 2 - 500).astype(np.int32)
        return np.clip(iy, 0, H - 1), np.clip(ix, 0, W - 1)

    for _ in range(n_steps):
        xn = np.sin(a * y) - np.cos(b * x)
        yn = np.sin(c * x) - np.cos(d * y)
        iy, ix = idx(xn, yn)
        np.add.at(dens, (iy, ix), 1.0)
        np.add.at(sx, (iy, ix), xn - x)
        np.add.at(sy, (iy, ix), yn - y)
        x, y = xn, yn
    cyc = CYC
    hue = (np.arctan2(sy, sx) + np.pi) / (2 * np.pi)
    val = np.log1p(dens)
    val /= val.max()
    rgb = np.asarray(cyc(hue))[..., :3] * (0.95 * val ** 0.55)[..., None]
    rgb[dens < 1] = hex_to_rgb(BG)
    return save_field("dejong", rgb)


PLATE_MARGIN = 96  # mat around the artwork
EQ_SCALE = 58 / 225  # 54 pt at 300 dpi is a 225 px em; every caption sets at 58 px
EQ_LINE_H = 92  # minimum line pitch in the caption band, in pixels
EQ_LEAD = 34  # leading added to each caption line
EQ_PAD = 48  # clear space demanded around a caption set inside the artwork
BG_LUM = 13.8  # luminance of BG; a window this flat carries no artwork


def caption(lines: list[str], color: str, max_w: int,
            scale: float = EQ_SCALE) -> list[Image.Image]:
    """Set each equation line in Computer Modern, one tight-cropped figure per line.

    Type size is fixed by scale, so every plate carries the same caption weight.
    """
    import io

    plt.rcParams["mathtext.fontset"] = "cm"
    rows = []
    for line in lines:
        fig = plt.figure(figsize=(18, 1.4))
        fig.text(0.99, 0.5, line, fontsize=54, color=color, ha="right", va="center")
        buf = io.BytesIO()
        fig.savefig(buf, dpi=300, transparent=True, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        rows.append(Image.open(buf).convert("RGBA"))

    sc = min(scale, 0.92 * max_w / max(r.width for r in rows))
    rows = [r.resize((max(1, int(r.width * sc)), max(1, int(r.height * sc))),
                     Image.LANCZOS) for r in rows]
    return rows


def stack(rows: list[Image.Image], align: str) -> Image.Image:
    """Stack the caption lines at a per-line pitch, so a tall line cannot touch its neighbour.

    align is "left", "center" or "right".
    """
    hs = [max(r.height, EQ_LINE_H - EQ_LEAD) + EQ_LEAD for r in rows]
    eq = Image.new("RGBA", (max(r.width for r in rows), sum(hs)), (0, 0, 0, 0))
    y = 0
    off = {"left": lambda rw: 0, "center": lambda rw: (eq.width - rw) // 2,
           "right": lambda rw: eq.width - rw}[align]
    for r, h in zip(rows, hs):
        eq.paste(r, (off(r.width), y + (h - r.height) // 2), r)
        y += h
    return eq


CREDIT = r"$\mathrm{\copyright\ 2026\ DiamonDinoia}$"  # set in Computer Modern, like the equations
CREDIT_LICENSE = r"$\mathrm{CC\ BY\ 4.0}$"  # subordinate to the name, on its own line
CREDIT_LICENSE_SCALE = 0.62  # of the credit type size
CREDIT_SCALE = 0.75  # of the caption type size
CREDIT_COLOR = "#54677e"
CREDIT_ALPHA = 0.8
CORNERS = ("bottom-right", "bottom-left", "top-right", "top-left")
INK_BLOCK = 8  # ink is max-pooled onto this grid before the window search


def band_x(art: Image.Image, y: int, w: int, h: int) -> int:
    """The x placing a w x h caption over the least ink in the horizontal band at y.

    Snapping to a third of the plate is arbitrary; the band is scanned instead, and the
    window with the fewest lit pixels wins.
    """
    lum = np.asarray(art.convert("RGB"), dtype=np.float32) @ np.array(
        [0.2126, 0.7152, 0.0722], dtype=np.float32
    )
    ink = lum[y - EQ_PAD:y + h + EQ_PAD] > BG_LUM + 8
    sat = np.zeros((ink.shape[0], ink.shape[1] + 1), dtype=np.int64)
    sat[:, 1:] = ink.cumsum(1)
    ww = w + 2 * EQ_PAD
    xs = np.arange(PLATE_MARGIN, W - PLATE_MARGIN - ww + 1, INK_BLOCK)
    return int(xs[(sat[:, xs + ww] - sat[:, xs]).sum(0).argmin()]) + EQ_PAD


def anchor_xy(anchor: str, w: int, h: int, art: Image.Image | None = None) -> tuple[int, int]:
    """Top-left pixel of a w x h caption at a named inset position.

    The name is "<top|middle|bottom>[-<left|center|right>]". Named without a side, the
    caption slides along that band to wherever it covers the least artwork.
    """
    v, _, hz = anchor.partition("-")
    y = {"top": PLATE_MARGIN + EQ_PAD, "middle": (H - h) // 2,
         "bottom": H - PLATE_MARGIN - EQ_PAD - h}[v]
    if not hz:
        return band_x(art, y, w, h), y
    x = {"left": PLATE_MARGIN + EQ_PAD, "center": (W - w) // 2,
         "right": W - PLATE_MARGIN - EQ_PAD - w}[hz]
    return x, y


def empty_corner(art: Image.Image, w: int, h: int) -> str | None:
    """The first corner whose w x h window holds no artwork, else None.

    A window qualifies only if it is flat background: the test is on the 99.9th percentile
    and the maximum, so a handful of stray pixels still disqualifies it.
    """
    lum = np.asarray(art.convert("RGB"), dtype=np.float32) @ np.array(
        [0.2126, 0.7152, 0.0722], dtype=np.float32
    )
    for anchor in CORNERS:
        x, y = anchor_xy(anchor, w, h)
        win = lum[y - EQ_PAD:y + h + EQ_PAD, x - EQ_PAD:x + w + EQ_PAD]
        if np.percentile(win, 99.9) <= BG_LUM + 8 and win.max() <= BG_LUM + 14:
            return anchor
    return None


def empty_window(art: Image.Image, w: int, h: int) -> tuple[int, int] | None:
    """Top-left of the emptiest w x h window anywhere in the plate, or None.

    Ink is max-pooled onto a coarse grid, so one bright pixel disqualifies its whole cell,
    and a summed-area table then rejects every window that contains any. Among the windows
    that survive, the one whose centre is farthest from the nearest ink wins, which puts the
    caption in the middle of the largest hole rather than against its edge.
    """
    from scipy import ndimage

    b = INK_BLOCK
    lum = np.asarray(art.convert("RGB"), dtype=np.float32) @ np.array(
        [0.2126, 0.7152, 0.0722], dtype=np.float32
    )
    ink = lum.reshape(H // b, b, W // b, b).max(axis=(1, 3)) > BG_LUM + 8
    sat = np.zeros((ink.shape[0] + 1, ink.shape[1] + 1), dtype=np.int64)
    sat[1:, 1:] = ink.cumsum(0).cumsum(1)
    cw, ch = -(-(w + 2 * EQ_PAD) // b), -(-(h + 2 * EQ_PAD) // b)
    if cw >= ink.shape[1] or ch >= ink.shape[0]:
        return None
    count = sat[ch:, cw:] - sat[:-ch, cw:] - sat[ch:, :-cw] + sat[:-ch, :-cw]
    m = PLATE_MARGIN // b
    ok = np.zeros(count.shape, dtype=bool)
    ok[m:count.shape[0] - m, m:count.shape[1] - m] = True
    ok &= count == 0
    if not ok.any():
        return None
    clear = ndimage.distance_transform_edt(~ink)
    cy = np.clip(np.arange(count.shape[0])[:, None] + ch // 2, 0, clear.shape[0] - 1)
    cx = np.clip(np.arange(count.shape[1])[None, :] + cw // 2, 0, clear.shape[1] - 1)
    i, j = np.unravel_index(np.where(ok, clear[cy, cx], -1.0).argmax(), count.shape)
    return j * b + (cw * b - w) // 2, i * b + (ch * b - h) // 2


def sign(art: Image.Image, taken: tuple[int, int, int, int]) -> str:
    """Set the credit in the corner with the least ink that the caption does not hold.

    Overlap costs more than any amount of artwork, so the signature always yields the
    corner to the equation. Returns the corner it took.
    """
    sc = EQ_SCALE * CREDIT_SCALE
    name = caption([CREDIT], CREDIT_COLOR, W, scale=sc)[0]
    lic = caption([CREDIT_LICENSE], CREDIT_COLOR, W, scale=sc * CREDIT_LICENSE_SCALE)[0]
    w, h = max(name.width, lic.width), name.height + 14 + lic.height
    lum = np.asarray(art.convert("RGB"), dtype=np.float32) @ np.array(
        [0.2126, 0.7152, 0.0722], dtype=np.float32
    )
    at, xy, best = None, None, None
    for anchor in CORNERS:
        x, y = anchor_xy(anchor, w, h)
        win = lum[y - EQ_PAD:y + h + EQ_PAD, x - EQ_PAD:x + w + EQ_PAD]
        hit = not (x + w < taken[0] or x > taken[2] or y + h < taken[1] or y > taken[3])
        score = (win > BG_LUM + 8).mean() + 10.0 * hit
        if best is None or score < best:
            at, xy, best = anchor, (x, y), score
    sig = Image.new("RGBA", (w, h), (0, 0, 0, 0))  # the license hangs off the same edge
    off = (lambda im: 0) if at.endswith("left") else (lambda im: w - im.width)
    sig.paste(name, (off(name), 0))
    sig.paste(lic, (off(lic), name.height + 14))
    art.paste(sig.convert("RGB"), xy,
              sig.getchannel("A").point(lambda v: int(v * CREDIT_ALPHA)))
    return at


def overlay_eq(name: str, lines: list[str], anchor: str | None = None,
               color: str = "#9db3cd", alpha: float = 0.92) -> Path:
    """Set the equation inside an empty corner of the artwork, or frame the artwork below it.

    The caption never covers artwork it would obscure. When the plate has a corner of flat
    background wide enough to hold it, the image stays full-bleed and the equation sits in
    that corner. Otherwise the artwork is matted into a plate with a band of its own below.
    An explicit anchor names a corner the plate can carry, and skips the search.
    """
    plate_w = W - 2 * PLATE_MARGIN
    rows = caption(lines, color, plate_w)
    eq = stack(rows, "right")
    art = Image.open(RAW / f"{name}.png").convert("RGB")
    p = OUT / f"{name}.png"

    at = anchor or empty_corner(art, eq.width, eq.height)
    if at is not None:  # a named position, or a corner the artwork leaves clear
        eq = stack(rows, at.partition("-")[2] or "center")
        xy = anchor_xy(at, eq.width, eq.height, art)
    else:  # no clear corner: look for a hole anywhere in the plate
        at, eq = "hole", stack(rows, "center")  # same size, so the search is unaffected
        xy = empty_window(art, eq.width, eq.height)
    if xy is not None:
        art.paste(eq.convert("RGB"), xy,
                  eq.getchannel("A").point(lambda v: int(v * alpha)))
        sg = sign(art, (xy[0], xy[1], xy[0] + eq.width, xy[1] + eq.height))
        art.save(p)
        print(f"inset {p} {at} at {xy}, signed {sg}")
        return p

    band = eq.height + PLATE_MARGIN + 64  # gap to the plate, then an even bottom mat
    plate_h = H - PLATE_MARGIN - band
    sc = max(plate_w / art.width, plate_h / art.height)  # centre-crop to the plate
    cw, ch = int(plate_w / sc), int(plate_h / sc)
    art = art.crop(((art.width - cw) // 2, (art.height - ch) // 2,
                    (art.width - cw) // 2 + cw, (art.height - ch) // 2 + ch))
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(art.resize((plate_w, plate_h), Image.LANCZOS), (PLATE_MARGIN, PLATE_MARGIN))
    canvas.paste(eq.convert("RGB"),
                 (PLATE_MARGIN + plate_w - eq.width, H - PLATE_MARGIN - eq.height),
                 eq.getchannel("A").point(lambda v: int(v * alpha)))
    sign(canvas, (PLATE_MARGIN + plate_w - eq.width, H - PLATE_MARGIN - eq.height, W, H))
    canvas.save(p)
    print(f"framed {p} (plate {plate_w}x{plate_h}, band {band})")
    return p


def apply_overlays() -> None:
    missing = [n for n in NAMES if not (RAW / f"{n}.png").exists()]
    assert not missing, f"no raw render for {missing}"
    overlay_eq(
        "wavefield",
        [r"$a(\mathbf{x}) = \sum_{j=1}^{42} "
         r"\frac{e^{i k_j |\mathbf{x}-\mathbf{x}_j|}}{\sqrt{|\mathbf{x}-\mathbf{x}_j|}}, "
         r"\quad k_j \in [18, 34], \quad I = |a|^2$"],
    )
    overlay_eq(
        "lorenz",
        [r"$\dot{x}=\sigma(y-x)$", r"$\dot{y}=x(\rho-z)-y$", r"$\dot{z}=xy-\beta z$",
         r"$(\sigma,\rho,\beta)=(10,\,28,\,8/3)$",
         r"$x\!-\!z\ \mathrm{projection}$"],
    )
    overlay_eq(
        "streamlines",
        [r"$\dot{x} = x - 0.3y - \sum_i \gamma_i\,(y - c_i^y)/r_i^2$",
         r"$\dot{y} = -0.4x - 0.15y + \sum_i \gamma_i\,(x - c_i^x)/r_i^2$",
         r"$6\ \mathrm{vortices},\ \gamma_i \in \pm[0.6, 1.4],\ \mathrm{RK2}$"],
        anchor="top-left",  # the vortex sits right of centre, the streaks here are sparse
    )
    overlay_eq(
        "torus",
        [r"$X=(R+r^*\cos v)\cos u$", r"$Y=(R+r^*\cos v)\sin u$", r"$Z=r^*\sin v$",
         r"$r^*(u,v) = r + 0.12\,\sin 6u\,\cos 3v$"],
    )
    overlay_eq(
        "dejong",
        [r"$x_{n+1}=\sin(ay_n)-\cos(bx_n)$", r"$y_{n+1}=\sin(cx_n)-\cos(dy_n)$",
         r"$(a,b,c,d)=(1.4,\,-2.3,\,2.4,\,-2.1)$",
         r"$\mathrm{hue = flow\ direction, \ brightness = density}$"],
    )
    apply_overlays_numerics()




# --------------------------------------------------------------------------
# numerics set: kernels, spectra, quadrature-adjacent geometry
# --------------------------------------------------------------------------

def fig4k(lim: tuple[float, float, float, float]):
    """Axes-free 4K figure; lim = (x0, x1, y0, y1) in data units, filled edge to edge."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set_axis_off()
    return fig, ax


def save_fig(name: str, fig) -> Path:
    RAW.mkdir(exist_ok=True)
    p = RAW / f"{name}.png"
    fig.savefig(p, dpi=100, facecolor=BG)
    plt.close(fig)
    print("wrote", p)
    return p


def frame(x_half: float):
    """Data limits that fill 3840x2160 with x in [-x_half, x_half]."""
    y_half = x_half * H / W
    return (-x_half, x_half, -y_half, y_half)


def talbot() -> Path:
    """Talbot carpet: Fresnel self-imaging of a duty-1/2 grating, |u(x,z)|^2."""
    n = np.arange(-100, 101)
    c = np.where(n == 0, 0.5, np.sinc(n / 2) / 2)  # binary grating coefficients
    x = np.linspace(-0.9, 0.9, W)  # x/a, 1.8 periods
    zeta = np.linspace(0.02, 0.28, H)  # z/z_T, z_T = 2a^2/lambda
    modes = c * np.exp(2j * np.pi * np.outer(x, n))  # (W, N)
    prop = np.exp(-2j * np.pi * np.outer(zeta, n.astype(np.float64) ** 2))  # (H, N)
    u = prop @ modes.T
    amp = np.abs(u) ** 2
    amp = np.clip(amp / np.percentile(amp, 99.9), 0, 1) ** 1.8
    return save_field("talbot", MONO(amp)[..., :3])


def spread(m: int = 38, w: int = 7) -> Path:
    """Type-1 NUFFT: ES-kernel level sets over the sigma=2 grid, one set per source."""
    rng = np.random.default_rng(23)
    beta = 2.30 * w
    R = 300.0  # kernel support half-width, w/2 grid cells
    body = []
    step = H // 9
    for gx in range(0, W, step):  # upsampled grid
        body.append(poly([(gx, 0), (gx, H)], "#1a2534", 1.6, 1.0))
    for gy in range(0, H, step):
        body.append(poly([(0, gy), (W, gy)], "#1a2534", 1.6, 1.0))
    th = np.linspace(0, 2 * np.pi, 361)
    dec = np.arange(0.5, 7.0, 0.5)  # phi = 10^-dec, down to the kernel floor e^{-beta}
    dec = dec[dec * np.log(10) < beta]
    rad = R * np.sqrt(1 - (1 - dec * np.log(10) / beta) ** 2)
    for k in range(m):
        sx, sy = rng.uniform(-R, W + R), rng.uniform(-R, H + R)
        col = ramp3(TEAL, BREEZE, VIOLET, rng.random())
        body.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="7" fill="{AMBER}"/>')
        for li, r in enumerate(rad):
            pts = np.column_stack([sx + r * np.cos(th), sy + r * np.sin(th)])
            body.append(poly(pts, col, 2.2, 0.85 - 0.05 * li))
    return write_svg("spread", body)


def _grcar(n: int = 32) -> np.ndarray:
    a = np.zeros((n, n))
    for k in range(-1, 4):
        np.fill_diagonal(a[max(0, -k):, max(0, k):], -1.0 if k == -1 else 1.0)
    return a


def pseudospectra(nx: int = 1000) -> Path:
    """eps-pseudospectra of the Grcar matrix: level sets of sigma_min(zI - A)."""
    a = _grcar()
    half = 5.9  # spectrum spans Re 0.1..1.7, Im +-2.25
    lim = (0.9 - half, 0.9 + half, -half * H / W, half * H / W)
    ny = int(nx * H / W)
    zr = np.linspace(lim[0], lim[1], nx)
    zi = np.linspace(lim[2], lim[3], ny)
    z = zr[None, :] + 1j * zi[:, None]
    eye = np.eye(a.shape[0])
    smin = np.empty(z.shape)
    flat = z.ravel()
    out = np.empty(flat.size)
    for i in range(0, flat.size, 8192):
        blk = flat[i:i + 8192, None, None] * eye - a
        out[i:i + 8192] = np.linalg.svd(blk, compute_uv=False)[:, -1]
    smin = out.reshape(z.shape)
    fig, ax = fig4k(lim)
    v = np.clip(-np.log10(smin), 0, 9)  # bright where the resolvent blows up
    lv = np.linspace(0, 9, 37)
    ax.contourf(zr, zi, v, levels=lv, cmap=CMAP, extend="both")
    ax.contour(zr, zi, v, levels=lv[::4], colors=[BG], linewidths=1.4, alpha=0.55)
    ev = np.linalg.eigvals(a)
    ax.plot(ev.real, ev.imag, ".", color="#fff2d6", ms=14)
    return save_fig("pseudospectra", fig)


def stability(p_max: int = 8) -> Path:
    """Absolute stability: |R_p(z)| = 1 for the order-p Taylor/explicit-RK polynomials."""
    lim = tuple(np.add(frame(8.2), (-2.0, -2.0, 0, 0)))
    zr = np.linspace(lim[0], lim[1], 1800)
    zi = np.linspace(lim[2], lim[3], 1013)
    z = zr[None, :] + 1j * zi[:, None]
    fig, ax = fig4k(lim)
    for p in range(p_max, 0, -1):
        r = sum(z**k / math.factorial(k) for k in range(p + 1))
        t = (p - 1) / (p_max - 1)
        c = ramp3(TEAL, BREEZE, VIOLET, 1.5 * t) if t < 2 / 3 else lerp_hex(VIOLET, AMBER, 3 * t - 2)
        ax.contourf(zr, zi, np.abs(r), levels=[0, 1], colors=[c], alpha=0.16)
        ax.contour(zr, zi, np.abs(r), levels=[1], colors=[c], linewidths=3.0)
    ax.plot([lim[0], lim[1]], [0, 0], color="#39414f", lw=1.5)
    ax.plot([0, 0], [lim[2], lim[3]], color="#39414f", lw=1.5)
    return save_fig("stability", fig)


def gammaplane() -> Path:
    """Domain colouring of Gamma(z): hue = arg, brightness bands = log2 |Gamma|."""
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
    v = 0.35 + 0.65 * (band - np.floor(band))  # one band per octave
    v *= 1 / (1 + np.exp(-0.55 * band))  # fade the deep zeros
    v *= 1 / (1 + np.exp((band - 7.0) / 2.0))  # and the overflow side
    rgb = np.asarray(CYC(hue))[..., :3]
    return save_field("gamma", rgb * v[..., None] ** 0.9)


def littlewood(deg: int = 20, chunk: int = 16384) -> Path:
    """Roots of every Littlewood polynomial sum_k a_k z^k, a_k = +-1, of degree 20."""
    lim = frame(1.75)
    nx, ny = W, H
    dens = np.zeros((ny, nx), dtype=np.float32)
    n_poly = 1 << deg
    bits = 1 << np.arange(deg)
    comp = np.zeros((deg, deg))
    comp[1:, :-1] = np.eye(deg - 1)
    for start in range(0, n_poly, chunk):
        idx = np.arange(start, min(start + chunk, n_poly))
        sign = np.where((idx[:, None] & bits) > 0, 1.0, -1.0)  # a_0..a_{deg-1}
        m = np.repeat(comp[None], len(idx), 0)
        m[:, 0, :] = -sign  # monic leading coefficient a_deg = 1
        r = np.linalg.eigvals(m).ravel()
        ix = ((r.real - lim[0]) / (lim[1] - lim[0]) * (nx - 1)).astype(np.int32)
        iy = ((r.imag - lim[2]) / (lim[3] - lim[2]) * (ny - 1)).astype(np.int32)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        np.add.at(dens, (iy[ok], ix[ok]), 1.0)
    v = np.log1p(dens)
    v = (v / v.max()) ** 0.62  # lift the midtones without clipping the ring
    return save_field("littlewood", CMAP(v)[..., :3])


def helmholtz(ka: float = 12.0, n_max: int = 40) -> Path:
    """Sound-soft cylinder scattering: Re(u), Mie series u = e^{ikx} + u_sc."""
    from scipy.special import hankel1, jv

    lim = frame(6.4)
    x = np.linspace(lim[0], lim[1], W)
    y = np.linspace(lim[2], lim[3], H)
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)
    k = ka  # unit cylinder radius, so k = ka
    rs = np.linspace(1.0, r.max() + 0.01, 6000)  # radial table, interpolated onto the grid
    u = np.exp(1j * k * X).astype(np.complex128)
    for n in range(-n_max, n_max + 1):
        cn = -(1j**n) * jv(n, ka) / hankel1(n, ka)
        hs = hankel1(n, k * rs)
        h = np.interp(r, rs, hs.real) + 1j * np.interp(r, rs, hs.imag)
        u += cn * h * np.exp(1j * n * th)
    f = np.real(u)
    f[r < 1.0] = 0.0
    f = np.clip(f / 1.9, -1, 1)
    f = np.sign(f) * np.abs(f) ** 1.4  # let the crests carry the contrast
    return save_field("helmholtz", DIVERGE(0.5 + 0.5 * f)[..., :3])


SVGS = ("lorenz", "streamlines", "torus", "chebyshev", "spread")
NAMES = [
    "chebyshev", "dejong", "gamma", "helmholtz", "littlewood", "lorenz",
    "pseudospectra", "spread", "stability", "streamlines", "talbot", "torus", "wavefield",
]


def thumbs(width: int = 960) -> None:
    """960x540 gallery thumbnails, uploaded next to the 4K assets in the release."""
    d = OUT / "thumbs"
    d.mkdir(exist_ok=True)
    for name in NAMES:
        src = OUT / f"{name}.png"
        if not src.exists():
            continue
        im = Image.open(src)
        im.thumbnail((width, width), Image.LANCZOS)
        im.convert("RGB").save(d / f"{name}-thumb.png", optimize=True)
    print("wrote", d)


def apply_overlays_numerics() -> None:
    overlay_eq(
        "chebyshev",
        [r"$T_n(\cos\theta) = \cos n\theta, \quad n = 1 \ldots 8$"],
    )
    overlay_eq(
        "talbot",
        [r"$u(x,z) = \sum_n c_n\, e^{2\pi i n x/a}\, e^{-i\pi \lambda z n^2/a^2}, "
         r"\quad z_T = 2a^2/\lambda$"],
        anchor="top",  # the carpet fills the plate, so the caption takes its dimmest band
    )
    overlay_eq(
        "spread",
        [r"$\varphi(z) = e^{\beta(\sqrt{1-z^2}-1)}, \quad \beta = 2.30\,w, \ w = 7$",
         r"$f_j = \sum_k c_k\, \varphi(\sigma x_k - j), \quad \sigma = 2, "
         r"\quad \varphi = 10^{-1/2} \ldots 10^{-13/2}$"],
        anchor="top",  # the sigma=2 grid rules out every window; the band scan clears the rings
    )
    overlay_eq(
        "pseudospectra",
        [r"$\Lambda_\varepsilon(A) = \{z : \sigma_{\min}(zI - A) < \varepsilon\}$",
         r"$A = \mathrm{Grcar}(32), \quad \varepsilon = 10^{-9} \ldots 10^{0}$"],
    )
    overlay_eq(
        "stability",
        [r"$R_p(z) = \sum_{k=0}^{p} z^k/k!, \quad |R_p(z)| = 1, \ p = 1 \ldots 8$"],
    )
    overlay_eq(
        "gamma",
        [r"$\Gamma(z) = \int_0^\infty t^{z-1} e^{-t}\,dt$",
         r"$\mathrm{hue} = \arg\Gamma, \quad \mathrm{bands} = \log_2|\Gamma|$"],
    )
    overlay_eq(
        "littlewood",
        [r"$\{z \ : \ \sum_{k=0}^{20} a_k z^k = 0, \ \ a_k = \pm 1, \ a_{20} = 1\}$",
         r"$2^{20}\ \mathrm{polynomials}, \ 2.1 \times 10^{7}\ \mathrm{roots}$"],
    )
    overlay_eq(
        "helmholtz",
        [r"$u = e^{ikx} - \sum_n i^n \, [J_n(ka)/H^{(1)}_n(ka)] \, H^{(1)}_n(kr)\, e^{in\theta}, "
         r"\ ka = 12$"],
    )


def selfcheck() -> None:
    """Ring radii invert the ES kernel, the emptiness test fires both ways, framing is idempotent."""
    beta = 2.30 * 7
    dec = np.arange(0.5, 7.0, 0.5)
    dec = dec[dec * np.log(10) < beta]
    rad = np.sqrt(1 - (1 - dec * np.log(10) / beta) ** 2)
    phi = np.exp(beta * (np.sqrt(1 - rad**2) - 1))
    assert np.allclose(phi, 10.0**-dec, rtol=1e-9), phi
    if (RAW / "streamlines.png").exists():  # the emptiness test must fire both ways
        busy = Image.open(RAW / "streamlines.png")
        clear = Image.open(RAW / "pseudospectra.png")
        assert empty_corner(busy, 1000, 300) is None, "empty_corner accepted a busy plate"
        assert empty_corner(clear, 1000, 300) is not None, "empty_corner rejected a clear plate"
        assert empty_window(Image.open(RAW / "littlewood.png"), 900, 300) is not None
        assert empty_window(Image.open(RAW / "spread.png"), 900, 300) is None
        art, w, h = Image.open(RAW / "talbot.png"), 1200, 160  # the band scan must beat its centre
        lum = np.asarray(art.convert("RGB"), dtype=np.float32) @ np.array(
            [0.2126, 0.7152, 0.0722], dtype=np.float32
        )
        y = PLATE_MARGIN + EQ_PAD
        ink = lambda x: (lum[y - EQ_PAD:y + h + EQ_PAD,
                             x - EQ_PAD:x + w + EQ_PAD] > BG_LUM + 8).sum()
        assert ink(band_x(art, y, w, h)) < ink((W - w) // 2) / 10, "band_x found no emptier x"
    blank = Image.new("RGB", (W, H), BG)  # the credit yields its corner to the caption
    assert sign(blank.copy(), (0, 0, 1, 1)) == "bottom-right"
    assert sign(blank.copy(), (W - 900, H - 400, W, H)) != "bottom-right"
    src = RAW / f"{NAMES[0]}.png"
    if src.exists():  # framing reads raw/ and writes here, so a second pass is a no-op
        (RAW / "_check.png").write_bytes(src.read_bytes())
        out = [overlay_eq("_check", [r"$\varphi$"]).read_bytes() for _ in range(2)]
        (RAW / "_check.png").unlink()
        (OUT / "_check.png").unlink()
        assert out[0] == out[1], "overlay_eq is not idempotent"
    print("selfcheck ok")


def rasterize(names: tuple[str, ...] = SVGS) -> None:
    """inkscape converts each SVG to a 4K PNG in raw/."""
    import subprocess
    RAW.mkdir(exist_ok=True)
    for name in names:
        subprocess.run(["nice", "-n19", "inkscape", str(OUT / f"{name}.svg"),
                        "-o", str(RAW / f"{name}.png"), "-w", str(W), "-h", str(H)],
                       check=True, capture_output=True)
        print("rasterized", name)


if __name__ == "__main__":
    wavefield()
    lorenz()
    streamlines()
    torus()
    chebyshev()
    dejong()
    talbot()
    spread()
    pseudospectra()
    stability()
    gammaplane()
    littlewood()
    helmholtz()
    rasterize()
    selfcheck()
    apply_overlays()
    thumbs()
