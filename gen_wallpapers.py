#!/usr/bin/env python3
"""Generate math/engineering wallpapers: 4K PNGs, SVGs for the line art.

inkscape converts the SVGs. Re-run after tweaking any constant.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

W, H = 3840, 2160
OUT = Path(__file__).parent
BG = "#0b0e14"  # near-black, matches dark KDE schemes

BREEZE = "#1d99f3"  # KDE blue
AMBER = "#f5a211"
TEAL = "#1abc9c"
VIOLET = "#9b59b6"


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
    p = OUT / f"{name}.png"
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(p)
    print("wrote", p)
    return p


def wavefield() -> Path:
    """Interference |sum of Green's functions|^2 from point sources in a cavity."""
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
    scx, scy = 62.0, 34.0
    px = (W / 2 + traj[:, 0] * scx, H * 1.04 - traj[:, 2] * scy)
    pts = np.column_stack(px)
    body = _segments(
        (pts, pts), BREEZE, VIOLET, width=(1.6,), glow=(9.0, 0.07)
    )
    return write_svg("lorenz", body)


def streamlines() -> Path:
    """Phase portrait: RK4 streamlines of a vortex+saddle vector field."""
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
    tilt = np.deg2rad(24.0)
    Yv = 0.45 * Z + 0.89 * Y - 0.13 * np.cos(tilt)  # cheap orthographic tilt
    Xv = X
    D = 0.08 * Z  # depth cue from height

    def project(a, b, d):
        s = 640.0
        return np.column_stack([W / 2 + a * s, H / 2 - b * s]), d

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
    cx, cy, R = W / 2, H / 2, 760.0
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
    body += text_block(
        W - 240, 260,
        ["T\u2099(cos\u2009\u03b8) = cos(n\u2009\u03b8)", "polyfit \u00b7 treeweave"],
        size=56, anchor="end",
    )
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
        ix = ((xv + 2.3) / 4.6 * (H - 1) + (W - H) // 2).astype(np.int32)
        return np.clip(iy, 0, H - 1), np.clip(ix, 0, W - 1)

    for _ in range(n_steps):
        xn = np.sin(a * y) - np.cos(b * x)
        yn = np.sin(c * x) - np.cos(d * y)
        iy, ix = idx(xn, yn)
        np.add.at(dens, (iy, ix), 1.0)
        np.add.at(sx, (iy, ix), xn - x)
        np.add.at(sy, (iy, ix), yn - y)
        x, y = xn, yn
    from matplotlib.colors import hsv_to_rgb

    hue = (np.arctan2(sy, sx) + np.pi) / (2 * np.pi)
    val = np.log1p(dens)
    val /= val.max()
    rgb = hsv_to_rgb(
        np.stack([hue, 0.8 * np.ones_like(hue), 0.9 * val ** 0.55], -1)
    )
    rgb[dens < 1] = hex_to_rgb(BG)
    return save_field("dejong", rgb)


def overlay_eq(name: str, lines: list[str], margin: int = 150, where: str = "auto",
               fontsize: int = 54) -> Path:
    """Composite LaTeX-style (Computer Modern) equations onto a PNG.

    where: 'auto' picks the darkest corner; explicit corners 'br'/'tr'/'bl'/'tl'
    or centered edges 'bc'/'tc'.
    """
    import io

    plt.rcParams["mathtext.fontset"] = "cm"
    fig = plt.figure(figsize=(18, 0.9 * len(lines) + 0.5))
    for i, line in enumerate(lines[::-1]):
        fig.text(
            0.99, (0.08 + 0.92 * i / max(1, len(lines))), line,
            fontsize=fontsize, color="#d8e2ef", ha="right", va="bottom",
        )
    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    eq = Image.open(buf).convert("RGBA")
    if eq.width > 2000:
        eq = eq.resize((2000, int(eq.height * 2000 / eq.width)), Image.LANCZOS)
    p = OUT / f"{name}.png"
    base = Image.open(p).convert("RGBA")

    def rect(corner: str):
        y0 = H - margin - eq.height if "b" in corner else margin
        x0 = (W - eq.width) // 2 if corner.endswith("c") else (
            W - margin - eq.width if "r" in corner else margin)
        return x0, y0

    if where == "auto":
        a = np.asarray(base.convert("RGB")).max(-1)

        def score(c):
            sx, sy = rect(c)
            return (a[sy:sy + eq.height, sx:sx + eq.width] > 60).mean()

        where = min(("br", "tr", "bl", "tl"), key=score)
    x0, y0 = rect(where)
    halo = eq.getchannel("A").point(lambda v: min(200, v + 60))  # drop shadow
    shadow = Image.new("RGBA", eq.size, (0, 0, 0, 0))
    shadow.putalpha(halo.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(5)))
    base.alpha_composite(shadow, (x0 + 8, y0 + 8))
    base.alpha_composite(eq, (x0, y0))
    base.convert("RGB").save(p)
    print(f"equations on {p} ({where})")
    return p


def apply_overlays() -> None:
    overlay_eq(
        "wavefield",
        [r"$a(\mathbf{x}) = \sum_{k=1}^{42} "
         r"\frac{e^{i k |\mathbf{x}-\mathbf{x}_k|}}{\sqrt{|\mathbf{x}-\mathbf{x}_k|}}, "
         r"\quad I = |a|^2$"],
    )
    overlay_eq(
        "lorenz",
        [r"$\dot{x}=\sigma(y-x)\qquad \dot{y}=x(\rho-z)-y\qquad \dot{z}=xy-\beta z$",
         r"$(\sigma,\rho,\beta)=(10,\,28,\,8/3), \quad x\!-\!z\ \mathrm{projection}$"],
    )
    overlay_eq(
        "streamlines",
        [r"$\dot{x} = x - 0.3y - \sum_i \gamma_i\,(y - c_i^y)/r_i^2$",
         r"$\dot{y} = -0.4x - 0.15y + \sum_i \gamma_i\,(x - c_i^x)/r_i^2$",
         r"$6\ \mathrm{vortices},\ \gamma_i \in [-1.4, 1.4],\ \mathrm{RK2}$"],
        where="tl", fontsize=40,
    )
    overlay_eq(
        "torus",
        [r"$X=(R+r^*\cos v)\cos u \quad Y=(R+r^*\cos v)\sin u \quad Z=r^*\sin v$",
         r"$r^*(u,v) = r + 0.12\,\sin 6u\,\cos 3v$"],
        where="bc",
    )
    overlay_eq(
        "dejong",
        [r"$x_{n+1}=\sin(ay_n)-\cos(bx_n)\qquad y_{n+1}=\sin(cx_n)-\cos(dy_n)$",
         r"$(a,b,c,d)=(1.4,\,-2.3,\,2.4,\,-2.1)$",
         r"$\mathrm{hue = flow\ direction, \ brightness = density}$"],
    )


if __name__ == "__main__":
    wavefield()
    lorenz()
    streamlines()
    torus()
    chebyshev()
    dejong()
