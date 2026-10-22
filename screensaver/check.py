"""Burn-in check for the Spread screensaver: render the shader Plasma runs, over hours.

    uv run --with pyside6==6.11.2 --with numpy --with pillow python screensaver/check.py [video.webm]

PySide6 must match the system Qt that loads the baked .qsb. The check renders
Spread.qml at 960x540 on FRAMES instants spread over HOURS and asserts that every
pixel lights up at least once and that the time-mean image is flat: a static
element (text, a fixed grid) leaves its shape in the mean. The optional video is the
positive control: its mean holds the static equation overlay and must fail the
flatness bound.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtQuick import QQuickView

W, H = 960, 540
FRAMES, HOURS = 1500, 2.0
BG_LUMA = 13.8     # luma of BG #0b0e14
LIT = 6.0          # a pixel counts as lit this far above BG
FLAT = 0.25        # bound on (p99 - p1) / median of the block-averaged mean luma
BLOCK = 8          # px; averages out the sampling noise of FRAMES random frames
GRID_LUMA = 35.7   # luma of GRID #1a2534


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3].astype(np.float64) @ [0.2126, 0.7152, 0.0722]


def flatness(mean: np.ndarray) -> float:
    h, w = (s - s % BLOCK for s in mean.shape)
    b = mean[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean((1, 3))
    lo, med, hi = np.percentile(b, [1, 50, 99])
    return (hi - lo) / med


def shader_frames():
    app = QGuiApplication(sys.argv)
    view = QQuickView()
    view.setResizeMode(QQuickView.SizeRootObjectToView)
    view.setSource(QUrl.fromLocalFile(str(Path(__file__).parent / "package/contents/ui/Spread.qml")))
    if view.status() != QQuickView.Ready:
        sys.exit(f"Spread.qml: {[e.toString() for e in view.errors()]}")
    view.resize(W, H)
    root = view.rootObject()
    root.setProperty("width", W)  # a hidden view never resizes its root: the shader would see res 0
    root.setProperty("height", H)
    for t in np.linspace(0, HOURS * 3600, FRAMES):
        root.setProperty("time", float(t))
        img = view.grabWindow().convertToFormat(QImage.Format_RGB888)
        h, w = img.height(), img.width()  # device pixels: a scaled screen grabs at its scale
        yield np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine())[:, :3 * w].reshape(h, w, 3)
    del app


def video_frames(path: str):
    n = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                                     "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path]))
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select='not(mod(n,{max(1, n // 300)}))',scale={W}:{H}",
                                   "-fps_mode", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def grid_luma(y: np.ndarray) -> float:
    """At time 0 the grid has not drifted: a line runs down x = height / 9."""
    x = round(y.shape[0] / 9)
    return float(np.median(y[:, x - 2:x + 3], axis=0).max())


def stats(frames):
    acc, peak, n, bright, grid = 0.0, None, 0, 0, None
    for f in frames:
        y = luma(f)
        grid = grid_luma(y) if grid is None else grid
        bright += np.median(y) > BG_LUMA + LIT  # most of every frame is background
        peak = y if peak is None else np.maximum(peak, y)
        acc = acc + y
        n += 1
    return acc / n, peak, n, bright, grid


def main() -> None:
    mean, peak, n, bright, grid = stats(shader_frames())
    dark = int((peak < BG_LUMA + LIT).sum())
    flat = flatness(mean)
    Image.fromarray(np.clip(mean * 4, 0, 255).astype(np.uint8)).save("/tmp/spread-mean.png")
    print(f"shader: {n} frames over {HOURS} h at {mean.shape[1]}x{mean.shape[0]} px; pixels never lit: {dark}; "
          f"mean luma {np.median(mean):.1f}, flatness {flat:.3f} (bound {FLAT}); frames not mostly BG: {bright}; "
          f"grid line luma {grid:.1f} (BG {BG_LUMA}, GRID {GRID_LUMA})")
    # a blank or white screen is lit everywhere and flat: the luma bounds reject both
    assert n == FRAMES and bright == 0 and BG_LUMA < np.median(mean) < 40, "shader frames are not the drop scene"
    assert dark == 0 and flat < FLAT, "shader leaves a pixel dark or a static shape"
    assert grid > (BG_LUMA + GRID_LUMA) / 2, "the grid does not draw"
    if len(sys.argv) > 1:
        vm, _, vn, _, _ = stats(video_frames(sys.argv[1]))
        vflat = flatness(vm)
        print(f"control {sys.argv[1]}: {vn} frames, flatness {vflat:.3f}: the static overlay "
              f"{'fails' if vflat >= FLAT else 'PASSES'} the bound")
        assert vflat >= FLAT, "control: a video with a static overlay must fail the flatness bound"
    print("burn-in check ok")


if __name__ == "__main__":
    main()
