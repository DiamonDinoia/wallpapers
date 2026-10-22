#!/bin/bash
# Installs the deb in a clean debian:sid container and runs Plasma and the lock screen
# greeter on Xvfb with Mesa:
#   a  user u, before the install, the control: Plasma's own wallpaper, no lock screen
#      plugin.
#   b  u after it: the update script set the plate slideshow and the Spread lock screen,
#      and the screen matches one installed plate pixel by pixel. The user then picks
#      Spread for the desktop.
#   c  u again: the script does not run twice, so that choice stays, and the screen
#      shows Spread rendered by the system Qt from the installed shaders.
#   g  u's lock screen greeter shows Spread.
#   e  user v, created after the install: the script sets the same defaults.
#   d  v after apt-get purge: Plasma still starts with the defaults the script set.
#      Plasma 6.7 shows a black screen for any wallpaper plugin that is not installed,
#      so u, who picked Spread, is not tested here (README).
#   h  u's greeter after the purge still starts and no longer shows Spread, the
#      control for g.
#
#   deb/test.sh DEB [SHOTS_DIR]
#
# SHOTS_DIR receives the screenshots, a.png b.png c.png d.png e.png g.png h.png.
# IMAGE overrides the image, such as a sid image with Plasma already installed.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
deb=$(realpath "$1")
shots=${2:-$(mktemp -d)}
mkdir -p "$shots"
shots=$(realpath "$shots")
engine=$(command -v podman || command -v docker) || { echo "FAIL  no podman or docker"; exit 1; }

"$engine" run --rm -i -v "$deb:/pkg/$(basename "$deb"):ro" -v "$here/session.sh:/session.sh:ro" -v "$here/greeter.sh:/greeter.sh:ro" -v "$shots:/shots" \
    "${IMAGE:-docker.io/library/debian:sid}" bash -euo pipefail -s <<'SCRIPT'
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends plasma-desktop xvfb xauth libgl1-mesa-dri \
    qt6-tools-dev-tools imagemagick python3-numpy python3-pil > /dev/null
install -d -m1777 /t
cd /t
user() { useradd -m "$1"; install -d -m700 -o "$1" "/tmp/xdg-$1"; }
run() {  # USER SCRIPT ARGS: one X display and session bus for USER, in /t
    su "$1" -c "cd /t && LANG=C.UTF-8 XDG_RUNTIME_DIR=/tmp/xdg-$1 \
        xvfb-run -a -s '-screen 0 1920x1080x24' dbus-run-session $2"
}
session() { run "$1" "/session.sh $2"; }
greeter() { run "$1" "/greeter.sh $2 '$3'"; }
user u
greeter='ConfigFile("kscreenlockerrc", "Greeter").readEntry("WallpaperPlugin")'
plugins='desktops().map(function (d) { return d.wallpaperPlugin; })'
echo "print($plugins + '|' + $greeter)" > a.js
cat > b.js <<EOF
var d = desktops()[0];
d.currentConfigGroup = ["Wallpaper", "org.kde.slideshow", "General"];
print($plugins + "|" + $greeter + "|" + d.readConfig("SlidePaths"));
desktops().forEach(function (d) { d.wallpaperPlugin = "io.github.diamondinoia.spread"; });
EOF
echo "print($plugins)" > c.js
echo "print($plugins + '|' + $greeter)" > e.js
echo "print($plugins)" > d.js
session u a
apt-get install -y -qq /pkg/*.deb > /dev/null
session u b
session u c
greeter u g 'qt.quick.viewport: .*Spread_QMLTYPE_[0-9]+[(][^)]*geometry=0,0 [1-9]'
user v
session v e
cp -r /usr/share/wallpapers/DiamonDinoia plates  # the checks below run after the purge
apt-get purge -y -qq diamondinoia-wallpapers > /dev/null
session v d
greeter u h 'frame rendered'
cp a.png b.png c.png d.png e.png g.png h.png /shots/

python3 - <<'EOF'
from pathlib import Path
import numpy as np
from PIL import Image

def gray(img):
    return np.asarray(img.convert("L"), float)

def spread(g):
    """Background luma and grid peak above it, from column medians folded modulo the cell.

    The drops cover few rows of any column, and the fold finds the grid wherever its
    drift has taken it. Spread at 1920x1080 over 2 h: background 13.8, peak 14.4 to
    18.0; the 13 plates: peak 10.0 at most, for spread.png (2026-10-22, laptop).
    """
    cell = g.shape[0] // 9
    col = np.median(g, axis=0)
    f = np.median(col[: col.size // cell * cell].reshape(-1, cell), axis=0)
    return np.median(f), f.max() - np.median(f)

BG = 13.8  # luma of #0b0e14, Spread's background
out = {n: Path(f"{n}.out").read_text().strip() for n in "abcde"}
shot = {n: Image.open(f"{n}.png") for n in "abcdegh"}
fails = []

def check(ok, claim):
    print(("ok    " if ok else "FAIL  ") + claim)
    fails.extend([] if ok else [claim])

check(out["a"] == "org.kde.image|",
      f"a: before the install Plasma shows its own image, no lock screen plugin ({out['a']})")
plugins, greeter, paths = out["b"].split("|")
check(set(plugins.split(",")) == {"org.kde.slideshow"} and greeter == "io.github.diamondinoia.spread"
      and paths == "/usr/share/wallpapers/DiamonDinoia/",
      f"b: the update script set the slideshow and the Spread lock screen ({out['b']})")
plates = sorted(Path("plates").glob("*.png"))
# rows above the panel: plasmashell adds the panel after it logs startup complete
top = 9 * shot["b"].height // 10
d = sorted((np.abs(gray(shot["b"]) - gray(Image.open(p).resize(shot["b"].size, Image.LANCZOS)))[:top].mean(), p.name)
           for p in plates)
check(len(plates) > 1 and d[0][0] < 3 and d[1][0] > 4 * d[0][0] + 3,
      f"b: the screen is {d[0][1]}, mean |screen - plate| {d[0][0]:.2f}, "
      f"{d[1][0]:.2f} for the nearest other of {len(plates)} installed plates")
check(set(out["c"].split(",")) == {"io.github.diamondinoia.spread"},
      f"c: the script ran once, the user's Spread desktop stays ({out['c']})")
base, peak = spread(gray(shot["c"]))
check(abs(base - BG) < 1.5 and peak > 12,
      f"c: Spread renders: background {base:.1f} (Spread {BG}), grid peak {peak:.1f} (needs 12)")
control = {p.name: spread(gray(Image.open(p).resize(shot["c"].size, Image.LANCZOS))) for p in plates}
worst = max(control, key=lambda n: control[n][1])
check(all(not (abs(b - BG) < 1.5 and k > 12) for b, k in control.values()),
      f"control: no installed plate passes for Spread, the highest grid peak is {control[worst][1]:.1f} ({worst})")
base, peak = spread(gray(shot["g"]))
check(abs(base - BG) < 1.5 and peak > 12,
      f"g: the lock screen greeter shows Spread: background {base:.1f}, grid peak {peak:.1f} (needs 12)")
check(out["e"] == "org.kde.slideshow|io.github.diamondinoia.spread",
      f"e: a user created after the install gets the slideshow and the Spread lock screen ({out['e']})")
gone = not any(Path(f).exists() for f in (paths, "/usr/share/plasma/wallpapers/io.github.diamondinoia.spread"))
check(gone and out["d"] == "org.kde.slideshow",
      f"d: purge removed the plates and the plugin, and Plasma still starts for v ({out['d']})")
base, peak = spread(gray(shot["h"]))
check(not (abs(base - BG) < 1.5 and peak > 12),
      f"h: after the purge the greeter starts without Spread: background {base:.1f}, grid peak {peak:.1f}")
assert not fails, fails
print("deb test ok")
EOF
SCRIPT
