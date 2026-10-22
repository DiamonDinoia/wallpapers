#!/bin/bash
# Builds diamondinoia-wallpapers_<version>_all.deb into OUT_DIR from the given 4K plates
# and the Spread screensaver plugin. The version is 1.<commits on HEAD>, so every
# commit on main sorts above the one before it.
#
#   deb/build.sh OUT_DIR PLATE.png...
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out=$1; shift
[ $# -gt 0 ] || { echo "FAIL  no plates given"; exit 1; }
version=1.$(git -C "$here" rev-list --count HEAD)
maintainer="Marco Barbone <mbarbone@flatironinstitute.org>"
tree=$(mktemp -d)
trap 'rm -rf "$tree"' EXIT

install -Dm644 -t "$tree/usr/share/wallpapers/DiamonDinoia" "$@"
mkdir -p "$tree/usr/share/plasma/wallpapers"
cp -r "$here/../screensaver/package" "$tree/usr/share/plasma/wallpapers/io.github.diamondinoia.spread"
# plasmashell runs each script in this directory once per user, at its next start
install -Dm644 "$here/diamondinoia-wallpapers.js" \
    "$tree/usr/share/plasma/shells/org.kde.plasma.desktop/contents/updates/diamondinoia-wallpapers.js"
doc=$tree/usr/share/doc/diamondinoia-wallpapers
install -Dm644 "$here/../LICENSE" "$doc/copyright"
printf 'diamondinoia-wallpapers (%s) unstable; urgency=medium\n\n  * Built from %s.\n\n -- %s  %s\n' \
    "$version" "$(git -C "$here" rev-parse --short HEAD)" "$maintainer" \
    "$(git -C "$here" log -1 --format=%aD)" | gzip -9n > "$doc/changelog.gz"
find "$tree" -type d -exec chmod 755 {} +
find "$tree" -type f -exec chmod 644 {} +

mkdir "$tree/DEBIAN"
cat > "$tree/DEBIAN/control" <<EOF
Package: diamondinoia-wallpapers
Version: $version
Architecture: all
Maintainer: $maintainer
Depends: plasma-desktop (>= 4:6.0), libqt6quick6 (>= 6.11)
Installed-Size: $(du -sk --exclude=DEBIAN "$tree" | cut -f1)
Section: x11
Priority: optional
Homepage: https://github.com/DiamonDinoia/wallpapers
Description: Generative maths wallpapers and the Spread screensaver for KDE Plasma
 Installs $# 3840x2160 plates in /usr/share/wallpapers/DiamonDinoia and the
 Spread lock screen wallpaper plugin. At the next Plasma start of each user,
 the desktop becomes a slideshow of the plates and the lock screen shows
 Spread. Each user can change both afterwards; the package sets them only once.
EOF
mkdir -p "$out"
dpkg-deb --root-owner-group -Zxz --build "$tree" "$out/diamondinoia-wallpapers_${version}_all.deb"
