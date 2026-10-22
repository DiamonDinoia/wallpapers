#!/bin/bash
# One Plasma session on the current X display and session bus. Starts plasmashell,
# waits until it reports startup complete, saves the screen to NAME.png, prints what
# the Plasma script NAME.js returns, and quits.
#
#   xvfb-run -a dbus-run-session deb/session.sh NAME
set -euo pipefail

QT_LOGGING_RULES=kde.plasmashell.debug=true plasmashell --no-respawn > "$1.log" 2>&1 &
shell=$!
# logged once every desktop containment is ready; the timeout is a hang detector
grep -m1 -q "Plasma Shell startup completed" < <(timeout 180 tail -n +1 -f "$1.log")
import -window root "$1.png"
qdbus6 org.kde.plasmashell /PlasmaShell org.kde.PlasmaShell.evaluateScript "$(cat "$1.js")" > "$1.out"
qdbus6 org.kde.plasmashell /MainApplication quit > /dev/null
timeout 60 tail --pid=$shell -f /dev/null
