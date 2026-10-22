#!/bin/bash
# The lock screen greeter in testing mode on the current X display and session bus.
# Waits for the first frame rendered after an item matching PATTERN is laid out,
# saves the screen to NAME.png, and quits.
#
#   xvfb-run -a dbus-run-session deb/greeter.sh NAME PATTERN
set -euo pipefail

greet=$(dpkg -L libkscreenlocker6 | grep '/kscreenlocker_greet$')
QT_LOGGING_RULES="qt.quick.viewport.debug=true;qt.scenegraph.time.renderloop.debug=true" \
    "$greet" --testing > "$1.log" 2>&1 &
greeter=$!
# the timeout is a hang detector
awk -v p="$2" '$0 ~ p { s = 1 } s && /frame rendered/ { ok = 1; exit } END { exit !ok }' \
    < <(timeout 120 tail -n +1 -f "$1.log")
import -window root "$1.png"
kill "$greeter"
timeout 60 tail --pid=$greeter -f /dev/null
