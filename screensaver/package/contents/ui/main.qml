import QtQuick
import QtQuick.Window
import org.kde.plasma.plasmoid

WallpaperItem {
    Spread {
        anchors.fill: parent
        seed: Screen.virtualX + 7 * Screen.virtualY  // each output gets its own drops
        // 30 fps, not every vsync: the ripples move a few px per frame
        Timer {
            readonly property real t0: Date.now()
            interval: 33; repeat: true; running: true
            onTriggered: parent.time = (Date.now() - t0) / 1000
        }
    }
}
