import QtQuick

// The spread plate as an endless screensaver: every drop slot lands at a new random
// place each cycle and the grid drifts, so no pixel holds a static image. Each drop is
// its own quad, so a pixel shades only the few drops that reach it.
Item {
    id: root
    property real time: 0  // s
    property real seed: 0
    readonly property real u: height / 2160  // px per 4K plate px
    readonly property real r: 300 * u
    readonly property real reach: r + Math.max(2.2 * u + 1, 12 * u)
    readonly property real cell: height / 9
    // the grid drifts one cell in 600 s along x and at an incommensurate rate along y, so its lines sweep every pixel
    readonly property vector2d off: Qt.vector2d(time / 600 % 1 * cell, time / (600 * 1.618034) % 1 * cell)
    readonly property real sd: hash((seed + 0x9e3779b9) >>> 0)

    function hash(x) {  // lowbias32
        x ^= x >>> 16; x = Math.imul(x, 0x7feb352d); x ^= x >>> 15; x = Math.imul(x, 0x846ca68b); x ^= x >>> 16;
        return x >>> 0;
    }
    function lo16(h) { return (h & 0xffff) / 65536; }
    function hi16(h) { return (h >>> 16) / 65536; }

    // 1080 lines like the videos, scaled to the output: a 4K output costs a quarter
    layer.enabled: true
    layer.smooth: true
    layer.textureSize: Qt.size(Math.round(1080 * width / Math.max(height, 1)), 1080)

    ShaderEffect {
        anchors.fill: parent
        property vector2d res: Qt.vector2d(width, height)
        property vector2d off: root.off
        property real cell: root.cell
        property real u: root.u
        fragmentShader: Qt.resolvedUrl("../shaders/grid.frag.qsb")
    }
    Repeater {
        model: 38
        ShaderEffect {
            required property int index
            readonly property real slot: root.hash(index ^ root.sd)  // 16-bit fields: period and phase of this slot
            readonly property real k: root.time / (5 + 5 * root.lo16(slot)) + root.hi16(slot)  // 6 to 12 drops per minute
            readonly property real at: root.hash(slot ^ Math.imul(Math.floor(k), 0x9e3779b9))  // this drop's place
            property vector2d centre: Qt.vector2d(root.lo16(at) * (root.width + 2 * root.r) - root.r,
                                                  root.hi16(at) * (root.height + 2 * root.r) - root.r)
            property vector2d off: root.off
            property real cell: root.cell
            property real u: root.u
            property real r: root.r
            property real reach: root.reach
            property real t: k - Math.floor(k)
            property real hue: (root.hash(at) >>> 8) / 16777216
            x: centre.x - reach
            y: centre.y - reach
            width: 2 * reach
            height: 2 * reach
            fragmentShader: Qt.resolvedUrl("../shaders/drop.frag.qsb")
        }
    }
}
