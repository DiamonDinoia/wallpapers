#version 440
// Bake: pyside6-qsb --glsl "300es,330" --hlsl 50 --msl 12 -o drop.frag.qsb drop.frag
layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;
layout(std140, binding = 0) uniform buf {
    mat4 qt_Matrix;
    float qt_Opacity;
    vec2 centre;  // the drop, in screen px
    vec2 off;     // the grid's drift
    float cell;
    float u;      // px per 4K plate px
    float r;      // kernel radius
    float reach;  // half the quad: rings, node glows and dots
    float t;      // drop phase, 0 to 1
    float hue;
};

// gen_wallpapers palette; sizes are the 4K plate's, scaled by u
const vec3 TEAL = vec3(26, 188, 156) / 255.0;
const vec3 BREEZE = vec3(29, 153, 243) / 255.0;
const vec3 VIOLET = vec3(155, 89, 182) / 255.0;
const vec3 AMBER = vec3(245, 162, 17) / 255.0;
const int RINGS = 5;
const float GAP = 0.1;
const float RISE = 1.0 - GAP * float(RINGS - 1);

void main() {
    vec2 p = centre + (2.0 * qt_TexCoord0 - 1.0) * reach;
    float d = distance(p, centre);
    float hw = 2.2 * u;  // ring half width: the video's 3 px stamp at 1080p
    vec3 col = hue < 0.5 ? mix(TEAL, BREEZE, 2.0 * hue) : mix(BREEZE, VIOLET, 2.0 * hue - 1.0);
    vec3 c = vec3(0.0);
    for (int j = 0; j < RINGS; ++j) {
        float z = (t - GAP * float(j)) / RISE;
        if (z > 0.0 && z < 1.0) {
            // sqrt(1 - z^2) = 1 - log10 phi(z) / log10 phi(1): the kernel weight in decades
            float a = (0.85 - 0.05 * float(j)) * sqrt(1.0 - z * z);
            c += col * a * clamp(hw + 0.5 - abs(d - r * z), 0.0, 1.0);
        }
    }
    // a grid node takes its kernel weight when the first ring reaches it, then fades out by t = 1
    vec2 node = p - (mod(p - off + 0.5 * cell, cell) - 0.5 * cell);
    float zn = distance(node, centre) / r;
    if (zn < 1.0 && t >= zn * RISE) {
        float glow = sqrt(1.0 - zn * zn) * pow((1.0 - t) / (1.0 - zn * RISE), 2.0);
        c += col * 1.5 * glow * clamp(1.5 - 1.5 * distance(p, node) / (5.5 * 2.0 * u), 0.0, 1.0);
    }
    float rd = d / (2.0 * u);  // the video's dot sizes at 1080p
    c += AMBER * (clamp(1.5 - 1.5 * rd / 3.5, 0.0, 1.0)
                  + 0.8 * pow(1.0 - t, 12.0) * clamp(1.5 - 1.5 * rd / 5.5, 0.0, 1.0));
    fragColor = vec4(c, 0.0) * qt_Opacity;  // alpha 0: premultiplied blending adds c to the grid
}
