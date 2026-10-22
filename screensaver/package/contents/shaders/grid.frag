#version 440
// Bake: pyside6-qsb --glsl "300es,330" --hlsl 50 --msl 12 -o grid.frag.qsb grid.frag
layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;
layout(std140, binding = 0) uniform buf {
    mat4 qt_Matrix;
    float qt_Opacity;
    vec2 res;
    vec2 off;
    float cell;
    float u;
};

// gen_wallpapers palette
const vec3 BG = vec3(11, 14, 20) / 255.0;
const vec3 GRID = vec3(26, 37, 52) / 255.0;

void main() {
    vec2 q = abs(mod(qt_TexCoord0 * res - off + 0.5 * cell, cell) - 0.5 * cell);
    fragColor = vec4(mix(BG, GRID, clamp(1.3 * u + 0.5 - min(q.x, q.y), 0.0, 1.0)), 1.0) * qt_Opacity;
}
