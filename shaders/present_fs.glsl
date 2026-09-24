#version 460

out vec4 color;

// Trails image (16-bit float, can go above 1 where many boids overlap)
uniform sampler2D trails;
uniform vec2 windowSize;

const vec3 background = vec3(0.125,0.125,0.125);

void main()
{
    vec3 pixel = texture(trails, gl_FragCoord.xy / windowSize).rgb;
    // Light added by the boids, compressed (Reinhard tone mapping: x / (1 + x)) so that
    // dense flocks keep their gradations instead of all turning white
    vec3 light = max(pixel - background, 0.0);
    color = vec4(background + (1.0 - background) * light / (1.0 + light), 1.0);
}
