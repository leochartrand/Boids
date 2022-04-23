#version 460

layout (location=0) in vec2 vPos;
layout (location=1) in vec3 vColor;

out vec3 fragColor;

void main()
{
    gl_Position = vec4(vPos, 0.0, 1.0);
    fragColor = vColor;
}
