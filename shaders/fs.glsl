#version 460

out vec4 color;
layout (location = 0) in vec4 inColor;

void main()
{
    color = inColor;
}