#version 460

layout (location=0) in vec3 data;

layout (location = 0) out vec4 outColor;

void main()
{
    gl_Position = vec4(data.x, data.y, 0.0, 1.0);
    // Spotlight
    if (data.z == 1.0)
    {
        outColor = vec4(1.0,1.0,1.0,1.0);
    }
    // Separation radius
    else if (data.z == 2.0)
    {
        outColor = vec4(1.0,0.4,0.2,1.0);
    }
    // Neighbor radius
    else if (data.z == 3.0)
    {
        outColor = vec4(0.7,0.9,0.3,1.0);
    }
    else if (data.z == 4.0)
    // Neighbor cells
    {
        outColor = vec4(0.9,1.0,0.8,1.0);
    }
    // Normal boid color
    else // (data.z == 0.0)
    {
        outColor = vec4(0.6,0.9,1.0,1.0);
    }
}
