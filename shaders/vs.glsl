#version 460

layout (location=0) in vec3 data;

layout (location = 0) out vec4 outColor;

// Light a normal boid adds to its pixel (boids are drawn additively)
uniform float boidIntensity;

void main()
{
    gl_Position = vec4(data.x, data.y, 0.0, 1.0);
    // Spotlight colors are brighter than normal boids, so they stand out
    float spotlightIntensity = 2.0 * boidIntensity;
    // Spotlight
    if (data.z == 1.0)
    {
        outColor = vec4(1.0,1.0,1.0,spotlightIntensity);
    }
    // Separation radius
    else if (data.z == 2.0)
    {
        outColor = vec4(1.0,0.4,0.2,spotlightIntensity);
    }
    // Neighbor radius
    else if (data.z == 3.0)
    {
        outColor = vec4(0.7,0.9,0.3,spotlightIntensity);
    }
    else if (data.z == 4.0)
    // Neighbor cells
    {
        outColor = vec4(0.9,1.0,0.8,spotlightIntensity);
    }
    // Normal boid color
    else // (data.z == 0.0)
    {
        outColor = vec4(0.6,0.9,1.0,boidIntensity);
    }
}
