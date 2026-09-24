import glfw

import compute as cp
import util

##############################################################################
# Simulation parameters
POPULATION = 1000000
SPEED = 2.0
COHESION = 0.01
ALIGNMENT = 0.2
SEPARATION = 0.1
NEIGHBOR_DIST = 50
SEPARATION_DIST = 0.4 # % of NEIGHBOR_DIST
# How separation fades with distance: 0 same push at any distance, 1 linear, higher mostly at close range
SEPARATION_FALLOFF = 1.0
# Randomness, weighted like the rules. Jitter: a new random push every 1/60 s (Gaussian)
JITTER = 0.0
# Wander (Reynolds): steering towards a target drifting on a circle ahead of the boid, for smooth, committed turns.
# WANDER_RATE: how fast the target drifts (standard deviation per 1/60 s, in degrees)
WANDER = 0.0
WANDER_RATE = 10.0
# Cursor: hold left click to attract boids (or repel them, with CURSOR_REPEL), right click for the opposite.
# CURSOR_ATTENTION: share of a boid's attention the cursor takes (0 to 1), next to it and fading to 0 at CURSOR_RADIUS pixels
CURSOR_ATTENTION = 0.5
CURSOR_RADIUS = 400.0
CURSOR_REPEL = False
WRAP_AROUND = False
SPOTLIGHT = False
FADING_TRAILS = True
# Light a boid adds to its pixel (like an exposure): lower looks finer, and dense flocks show brighter
BOID_INTENSITY = 2.0
# Max agents tested per neighbor search. Above it, a random sample is tested:
# much faster in dense clumps, but separation gets noisier and weaker there
CANDIDATE_BUDGET = 2048
# Full screen on the largest monitor, or a window of WIDTH x HEIGHT
FULLSCREEN = False
# Window size
WIDTH  = (NEIGHBOR_DIST) * 36*2
HEIGHT = (NEIGHBOR_DIST) * 20*2
##############################################################################
# Initialisation
if not glfw.init():
    raise SystemExit("Could not initialise GLFW")
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.RESIZABLE, False)
if FULLSCREEN:
    # Largest monitor, at its current resolution and refresh rate (no video mode change).
    # The simulated space (WIDTH x HEIGHT) is stretched to fill it
    monitor = max(glfw.get_monitors(), key=lambda m: glfw.get_video_mode(m).size.width * glfw.get_video_mode(m).size.height)
    mode = glfw.get_video_mode(monitor)
    glfw.window_hint(glfw.REFRESH_RATE, mode.refresh_rate)
    # Stay visible when another window gets the focus (e.g. on the other monitor)
    glfw.window_hint(glfw.AUTO_ICONIFY, False)
    window = glfw.create_window(mode.size.width, mode.size.height, "Boids", monitor, None)
else:
    window = glfw.create_window(WIDTH, HEIGHT, "Boids", None, None)
if not window:
    glfw.terminate()
    raise SystemExit("Could not create an OpenGL 4.6 window")
glfw.make_context_current(window)
# VSync
glfw.swap_interval(1)

showGUI = True
def onKey(key: int) -> None:
    global showGUI
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    elif key == glfw.KEY_SPACE:
        showGUI = not showGUI

util.init(window, util.Parameters(POPULATION, SPEED, COHESION, ALIGNMENT, SEPARATION,
    NEIGHBOR_DIST, SEPARATION_DIST, WIDTH, HEIGHT, WRAP_AROUND, SPOTLIGHT, FADING_TRAILS, CANDIDATE_BUDGET, SEPARATION_FALLOFF, BOID_INTENSITY, JITTER, WANDER, WANDER_RATE, CURSOR_ATTENTION, CURSOR_RADIUS, CURSOR_REPEL), onKey)
cp.init(util.parameters)
##############################################################################
# Simulation loop
lastTime = glfw.get_time()
while not glfw.window_should_close(window):
    glfw.poll_events()
    # Time since the last frame, in seconds: agents move according to it
    now = glfw.get_time()
    frameTime = now - lastTime
    lastTime = now
    # Kernels write positions straight into the OpenGL vertex buffer
    with util.mappedBoidBuffer(cp.stream) as renderData:
        cp.update(util.parameters, renderData, frameTime, util.cursorInput())
    util.renderSim()
    if showGUI:
        util.renderGUI(frameTime * 1000)
    glfw.swap_buffers(window)
# Quits when the window is closed
util.shutdown()
glfw.terminate()
