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
WRAP_AROUND = False
SPOTLIGHT = False
FADING_TRAILS = True
# Max agents tested per neighbor search. Above it, a random sample is tested:
# much faster in dense clumps, but separation gets noisier and weaker there
CANDIDATE_BUDGET = 2048
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
window = glfw.create_window(WIDTH, HEIGHT, "Boids", None, None)
if not window:
    glfw.terminate()
    raise SystemExit("Could not create an OpenGL 4.6 window")
glfw.make_context_current(window)
# VSync
glfw.swap_interval(1)

showGUI = True
def onKey(key):
    global showGUI
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    elif key == glfw.KEY_SPACE:
        showGUI = not showGUI

util.init(window, util.Parameters(POPULATION, SPEED, COHESION, ALIGNMENT, SEPARATION,
    NEIGHBOR_DIST, SEPARATION_DIST, WIDTH, HEIGHT, WRAP_AROUND, SPOTLIGHT, FADING_TRAILS, CANDIDATE_BUDGET, SEPARATION_FALLOFF), onKey)
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
        cp.update(util.parameters, renderData, frameTime)
    util.renderSim()
    if showGUI:
        util.renderGUI(frameTime * 1000)
    glfw.swap_buffers(window)
# Quits when the window is closed
util.shutdown()
glfw.terminate()
