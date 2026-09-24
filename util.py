from contextlib import contextmanager
from dataclasses import dataclass
from OpenGL.GL import *
import numpy as np
import cupy as cp
from cuda.bindings import runtime as cudart
from OpenGL.GL.shaders import compileProgram, compileShader
from numba.core.errors import NumbaPerformanceWarning
import glfw # before imgui_bundle, so it uses this GLFW and not the one it bundles
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

@dataclass
class Parameters:
    POPULATION      : int
    SPEED           : float
    COHESION        : float
    ALIGNMENT       : float
    SEPARATION      : float
    NEIGHBOR_DIST   : int
    SEPARATION_DIST : float
    WIDTH           : int
    HEIGHT          : int
    WRAP_AROUND     : bool
    SPOTLIGHT       : bool
    FADING_TRAILS   : bool
    CANDIDATE_BUDGET: int = 2048

OVERLAY = np.array((
            -1.0,-1.0,
            3.0,-1.0,
            -1.0,3.0), dtype=np.float32)

# Raises on CUDA errors, returns the other values of a cuda-bindings call
def checkCuda(result):
    err, *values = result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err.name}")
    return values

# onKey(key) is called for key presses that the GUI doesn't use
def init(glfwWindow, params, onKey):
    global window, shader, bgShader, impl, parameters, bgVao, bgVbo, boidVao, boidVbo, boidResource
    window = glfwWindow
    parameters = params
    # Enable alpha blending (for fading trails)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable( GL_BLEND )
    glClearColor(0.125,0.125,0.125,1)
    glClear(GL_COLOR_BUFFER_BIT)
    # Init shaders
    with open("shaders/vs.glsl", 'r') as f:
        vShader = f.readlines()
    with open("shaders/fs.glsl", 'r') as f:
        fShader = f.readlines()
    shader = compileProgram(
        compileShader(vShader, GL_VERTEX_SHADER),
        compileShader(fShader, GL_FRAGMENT_SHADER))
    with open("shaders/bg_fs.glsl", 'r') as f:
        fShader = f.readlines()
    bgShader = compileProgram(
        compileShader(vShader, GL_VERTEX_SHADER),
        compileShader(fShader, GL_FRAGMENT_SHADER))
    # Init overlay (trails)
    bgVao = glGenVertexArrays(1)
    glBindVertexArray(bgVao)
    bgVbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, bgVbo)
    glBufferData(GL_ARRAY_BUFFER, OVERLAY.nbytes, OVERLAY, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
    # Init Boids
    boidVao = glGenVertexArrays(1)
    glBindVertexArray(boidVao)
    boidVbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, boidVbo)
    glBufferData(GL_ARRAY_BUFFER, parameters.POPULATION * 3 * 4, None, GL_DYNAMIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    # Register the boid buffer with CUDA so kernels can write to it directly.
    # WriteDiscard: every frame overwrites all of it, so CUDA never needs its old content
    boidResource, = checkCuda(cudart.cudaGraphicsGLRegisterBuffer(
        int(boidVbo), cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))

    # GUI
    imgui.create_context()
    impl = GlfwRenderer(window)
    # Replaces the renderer's key callback, so forward keys to it first
    def keyCallback(window, key, scancode, action, mods):
        impl.keyboard_callback(window, key, scancode, action, mods)
        if action == glfw.PRESS and not imgui.get_io().want_capture_keyboard:
            onKey(key)
    glfw.set_key_callback(window, keyCallback)

# Maps the boid buffer for CUDA and yields it as a (POPULATION, 3) CuPy array.
# Map/unmap run on the given stream, so work launched on it inside the block
# finishes before OpenGL reads the buffer again
@contextmanager
def mappedBoidBuffer(stream):
    checkCuda(cudart.cudaGraphicsMapResources(1, boidResource, stream.ptr))
    try:
        pointer, size = checkCuda(cudart.cudaGraphicsResourceGetMappedPointer(boidResource))
        memory = cp.cuda.UnownedMemory(pointer, size, boidResource)
        yield cp.ndarray((parameters.POPULATION, 3), dtype=cp.float32, memptr=cp.cuda.MemoryPointer(memory, 0))
    finally:
        checkCuda(cudart.cudaGraphicsUnmapResources(1, boidResource, stream.ptr))

def shutdown():
    checkCuda(cudart.cudaGraphicsUnregisterResource(boidResource))
    impl.shutdown()

def renderSim():
    glViewport(0, 0, *glfw.get_framebuffer_size(window))
    # Background render
    if parameters.FADING_TRAILS:
        # Render overlay - semi transparent overlay
        # for fading trails
        glUseProgram(bgShader)
        glBindVertexArray(bgVao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
    else:
        glClearColor(0.125,0.125,0.125,1)
        glClear(GL_COLOR_BUFFER_BIT)
    # Boids
    glUseProgram(shader)
    glBindVertexArray(boidVao)
    glDrawArrays(GL_POINTS, 0, parameters.POPULATION)
    glBindVertexArray(0)
    glUseProgram(0)

def renderGUI(frameTime):
    global parameters
    impl.process_inputs()
    imgui.new_frame()
    imgui.set_next_window_size(imgui.ImVec2(600, 300))
    imgui.set_next_window_pos(imgui.ImVec2(0, 0))
    imgui.begin("Custom window", flags=imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_background)
    _,parameters.COHESION = imgui.slider_float('COHESION', parameters.COHESION, 0.0, 100.0, '%.2f', imgui.SliderFlags_.logarithmic)
    _,parameters.ALIGNMENT = imgui.slider_float('ALIGNMENT', parameters.ALIGNMENT, 0.0, 100.0, '%.2f', imgui.SliderFlags_.logarithmic)
    _,parameters.SEPARATION = imgui.slider_float('SEPARATION', parameters.SEPARATION, 0.0, 100.0, '%.2f', imgui.SliderFlags_.logarithmic)
    _,separationPercent = imgui.slider_int('NEIGHBOR/SEPARATION RATIO', round(parameters.SEPARATION_DIST*100), 0, 100, '%d%%')
    parameters.SEPARATION_DIST = separationPercent / 100
    _,parameters.SPEED = imgui.slider_float('SPEED', parameters.SPEED, 0.0, 10.0, '%.2f')
    _,parameters.CANDIDATE_BUDGET = imgui.slider_int('CANDIDATE BUDGET', parameters.CANDIDATE_BUDGET, 16, 16384, '%d', imgui.SliderFlags_.logarithmic)
    _,parameters.WRAP_AROUND = imgui.checkbox("WRAP AROUND", parameters.WRAP_AROUND)
    imgui.same_line(spacing=40)
    _,parameters.SPOTLIGHT = imgui.checkbox("SPOTLIGHT", parameters.SPOTLIGHT)
    imgui.same_line(spacing=45)
    _,parameters.FADING_TRAILS = imgui.checkbox("FADING_TRAILS", parameters.FADING_TRAILS)
    imgui.text(f"ms/frame: {frameTime:.1f}")
    imgui.push_style_var(imgui.StyleVar_.alpha, 0.2)
    imgui.text("(Press SPACE to hide)")
    imgui.pop_style_var(1)
    imgui.end()
    imgui.render()
    impl.render(imgui.get_draw_data())
