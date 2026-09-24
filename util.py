from __future__ import annotations

import ctypes
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cuda.bindings.runtime as cudart
import cupy as cp
import glfw  # before imgui_bundle, so it uses this GLFW and not the one it bundles
import numpy as np
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from numba.core.errors import NumbaPerformanceWarning
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_RGBA16F,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_VERTEX_SHADER,
    glBindBuffer,
    glBindFramebuffer,
    glBindTexture,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
    glCheckFramebufferStatus,
    glClear,
    glClearColor,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glFramebufferTexture2D,
    glGenBuffers,
    glGenFramebuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetUniformLocation,
    glTexParameteri,
    glTexStorage2D,
    glUniform1f,
    glUniform1i,
    glUniform2f,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)
from OpenGL.GL.shaders import compileProgram, compileShader

if TYPE_CHECKING:
    from glfw import _GLFWwindowPointerT as GlfwWindow

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
    SEPARATION_FALLOFF: float = 1.0
    BOID_INTENSITY: float = 2.0
    JITTER: float = 0.0
    WANDER: float = 0.0
    WANDER_RATE: float = 10.0

OVERLAY = np.array((
            -1.0,-1.0,
            3.0,-1.0,
            -1.0,3.0), dtype=np.float32)

# Raises on CUDA errors, returns the other values of a cuda-bindings call
def checkCuda(result: tuple[Any, ...]) -> list[Any]:
    err, *values = result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err.name}")
    return values

# onKey(key) is called for key presses that the GUI doesn't use
def init(glfwWindow: GlfwWindow, params: Parameters, onKey: Callable[[int], None]) -> None:
    global window, shader, bgShader, impl, parameters, bgVao, bgVbo, boidVao, boidVbo, boidResource
    global intensityLocation, trailsFramebuffer, trailsTexture, trailsSize, presentShader, windowSizeLocation
    window = glfwWindow
    parameters = params
    # Enable alpha blending (for fading trails)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable( GL_BLEND )
    # Trails image: the simulation is drawn in a 16-bit float texture, then copied to the window.
    # In the window's 8-bit colors, the fade got stuck a few levels above the background
    # (rounding), and the trails relied on the window keeping its previous image, which OpenGL
    # doesn't guarantee
    trailsSize = glfw.get_framebuffer_size(window)
    trailsTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, trailsTexture)
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, *trailsSize)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    trailsFramebuffer = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, trailsFramebuffer)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, trailsTexture, 0)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("Could not create the trails framebuffer")
    glClearColor(0.125,0.125,0.125,1)
    glClear(GL_COLOR_BUFFER_BIT)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
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
    intensityLocation = glGetUniformLocation(shader, "boidIntensity")
    # Draws the trails image in the window, tone mapped
    with open("shaders/present_fs.glsl", 'r') as f:
        fShader = f.readlines()
    presentShader = compileProgram(
        compileShader(vShader, GL_VERTEX_SHADER),
        compileShader(fShader, GL_FRAGMENT_SHADER))
    windowSizeLocation = glGetUniformLocation(presentShader, "windowSize")
    glUseProgram(presentShader)
    glUniform1i(glGetUniformLocation(presentShader, "trails"), 0)
    glUseProgram(0)
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
    _ = imgui.create_context()
    impl = GlfwRenderer(window)
    # Replaces the renderer's key callback, so forward keys to it first
    def keyCallback(window: GlfwWindow, key: int, scancode: int, action: int, mods: int) -> None:
        impl.keyboard_callback(window, key, scancode, action, mods)
        if action == glfw.PRESS and not imgui.get_io().want_capture_keyboard:
            onKey(key)
    _ = glfw.set_key_callback(window, keyCallback)

# Maps the boid buffer for CUDA and yields it as a (POPULATION, 3) CuPy array.
# Map/unmap run on the given stream, so work launched on it inside the block
# finishes before OpenGL reads the buffer again
@contextmanager
def mappedBoidBuffer(stream: cp.cuda.Stream) -> Generator[cp.ndarray, None, None]:
    _ = checkCuda(cudart.cudaGraphicsMapResources(1, boidResource, stream.ptr))
    try:
        pointer, size = checkCuda(cudart.cudaGraphicsResourceGetMappedPointer(boidResource))
        memory = cp.cuda.UnownedMemory(pointer, size, boidResource)
        yield cp.ndarray((parameters.POPULATION, 3), dtype=cp.float32, memptr=cp.cuda.MemoryPointer(memory, 0))
    finally:
        _ = checkCuda(cudart.cudaGraphicsUnmapResources(1, boidResource, stream.ptr))

def shutdown() -> None:
    _ = checkCuda(cudart.cudaGraphicsUnregisterResource(boidResource))
    impl.shutdown()

def renderSim() -> None:
    # Draw in the trails image
    glBindFramebuffer(GL_FRAMEBUFFER, trailsFramebuffer)
    glViewport(0, 0, *trailsSize)
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
    # Boids, added to the pixel (a fraction of their color): overlapping boids are brighter
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    glUseProgram(shader)
    glUniform1f(intensityLocation, parameters.BOID_INTENSITY)
    glBindVertexArray(boidVao)
    glDrawArrays(GL_POINTS, 0, parameters.POPULATION)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    # Draw the trails image in the window (full screen triangle, opaque)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    width, height = glfw.get_framebuffer_size(window)
    glViewport(0, 0, width, height)
    glUseProgram(presentShader)
    glUniform2f(windowSizeLocation, width, height)
    glBindTexture(GL_TEXTURE_2D, trailsTexture)
    glBindVertexArray(bgVao)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindVertexArray(0)
    glUseProgram(0)

def renderGUI(frameTime: float) -> None:
    global parameters
    impl.process_inputs()
    imgui.new_frame()
    imgui.set_next_window_size(imgui.ImVec2(600, 380))
    imgui.set_next_window_pos(imgui.ImVec2(0, 0))
    _ = imgui.begin("Custom window", flags=imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_background)
    _,parameters.COHESION = imgui.slider_float('COHESION', parameters.COHESION, 0.0, 100.0, '%.2f', imgui.SliderFlags_.logarithmic)
    _,parameters.ALIGNMENT = imgui.slider_float('ALIGNMENT', parameters.ALIGNMENT, 0.0, 100.0, '%.2f', imgui.SliderFlags_.logarithmic)
    _,parameters.SEPARATION = imgui.slider_float('SEPARATION', parameters.SEPARATION, 0.0, 100.0, '%.2f', imgui.SliderFlags_.logarithmic)
    _,separationPercent = imgui.slider_int('NEIGHBOR/SEPARATION RATIO', round(parameters.SEPARATION_DIST*100), 0, 100, '%d%%')
    parameters.SEPARATION_DIST = separationPercent / 100
    _,parameters.SEPARATION_FALLOFF = imgui.slider_float('SEPARATION FALLOFF', parameters.SEPARATION_FALLOFF, 0.0, 4.0, '%.2f')
    _,parameters.SPEED = imgui.slider_float('SPEED', parameters.SPEED, 0.0, 10.0, '%.2f')
    _,parameters.JITTER = imgui.slider_float('JITTER', parameters.JITTER, 0.0, 1.0, '%.3f', imgui.SliderFlags_.logarithmic)
    _,parameters.WANDER = imgui.slider_float('WANDER', parameters.WANDER, 0.0, 1.0, '%.3f', imgui.SliderFlags_.logarithmic)
    _,parameters.WANDER_RATE = imgui.slider_float('WANDER RATE', parameters.WANDER_RATE, 0.0, 60.0, '%.1f deg')
    _,parameters.BOID_INTENSITY = imgui.slider_float('BOID INTENSITY', parameters.BOID_INTENSITY, 0.1, 5.0, '%.2f', imgui.SliderFlags_.logarithmic)
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
