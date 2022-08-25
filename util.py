from dataclasses import dataclass
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
from numba.core.errors import NumbaPerformanceWarning
import imgui
from imgui.integrations.pygame import PygameRenderer
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

def init(size, params):
    global shader, bgShader, impl, parameters
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
    imgui.create_context()
    impl = PygameRenderer()
    io = imgui.get_io()
    io.display_size = size

overlay = np.array((
            -1.0,-1.0,
            3.0,-1.0,
            -1.0,3.0), dtype=np.float32)

def renderSim(buffer):
    # Render overlay - semi transparent overlay
    # for fading trails
    glUseProgram(bgShader)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, overlay.nbytes, overlay, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    # Boids
    glUseProgram(shader)
    array = np.copy(buffer)
    array.flatten()
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, array.nbytes, array, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glBindVertexArray(vao)
    glDrawArrays(GL_POINTS, 0, parameters.POPULATION)

def renderGUI(tickTime):
    global parameters
    imgui.new_frame()
    imgui.set_next_window_size(600, 300)
    imgui.set_next_window_position(0, 0)
    imgui.begin("Custom window", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_BACKGROUND)
    _,parameters.COHESION = imgui.core.slider_float('COHESION', parameters.COHESION, 0.0, 100.0, '%.2f', imgui.SLIDER_FLAGS_LOGARITHMIC)
    _,parameters.ALIGNMENT = imgui.core.slider_float('ALIGNMENT', parameters.ALIGNMENT, 0.0, 100.0, '%.2f', imgui.SLIDER_FLAGS_LOGARITHMIC)
    _,parameters.SEPARATION = imgui.core.slider_float('SEPARATION', parameters.SEPARATION, 0.0, 100.0, '%.2f', imgui.SLIDER_FLAGS_LOGARITHMIC)
    _,parameters.SEPARATION_DIST = imgui.slider_int('NEIGHBOR/SEPARATION RATIO', int(parameters.SEPARATION_DIST*100), 0, 100, '%d%%')
    parameters.SEPARATION_DIST /= 100
    _,parameters.SPEED = imgui.slider_float('SPEED', parameters.SPEED, 0.0, 10.0, '%.2f', 1.0)
    if imgui.radio_button("WRAP AROUND", parameters.WRAP_AROUND):
        parameters.WRAP_AROUND = not parameters.WRAP_AROUND
    imgui.same_line(spacing=50)
    if imgui.radio_button("SPOTLIGHT", parameters.SPOTLIGHT):
        parameters.SPOTLIGHT = not parameters.SPOTLIGHT
    imgui.same_line(spacing=50)
    imgui.text(f"ms/frame: {tickTime}")
    imgui.push_style_var(imgui.STYLE_ALPHA, 0.2)
    imgui.text("(Press SPACE to hide)")
    imgui.pop_style_var(1)
    imgui.end()
    imgui.end_frame()
    imgui.render()
    impl.render(imgui.get_draw_data())

def processEvents(event):
    impl.process_event(event)

def backupGLState():
    global last_program, last_vertex_array, last_array_buffer, last_element_array_buffer
    last_program = glGetIntegerv(GL_CURRENT_PROGRAM)
    last_array_buffer = glGetIntegerv(GL_ARRAY_BUFFER_BINDING)
    last_element_array_buffer = glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING)
    last_vertex_array = glGetIntegerv(GL_VERTEX_ARRAY_BINDING)

def restoreGLState():
    glUseProgram(last_program)
    glBindVertexArray(last_vertex_array)
    glBindBuffer(GL_ARRAY_BUFFER, last_array_buffer)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, last_element_array_buffer)
