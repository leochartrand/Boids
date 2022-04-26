from dataclasses import dataclass
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

@dataclass
class Settings:
    POPULATION      : int
    SPEED           : int
    COHESION        : float
    ALIGNMENT       : float
    SEPARATION      : float
    NEIGHBOR_DIST   : int
    SEPARATION_DIST : int
    WIDTH           : int
    HEIGHT          : int
    WRAP_AROUND     : bool

def init(pop):
    global shader, bgShader, popSize
    popSize = pop
    # Enable alpha blending (for fading trails)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable( GL_BLEND )
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


overlay = np.array((
            -1.0,-1.0,
            3.0,-1.0,
            -1.0,3.0), dtype=np.float32)


def render(buffer):
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
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
    glBindVertexArray(vao)
    glDrawArrays(GL_POINTS, 0, popSize)
    glFlush()
