from dataclasses import dataclass
from OpenGL.GL import *
import numpy as np
from OpenGL.GL.shaders import compileProgram, compileShader
from numba.core.errors import NumbaPerformanceWarning
import warnings

from numpy import float32

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

def initShader():
    with open("shaders/vs.glsl", 'r') as f:
        vShader = f.readlines()
    with open("shaders/fs.glsl", 'r') as f:
        fShader = f.readlines()
    shader = compileProgram(
        compileShader(vShader, GL_VERTEX_SHADER),
        compileShader(fShader, GL_FRAGMENT_SHADER))
    return shader

def initBGShader():
    with open("shaders/bg_vs.glsl", 'r') as f:
        vShader = f.readlines()
    with open("shaders/bg_fs.glsl", 'r') as f:
        fShader = f.readlines()
    shader = compileProgram(
        compileShader(vShader, GL_VERTEX_SHADER),
        compileShader(fShader, GL_FRAGMENT_SHADER))
    return shader

background = np.array((
            -1.0,-1.0,
            -1.0,1.0,
            1.0,1.0,
            1.0,-1.0), dtype=float32)

