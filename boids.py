import pygame as pg
import parallel as par
import sequential as seq
import util
from OpenGL.GL import *

pg.init()

##############################################################################
# Simulation parameters
PARALLEL = True
POPULATION = 100000
SPEED = 2
COHESION = 0.01
ALIGNMENT = 0.2
SEPARATION = 0.1
NEIGHBOR_DIST = 100 # Ideally kept at 100
SEPARATION_DIST = 50
WRAP_AROUND = True
# Window size
WIDTH  = (NEIGHBOR_DIST) * 36
HEIGHT = (NEIGHBOR_DIST) * 20
#
gameSettings = util.Settings(POPULATION, SPEED, COHESION, ALIGNMENT, 
    SEPARATION, NEIGHBOR_DIST, SEPARATION_DIST, WIDTH, HEIGHT, WRAP_AROUND)
##############################################################################
# PyGame/OpenGL parameters
SCREEN_SIZE = (gameSettings.WIDTH, gameSettings.HEIGHT)
gameGlock = pg.time.Clock()
pg.display.set_caption("CUDA Boids")
screen = pg.display.set_mode(SCREEN_SIZE, pg.OPENGL|pg.DOUBLEBUF)
util.init(POPULATION)
render = util.render
##############################################################################

if PARALLEL:
    par.init(gameSettings)
    renderData = par.renderData
    update = par.update
else:
    seq.init(gameSettings)
    renderData = seq.renderData
    update = seq.update

done = False
while not done:
    for event in pg.event.get(): 
        if event.type == pg.QUIT or event.type == pg.K_ESCAPE: 
              done = True 

    update()

    util.render(renderData)

    tickTime = gameGlock.tick()
    # print(tickTime)

    pg.display.flip()
# Quits when game is over
pg.quit()
