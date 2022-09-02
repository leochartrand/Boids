from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame as pg
import compute as cp
import util
##############################################################################
# Simulation parameters
POPULATION = 100000
SPEED = 2.0
COHESION = 0.01
ALIGNMENT = 0.2
SEPARATION = 0.1
NEIGHBOR_DIST = 50
SEPARATION_DIST = 0.4 # % of NEIGHBOR_DIST
WRAP_AROUND = False
SPOTLIGHT = False
FADING_TRAILS = True
# Window size
WIDTH  = (NEIGHBOR_DIST) * 36*2
HEIGHT = (NEIGHBOR_DIST) * 20*2
##############################################################################
# Initialisation
pg.init()
SCREEN_SIZE = (WIDTH,HEIGHT)
clock = pg.time.Clock()
pg.display.set_caption("Boids")
screen = pg.display.set_mode(SCREEN_SIZE, pg.OPENGL|pg.DOUBLEBUF)
util.init(SCREEN_SIZE, util.Parameters(POPULATION, SPEED, COHESION, ALIGNMENT, SEPARATION,
    NEIGHBOR_DIST, SEPARATION_DIST, WIDTH, HEIGHT, WRAP_AROUND, SPOTLIGHT, FADING_TRAILS))
cp.init(util.parameters)
##############################################################################
# Simulation loop
showGUI = True
done = False
while not done:
    for event in pg.event.get(): 
        if event.type == pg.QUIT or event.type == pg.K_ESCAPE: 
              done = True 
        if event.type == 768:
            showGUI = not showGUI
        util.processEvents(event)
    util.backupGLState()
    cp.update(util.parameters)
    util.renderSim(cp.renderData)
    tickTime = clock.tick()
    util.restoreGLState()
    if showGUI:
        util.renderGUI(tickTime)
    pg.display.flip()
# Quits when game is over
pg.quit()
