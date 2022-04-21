import pygame as pg
import compute as co
import util

pg.init()

# PyGame parameters
##############################################################################
WIDTH = 3500
HEIGHT = 2000
SPEED = 3
PARALLEL = 0
gameSettings = util.Settings(WIDTH, HEIGHT, SPEED, PARALLEL)
##############################################################################
SCREEN_SIZE = (gameSettings.WIDTH, gameSettings.HEIGHT)
BACKGROUND_COLOR = (30,30,30)
gameGlock = pg.time.Clock()
pg.display.set_caption("CUDA Boids")
screen = pg.display.set_mode(SCREEN_SIZE)
##############################################################################

# Main sprite group
Agents = util.getScenario(util.scenario.CLASSIC)

co.init(Agents, gameSettings)

screen.fill(BACKGROUND_COLOR)
# Fading trails
bg = pg.Surface((WIDTH,HEIGHT))
bg.set_alpha(64)
bg.fill(BACKGROUND_COLOR)

# Main loop
# Based on official docs:
# https://www.pygame.org/docs/ref/draw.html
done = False
while not done:
    for event in pg.event.get(): 
        if event.type == pg.QUIT or event.type == pg.K_ESCAPE: 
              done = True 

    tickTime = gameGlock.tick()
    print(tickTime)

    # For pretty fading trails
    screen.blit(bg, (0,0))
    
    co.update()
    Agents.update()
    Agents.draw(screen)
              
    pg.display.flip()
# Quits when game is over
pg.quit()
