import util
from enum import Enum
import pygame
import boids

class type(Enum):
    CLASSIC = 0
    TWO = 1
    HUNT = 2
    FEAST = 3
    BIG_CLASSIC = 4
    THOUSAND = 5

def getScenario(index, settings):
    Agents = pygame.sprite.Group()
    if index == type.CLASSIC:
        Boids = pygame.sprite.Group()
        for index in range(1000):
            Boids.add(boids.Boid(util.Species.CYAN, settings, index))
        Agents.add(Boids.sprites())
        return Agents

    if index == type.TWO:
        Boids = pygame.sprite.Group()
        for index in range(500):
            Boids.add(boids.Boid(util.Species.GREEN, settings, index))
        Agents.add(Boids.sprites())
        Boids2 = pygame.sprite.Group()
        for index in range(500, 1000):
            Boids2.add(boids.Boid(util.Species.ORANGE, settings, index))
        Agents.add(Boids2.sprites())
        return Agents

