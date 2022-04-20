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
        for index in range(100):
            Boids.add(boids.Boid(util.Species.CYAN, settings, index))
        Agents.add(Boids.sprites())
        return Agents

    if index == type.TWO:
        Boids = pygame.sprite.Group()
        for index in range(50):
            Boids.add(boids.Boid(util.Species.GREEN, settings, index))
        Agents.add(Boids.sprites())
        Boids2 = pygame.sprite.Group()
        for index in range(50, 100):
            Boids2.add(boids.Boid(util.Species.ORANGE, settings, index))
        Agents.add(Boids2.sprites())
        return Agents

    if index == type.HUNT:
        Boids = pygame.sprite.Group()
        for index in range(100):
            Boids.add(boids.Boid(util.Species.CYAN, settings, index))
        Agents.add(Boids.sprites())
        Preds = pygame.sprite.Group()
        for index in range(100, 105):
            Preds.add(boids.Boid(util.Species.PREDATOR, settings, index))
        Agents.add(Preds.sprites())
        return Agents

    if index == type.FEAST:
        Boids = pygame.sprite.Group()
        for index in range(50):
            Boids.add(boids.Boid(util.Species.GREEN, settings, index))
        Agents.add(Boids.sprites())
        Boids2 = pygame.sprite.Group()
        for index in range(50, 100):
            Boids2.add(boids.Boid(util.Species.ORANGE, settings, index))
        Agents.add(Boids2.sprites())
        Preds = pygame.sprite.Group()
        for index in range(100, 110):
            Preds.add(boids.Boid(util.Species.PREDATOR, settings, index))
        Agents.add(Preds.sprites())
        return Agents

    if index == type.BIG_CLASSIC:
        Boids = pygame.sprite.Group()
        for index in range(300):
            Boids.add(boids.Boid(util.Species.CYAN, settings, index))
        Agents.add(Boids.sprites())
        return Agents
            
    if index == type.THOUSAND:
        Boids = pygame.sprite.Group()
        for index in range(1000):
            Boids.add(boids.Boid(util.Species.CYAN, settings, index))
        Agents.add(Boids.sprites())
        return Agents

