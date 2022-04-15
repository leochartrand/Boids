import util
from enum import Enum
import pygame
import boids

class type(Enum):
    CLASSIC = 0
    TWO = 1
    PREDATOR = 2
    MAYHEM = 3
    BIG_CLASSIC = 4

def getScenario(index, settings):
    Agents = pygame.sprite.Group()
    if index == type.CLASSIC:
        Boids = pygame.sprite.Group()
        for _ in range(100):
            Boids.add(boids.Boid(util.Species.CYAN, settings))
        Agents.add(Boids.sprites())
        return Agents

    if index == type.TWO:
        Boids = pygame.sprite.Group()
        for _ in range(50):
            Boids.add(boids.Boid(util.Species.GREEN, settings))
        Agents.add(Boids.sprites())
        Boids2 = pygame.sprite.Group()
        for _ in range(50):
            Boids2.add(boids.Boid(util.Species.ORANGE, settings))
        Agents.add(Boids2.sprites())
        return Agents

    if index == type.PREDATOR:
        Boids = pygame.sprite.Group()
        for _ in range(100):
            Boids.add(boids.Boid(util.Species.CYAN, settings))
        Agents.add(Boids.sprites())
        Preds = pygame.sprite.Group()
        for _ in range(10):
            Preds.add(boids.Boid(util.Species.PREDATOR, settings))
        Agents.add(Preds.sprites())
        return Agents

    if index == type.MAYHEM:
        Boids = pygame.sprite.Group()
        for _ in range(50):
            Boids.add(boids.Boid(util.Species.GREEN, settings))
        Agents.add(Boids.sprites())
        Boids2 = pygame.sprite.Group()
        for _ in range(50):
            Boids2.add(boids.Boid(util.Species.ORANGE, settings))
        Agents.add(Boids2.sprites())
        Preds = pygame.sprite.Group()
        for _ in range(10):
            Preds.add(boids.Boid(util.Species.PREDATOR, settings))
        Agents.add(Preds.sprites())
        return Agents

    if index == type.BIG_CLASSIC:
        Boids = pygame.sprite.Group()
        for _ in range(300):
            Boids.add(boids.Boid(util.Species.CYAN, settings))
        Agents.add(Boids.sprites())
        return Agents

