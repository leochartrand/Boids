from dataclasses import dataclass
import pygame
from enum import Enum

@dataclass
class Settings:
    WIDTH : int
    HEIGHT: int
    SPEED: int

class Species(Enum):
    GREEN = 0
    ORANGE = 1
    CYAN = 2
    PREDATOR = 3

def getSpeciesImage(index):
    if index == Species.GREEN:
        return pygame.image.load("assets/boid_green.png")
    if index == Species.ORANGE:
        return pygame.image.load("assets/boid_orange.png")
    if index == Species.CYAN:
        return pygame.image.load("assets/boid_cyan.png")
    if index == Species.PREDATOR:
        return pygame.image.load("assets/predator_red.png")


