from ctypes import alignment
import pygame
import random
import math
from scipy.stats import circmean
import util

class Boid (pygame.sprite.Sprite):
    def __init__(self, species, settings):
        super().__init__()
        self.settings = settings
        self.position = pygame.math.Vector2(random.randint(0, settings.WIDTH),random.randint(0, settings.HEIGHT))
        self.direction = pygame.math.Vector2(1,0).rotate(random.randint(0,360))
        self.speed = settings.SPEED
        self.image = util.getSpeciesImage(species)
        self.rect = self.image.get_rect()
        self.imgBase = pygame.transform.rotate(self.image, -90) #used for updating (rotation)
        self.species = species

    def updateImage(self):
        self.rect.center = self.position
        angle = pygame.math.Vector2(1,0).angle_to(self.direction)
        self.image = pygame.transform.rotate(self.imgBase, -angle)
    
    def wrapAround(self):
        if self.position[0] > self.settings.WIDTH:
            self.position[0] = 0
        elif self.position[0] < 0 :
            self.position[0] = self.settings.WIDTH
        if self.position[1] > self.settings.HEIGHT:
            self.position[1] = 0
        elif self.position[1] < 0 :
            self.position[1] = self.settings.HEIGHT
    
    def update(self):
        self.adjustDirection()
        self.position += self.direction * self.speed
        self.wrapAround()
        self.updateImage()

    def adjustDirection(self):
        friends, strangers, predators = self.getNeighbors()
        if self.species != util.Species.PREDATOR:
            if len(friends) > 0:
                self.direction += self.alignment(friends) / 10
                self.direction += self.cohesion(friends) / 1000
                self.direction += self.separation(friends) 
            if len(strangers) > 0:
                self.direction -= self.cohesion(strangers) / 100
                self.direction += self.separation(strangers) *10
            if len(predators) > 0:
                self.direction -= self.cohesion(predators) 
                self.direction += self.separation(predators) *10
        else:
            if len(strangers) > 0:
                self.direction += self.cohesion(strangers) / 10

        self.clampSpeed()

    def clampSpeed(self):
        if self.direction.length() > 1.0:
            self.direction.normalize_ip()
        elif self.direction.length() < 0.5:
            self.direction.normalize_ip()
            self.direction /= 2

    def cohesion(self, neighbors):
        cohesionDir = pygame.math.Vector2(0,0)
        for agent in neighbors:
            cohesionDir += agent.position
        cohesionDir /= len(neighbors)
        cohesionDir -= self.position
        cohesionDir.normalize_ip()
        return cohesionDir

    def alignment(self, neighbors):
        alignmentVec = pygame.math.Vector2(0,0)
        for agent in neighbors:
            alignmentVec += agent.direction
        alignmentVec /= len(neighbors)
        return alignmentVec

    def separation(self, neighbors):
        separationPos = pygame.math.Vector2(0,0)
        numSeparation = 0
        for agent in neighbors:
            dist = (self.position - agent.position)
            sqrt = dist.length_squared()
            if dist.length() < 40:
                if dist.length() > 0:
                    dist /= sqrt
                separationPos += dist
                numSeparation += 1
        if numSeparation > 0:
            separationPos /= numSeparation
        return separationPos

    def getNeighbors(self):
        friends, strangers, predators = [], [], []
        for agent in self.groups()[1].sprites():
            if agent != self and agent.position.distance_to(self.position) < 100:
                if agent.species == self.species:
                    friends.append(agent)
                elif agent.species == util.Species.PREDATOR:
                    predators.append(agent)
                else:
                    strangers.append(agent)
        return friends, strangers, predators




