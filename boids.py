import pygame as pg
import random as rd
import util
import compute as co
import grid

class Boid (pg.sprite.Sprite):
    def __init__(self, species, settings, index):
        super().__init__()
        self.index = index
        self.settings = settings
        self.position = pg.math.Vector2(rd.randint(0, settings.WIDTH),rd.randint(0, settings.HEIGHT))
        self.direction = pg.math.Vector2(1,0).rotate(rd.randint(0,360))
        self.cell = grid.getCell(self.position)
        self.speed = settings.SPEED
        self.image = util.getSpeciesImage(species)
        self.rect = self.image.get_rect()
        self.imgBase = pg.transform.scale(pg.transform.rotate(self.image, -90), (10, 10)) #used for updating (rotation)
        self.species = species
    
    def update(self):
        self.adjustDirection()
        self.clampSpeed()
        self.position += self.direction * self.speed
        self.updateGridCell()
        self.wrapAround()
        #
        # self.position, self.direction = co.getPosAndDir(self.index)
        self.rect.center = self.position
        angle = pg.math.Vector2(1,0).angle_to(self.direction)
        self.image = pg.transform.rotate(self.imgBase, -angle)

    def updateGridCell(self):
        newCell = grid.getCell(self.position)
        if self.cell != newCell:
            grid.remove(self.index, self.cell)
            grid.add(self.index, newCell)
            self.cell = newCell
    
    def wrapAround(self):
        if self.position[0] > self.settings.WIDTH:
            self.position[0] = 0
        elif self.position[0] < 0 :
            self.position[0] = self.settings.WIDTH
        if self.position[1] > self.settings.HEIGHT:
            self.position[1] = 0
        elif self.position[1] < 0 :
            self.position[1] = self.settings.HEIGHT

    def adjustDirection(self):
        friends, strangers = grid.getCellNeighbors(self.index, self.cell)
        if len(friends) > 0:
            alignmentFriends = self.alignment(friends)
            cohesionFriends = self.cohesion(friends)
            separationFriends = self.separation(friends) 
            self.direction += alignmentFriends / 10
            self.direction += cohesionFriends / 1000
            self.direction += separationFriends
        if len(strangers) > 0:
            cohesionStrangers = self.cohesion(strangers)
            separationStrangers = self.separation(strangers)
            self.direction -= cohesionStrangers / 100
            self.direction += separationStrangers *10

    def clampSpeed(self):
        if self.direction.length() > 1.0:
            self.direction.normalize_ip()
        elif self.direction.length() < 0.5:
            self.direction.normalize_ip()
            self.direction /= 2

    def cohesion(self, neighbors):
        cohesionDir = pg.math.Vector2(0,0)
        for agent in neighbors:
            cohesionDir += agent.position
        cohesionDir /= len(neighbors)
        cohesionDir -= self.position
        cohesionDir.normalize_ip()
        return cohesionDir

    def alignment(self, neighbors):
        alignmentVec = pg.math.Vector2(0,0)
        for agent in neighbors:
            alignmentVec += agent.direction
        alignmentVec /= len(neighbors)
        alignmentVec.normalize_ip()
        return alignmentVec

    def separation(self, neighbors):
        separationPos = pg.math.Vector2(0,0)
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

