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
        self.cohesionFriends = pg.math.Vector2(0,0)
        self.alignmentFriends = pg.math.Vector2(0,0)
        self.separationFriends = pg.math.Vector2(0,0)
        self.cohesionStrangers = pg.math.Vector2(0,0)
        self.separationStrangers = pg.math.Vector2(0,0)
        self.cohesionPredators = pg.math.Vector2(0,0)
        self.separationPredators = pg.math.Vector2(0,0)
    
    def update(self):
        self.getNewValues()
        self.adjustDirection()
        self.checkWalls()
        self.clampSpeed()
        self.position += self.direction * self.speed
        self.updateGridCell()
        # self.wrapAround()
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

    def isNeighbor(self, a):
        tileValues = {-1, 0, 1}
        b=0
        for i in tileValues:
            for j in tileValues:
                b += 1
                addValue = pg.Vector2((i * self.settings.WIDTH),(j * self.settings.HEIGHT))
                if (a + addValue).distance_to(self.position) < 100:
                    return True
        return False

    def adjustDirection(self):
        # If it's a normal boid
        if self.species != util.Species.PREDATOR:
            self.direction += self.alignmentFriends / 10
            self.direction += self.cohesionFriends / 1000
            self.direction += self.separationFriends
            self.direction -= self.cohesionStrangers / 100
            self.direction += self.separationStrangers *10
            self.direction -= self.cohesionPredators
            self.direction += self.separationPredators *10
        # If it's a predator
        else:
            self.direction += self.cohesionStrangers / 10

    def clampSpeed(self):
        if self.direction.length() > 1.0:
            self.direction.normalize_ip()
        elif self.direction.length() < 0.5:
            self.direction.normalize_ip()
            self.direction /= 2

    def checkWalls(self):
        if self.position[0] > (self.settings.WIDTH - 200):
            self.direction[0] -= ((self.position[0] - (self.settings.WIDTH - 200)) * 0.001) ** 2
        elif self.position[0] < 200:
            self.direction[0] += ((200 - self.position[0]) * 0.001) ** 2
        if self.position[1] > (self.settings.HEIGHT - 200):
            self.direction[1] -= ((self.position[1] - (self.settings.HEIGHT - 200)) * 0.001) ** 2
        elif self.position[1] < 200:
            self.direction[1] += ((200 - self.position[1]) * 0.001) ** 2

    def getNewValues(self):
        # friends, strangers, predators = self.getNeighbors()
        friends, strangers, predators = grid.getCellNeighbors(self.index, self.cell)
        if len(friends) > 0:
            self.alignmentFriends = self.alignment(friends)
            self.cohesionFriends = self.cohesion(friends)
            self.separationFriends = self.separation(friends) 
        if len(strangers) > 0:
            self.cohesionStrangers = self.cohesion(strangers)
            self.separationtrangers = self.separation(strangers)
        if len(predators) > 0:
            self.cohesionPredators = self.cohesion(predators)
            self.separationPredators = self.separation(predators)

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

    def getNeighbors(self):
        friends, strangers, predators = [], [], []
        for agent in self.groups()[1].sprites():
            # if agent != self and self.isNeighbor(agent.position):
            if agent != self and agent.position.distance_to(self.position) < 100:
                if agent.species == self.species:
                    friends.append(agent)
                elif agent.species == util.Species.PREDATOR:
                    predators.append(agent)
                else:
                    strangers.append(agent)
        return friends, strangers, predators


