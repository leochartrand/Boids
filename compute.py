import numpy as np
import numba as nb
import random as rd

def init(Agents, settings):
    global WIDTH, HEIGHT, SPEED, PARALLEL, GRID_CELL_SIZE
    WIDTH = settings.WIDTH
    HEIGHT = settings.HEIGHT
    SPEED = settings.SPEED
    PARALLEL = settings.PARALLEL
    GRID_CELL_SIZE = 100
    #
    global boids, numBoids, dataBuffer, tempBuffer, gridBuffer, specBuffer
    boids = Agents.sprites()
    numBoids = len(boids)
    dataBuffer = np.zeros((numBoids, 4), dtype=float)
    tempBuffer = np.zeros((numBoids, 14), dtype=float)
    gridBuffer = np.empty((numBoids, 2), dtype=int)
    specBuffer = np.zeros((numBoids, 1), dtype=int)
    for i in range(numBoids):
        dataBuffer[i,0] = rd.uniform(0, WIDTH)
        dataBuffer[i,1] = rd.uniform(0, HEIGHT)
        tempX, tempY = normalize(rd.random()*2-1, rd.random()*2-1)
        dataBuffer[i,2] = tempX
        dataBuffer[i,3] = tempY
        specBuffer[i] = boids[i].species.value
        gridBuffer[i,0], gridBuffer[i,1] = getCell(dataBuffer[i,0], dataBuffer[i,1])
    # Grid initialization
    global cells, gridW, gridH
    cells = {}
    gridW = WIDTH//GRID_CELL_SIZE
    gridH = HEIGHT//GRID_CELL_SIZE

def getPosAndDir(index):
    return dataBuffer[index,0],dataBuffer[index,1],dataBuffer[index,2],dataBuffer[index,3]

def sendData():
    pass

def retrieveData():
    pass

#######################################
# Update
#######################################
# @nb.jit
def update():
    toTempBuf()
    for i in nb.prange(numBoids):
        updateBoid(i)
    toDataBuf()

# @nb.jit
def toTempBuf():
    global dataBuffer, tempBuffer
    tempBuffer[:,0] = dataBuffer[:,0]
    tempBuffer[:,1] = dataBuffer[:,1]
    tempBuffer[:,2] = dataBuffer[:,2]
    tempBuffer[:,3] = dataBuffer[:,3]

# @nb.jit
def toDataBuf():
    global dataBuffer, tempBuffer
    dataBuffer[:,0] = tempBuffer[:,0]
    dataBuffer[:,1] = tempBuffer[:,1]
    dataBuffer[:,2] = tempBuffer[:,2]
    dataBuffer[:,3] = tempBuffer[:,3]

def updateBoid(index):
    global tempBuffer, gridBuffer
    # Adjust directions
    x, y = 0.0, 0.0
    friends, strangers = getCellNeighbors(index, gridBuffer[index,0], gridBuffer[index,1])
    if len(friends) > 0:
        tempBuffer[index,4], tempBuffer[index,5] = alignment(friends)
        tempBuffer[index,6], tempBuffer[index,7] = cohesion(index,friends)
        tempBuffer[index,8], tempBuffer[index,9] = separation(index,friends)
        # Alignment
        x += tempBuffer[index,4] / 10
        y += tempBuffer[index,5] / 10
        # Cohesion
        x += tempBuffer[index,6] / 1000
        y += tempBuffer[index,7] / 1000
        # Separation
        x += tempBuffer[index,8]
        y += tempBuffer[index,9]
    if len(strangers) > 0:
        tempBuffer[index,10], tempBuffer[index,11] = cohesion(index,strangers)
        tempBuffer[index,12], tempBuffer[index,13] = separation(index,strangers)
        # Cohesion - Strangers
        x -= tempBuffer[index,10] / 50
        y -= tempBuffer[index,11] / 50
        # Separation - Strangers
        x += tempBuffer[index,12] * 10
        y += tempBuffer[index,13] * 10
    # Update directions
    tempBuffer[index,2] += x
    tempBuffer[index,3] += y
    # Clamp speed
    l = length(tempBuffer[index,2], tempBuffer[index,3])
    if l > 1.0:
        tempBuffer[index,2], tempBuffer[index,3] = normalize(tempBuffer[index,2], tempBuffer[index,3])
    elif l < 0.5:
        x, y = normalize(tempBuffer[index,2], tempBuffer[index,3])
        tempBuffer[index,2] = x / 2
        tempBuffer[index,3] = y / 2
    # Move position
    tempBuffer[index,0] += tempBuffer[index,2] * SPEED
    tempBuffer[index,1] += tempBuffer[index,3] * SPEED
    # Edge wrap
    if tempBuffer[index,0] > WIDTH:
        tempBuffer[index,0] = 0
    elif tempBuffer[index,0] < 0 :
        tempBuffer[index,0] = WIDTH
    if tempBuffer[index,1] > HEIGHT:
        tempBuffer[index,1] = 0
    elif tempBuffer[index,1] < 0 :
        tempBuffer[index,1] = HEIGHT
    # Update grid cells
    x, y = getCell(tempBuffer[index,0], tempBuffer[index,1])
    if gridBuffer[index,0] != x or gridBuffer[index,1] != y:
        remove(index, gridBuffer[index,0], gridBuffer[index,1])
        add(index, x, y)
        gridBuffer[index,0], gridBuffer[index,1] = x, y

#######################################
# Boids rules
#######################################
def cohesion(index, neighbors):
    global tempBuffer
    cohesionX, cohesionY = 0.0, 0.0
    for agent in neighbors:
        cohesionX += tempBuffer[agent, 0]
        cohesionY += tempBuffer[agent, 1]
    cohesionX -= tempBuffer[index, 0]
    cohesionY -= tempBuffer[index, 1]
    cohesionX, cohesionY = normalize(cohesionX, cohesionY)
    return cohesionX, cohesionY

def alignment(neighbors):
    global tempBuffer
    alignmentX, alignmentY = 0.0, 0.0
    for agent in neighbors:
        alignmentX += tempBuffer[agent, 2]
        alignmentY += tempBuffer[agent, 3]
    alignmentX, alignmentY = normalize(alignmentX, alignmentY)
    return alignmentX, alignmentY

def separation(index, neighbors):
    global tempBuffer
    separationX, separationY = 0.0, 0.0
    numSeparation = 0
    for agent in neighbors:
        distX = tempBuffer[index,0] - tempBuffer[agent,0]
        distY = tempBuffer[index,1] - tempBuffer[agent,1]
        distLength = length(distX, distY)
        sqrt = distLength ** 2
        if distLength < 40:
            if distLength > 0:
                distX /= sqrt
                distY /= sqrt
            separationX += distX
            separationY += distY
            numSeparation += 1
    return separationX, separationY

#######################################
# Utils
#######################################
def normalize(x: np.float64, y: np.float64):
    l = length(x,y)
    if l != 0.0:
        return x / l, y / l
    else:
        return x, y

def length(x,y):
    return np.sqrt(x**2 + y**2)

def distance(a,b):
    return abs(length(tempBuffer[a,0] - tempBuffer[b,0], tempBuffer[a,1] - tempBuffer[b,1]))

#######################################
# Spatial partitioning grid
#######################################
def getCell(x,y):
    return int(x//GRID_CELL_SIZE), int(y//GRID_CELL_SIZE)

def add(index, x, y):
    global cells
    cell = getCell(x,y)
    if cell in cells:
        cells[cell].append(index)
    else:
        cells[cell] = [index]

def remove(index, x, y):
    global cells
    cell = getCell(x,y)
    if cell in cells:
        if index in cells[cell]:
            cells[cell].remove(index)

def getCellNeighbors(index, x, y):
    global cells, gridW, gridH
    friends, strangers = [], []
    tileOffset = {-1, 0, 1}
    cell = getCell(x,y)
    if cell in cells:
        for i in tileOffset:
            for i in tileOffset:
                x = (cell[0] + i)%gridW
                y = (cell[1] + i)%gridH
                for agent in cells.get((x,y),[]):
                    if agent != index and distance(index, agent) < 100:
                        if specBuffer[agent] == specBuffer[index]:
                            friends.append(agent)
                        else:
                            strangers.append(agent)
    return friends, strangers

