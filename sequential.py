import numpy as np
import random as rd
import math
import datetime

def init(settings):
    #######################################
    # Constants
    #######################################
    global POPULATION, COHESION, ALIGNMENT, SEPARATION, NEIGHBOR_DIST, SEPARATION_DIST, WIDTH, HEIGHT, HALF_WIDTH, HALF_HEIGHT, SPEED, PARALLEL, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT
    POPULATION = settings.POPULATION
    SPEED = settings.SPEED
    COHESION = settings.COHESION
    ALIGNMENT = settings.ALIGNMENT
    SEPARATION = settings.SEPARATION
    NEIGHBOR_DIST = settings.NEIGHBOR_DIST
    SEPARATION_DIST = settings.SEPARATION_DIST
    WIDTH = settings.WIDTH
    HEIGHT = settings.HEIGHT
    HALF_WIDTH = WIDTH/2
    HALF_HEIGHT = HEIGHT/2
    GRID_CELL_SIZE = NEIGHBOR_DIST
    GRID_WIDTH = WIDTH//GRID_CELL_SIZE
    GRID_HEIGHT = HEIGHT//GRID_CELL_SIZE
    #######################################
    # Arrays
    #######################################
    global renderBuffer, inBuffer, outBuffer, dirBuffer, lookUpTable, tileIndexTable, boidTable, tileOffsetTable
    renderBuffer = np.zeros((POPULATION, 2), dtype=np.float32)
    inBuffer = np.zeros((POPULATION, 4), dtype=np.float32)
    outBuffer = np.zeros((POPULATION, 4), dtype=np.float32)
    dirBuffer = np.zeros((POPULATION, 27), dtype=np.float32)
    for i in range(POPULATION):
        inBuffer[i,0] = rd.uniform(0, WIDTH)
        inBuffer[i,1] = rd.uniform(0, HEIGHT)
        inBuffer[i,2] = rd.random()*2-1
        inBuffer[i,3] = rd.random()*2-1
    tileIndexTable = np.zeros(GRID_WIDTH*GRID_HEIGHT, dtype=np.int32)
    boidTable = np.zeros((POPULATION,2), dtype=np.int32)
    for index, cell in enumerate(boidTable):
        cell[0] = index
    # Look-Up table
    lookUpTable = np.zeros((GRID_WIDTH*GRID_HEIGHT,9), dtype=int)
    tileOffset = {-1, 0, 1}
    for index, cell in enumerate(lookUpTable):
        x = index % GRID_WIDTH
        y = index //GRID_WIDTH
        col = 0
        for i in tileOffset:
            for j in tileOffset:
                cell[col] = (x + i)%GRID_WIDTH + (y + j)%GRID_HEIGHT * GRID_WIDTH
                col += 1
    # Tile Offset Table (for distance check w/ edge wrapping)
    tileOffsetTable = np.empty((9,2), dtype=np.int32)
    tileValues = (-1, 0, 1)
    totIndex = 0
    for i in tileValues:
        for j in tileValues:
            tileOffsetTable[totIndex,0] = i * GRID_WIDTH
            tileOffsetTable[totIndex,1] = j * GRID_HEIGHT
            totIndex += 1

#######################################
# Update
#######################################
def update():
    start_time = datetime.datetime.now()
    #
    global inBuffer, outBuffer
    fillBoidTable()
    prepareTables()
    for index in range(POPULATION):
        getBoidDataFromTile(index)
    np.copyto(inBuffer, outBuffer)
    #
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    # print("para:",time_diff.total_seconds() * 1000)

def prepareTables():
    global tileIndexTable, boidTable
    boidTable = boidTable[boidTable[:,1].argsort()]
    tileIndexTable.fill(-1)
    currentIndex = -1
    for index, cell in enumerate(boidTable):
        if currentIndex != cell[1]:
            currentIndex = cell[1]
            tileIndexTable[currentIndex] = index

def fillBoidTable():
    global inBuffer, boidTable
    for index in range(POPULATION):
        x = int(inBuffer[index,0]//GRID_CELL_SIZE) % GRID_WIDTH
        y = int(inBuffer[index,1]//GRID_CELL_SIZE) % GRID_HEIGHT
        gridIndex = x + y * GRID_WIDTH
        boidTable[index,1] = gridIndex

def getBoidDataFromTile(index):
    global renderBuffer, inBuffer, lookUpTable, boidTable, tileIndexTable
    # Current  position
    x, y = inBuffer[index,0], inBuffer[index,1]
    # Boid direction
    dx, dy = 0.0, 0.0
    numNeighbors = 0
    alignmentX, alignmentY = 0.0, 0.0
    cohesionX, cohesionY = 0.0, 0.0
    separationX, separationY = 0.0, 0.0
    numSeparation = 0
    #Check if close to edge (for distanche check)
    if x < 100 or y < 100 or WIDTH - x < 100 or HEIGHT - y < 100:
        onEdge = True
    else: 
        onEdge = False
    # Get cell neighbors
    gridIndex = int(x//GRID_CELL_SIZE)%GRID_WIDTH + int(y//GRID_CELL_SIZE)%GRID_HEIGHT * GRID_WIDTH
    for tile in lookUpTable[gridIndex]:
        startIndex = tileIndexTable[tile]
        if startIndex != -1:
            value = tile
            while value == tile and startIndex < POPULATION - 1:
                agent = boidTable[startIndex,0]
                startIndex += 1
                value = boidTable[startIndex,1]
                ax = inBuffer[agent,0]
                ay = inBuffer[agent,1]
                adx = inBuffer[agent,2]
                ady = inBuffer[agent,3]
                if onEdge:
                    isNb = isNeighborEdge(x,y,ax,ay)
                else: 
                    isNb = (math.sqrt((x - ax)**2 + (y - ay)**2) < NEIGHBOR_DIST)
                if agent != index and isNb:
                    numNeighbors += 1
                    alignmentX += adx
                    alignmentY += ady
                    cohesionX += ax
                    cohesionY += ay
                    distX = x - ax
                    distY = y - ay
                    distLength = math.sqrt(distX**2 + distY**2)
                    sqrt = distLength ** 2
                    if distLength < SEPARATION_DIST:
                        distX /= sqrt
                        distY /= sqrt
                        separationX += distX
                        separationY += distY
                        numSeparation += 1
    if numNeighbors > 0:
        # Apply boid rules
        # Alignment
        alignmentX /= numNeighbors
        alignmentY /= numNeighbors
        # Normalize alignment vector
        l = math.sqrt(alignmentX**2 + alignmentY**2)
        alignmentX, alignmentY = alignmentX / l, alignmentY / l
        # Add alignment vector to direction
        dx += alignmentX * ALIGNMENT
        dy += alignmentY * ALIGNMENT
        # Cohesion
        cohesionX /= numNeighbors
        cohesionY /= numNeighbors
        # Get vector towards center of mass
        cohesionX -= x
        cohesionY -= y
        # Normalize cohesion vector
        l = math.sqrt(cohesionX**2 + cohesionY**2)
        cohesionX, cohesionY = cohesionX / l, cohesionY / l
        # Add cohesion vector to direction
        dx += cohesionX * COHESION
        dy += cohesionY * COHESION
        # Separation
        if numSeparation > 0:
            separationX /= numSeparation
            separationY /= numSeparation
            # Normalize separation vector
            l = math.sqrt(separationX**2 + separationY**2)
            separationX, separationY = separationX / l, separationY / l
        # Add separation vector to direction
        dx += separationX * SEPARATION
        dy += separationY * SEPARATION
        # Update directions
    dx = inBuffer[index,2] + dx * 3 # Weight w, ratio of 1:w with old/new direction
    dy = inBuffer[index,3] + dy * 3
    # Clamp speed
    l = math.sqrt(dx**2 + dy**2)
    dx, dy = dx / l, dy / l
    # Move position
    x += dx * SPEED
    y += dy * SPEED
    # Edge wrap
    if x > WIDTH:
        x = 0
    elif x < 0 :
        x = WIDTH
    if y > HEIGHT:
        y = 0
    elif y < 0 :
        y = HEIGHT
    # Render coordinates
    rx = x - HALF_WIDTH
    ry = y - HALF_HEIGHT
    rx /= HALF_WIDTH
    ry /= HALF_HEIGHT
    # WRITE TO OUTBUFFER
    renderBuffer[index,0] = rx
    renderBuffer[index,1] = ry
    outBuffer[index,0] = x
    outBuffer[index,1] = y
    outBuffer[index,2] = dx
    outBuffer[index,3] = dy

def isNeighborEdge(x,y,ax,ay):
    for offset in tileOffsetTable:
            if math.sqrt((x - (ax + offset[0]))**2 + (y - (ay + offset[1]))**2) < 100:
                return True
    return False

