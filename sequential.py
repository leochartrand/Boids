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
    global renderData, inBoidData, outBoidData, neighborTileData, lookUpTable, tileIndexTable, boidPositionTable
    renderData = np.zeros((POPULATION, 2), dtype=np.float32)
    inBoidData = np.zeros((POPULATION, 4), dtype=np.float32)
    outBoidData = np.zeros((POPULATION, 4), dtype=np.float32)
    neighborTileData = np.zeros((POPULATION, 27), dtype=np.float32)
    for i in range(POPULATION):
        inBoidData[i,0] = rd.uniform(0, WIDTH)
        inBoidData[i,1] = rd.uniform(0, HEIGHT)
        inBoidData[i,2] = rd.random()*2-1
        inBoidData[i,3] = rd.random()*2-1
    tileIndexTable = np.zeros(GRID_WIDTH*GRID_HEIGHT, dtype=np.int32)
    boidPositionTable = np.zeros((POPULATION,2), dtype=np.int32)
    for index, cell in enumerate(boidPositionTable):
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

#######################################
# Update
#######################################
def update():
    start_time = datetime.datetime.now()
    #
    global inBoidData, outBoidData
    fillBoidTable()
    prepareTables()
    for index in range(POPULATION):
        getBoidDataFromTile(index)
    np.copyto(inBoidData, outBoidData)
    #
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    # print("para:",time_diff.total_seconds() * 1000)

def prepareTables():
    global tileIndexTable, boidPositionTable
    boidPositionTable = boidPositionTable[boidPositionTable[:,1].argsort()]
    tileIndexTable.fill(-1)
    currentIndex = -1
    for index, cell in enumerate(boidPositionTable):
        if currentIndex != cell[1]:
            currentIndex = cell[1]
            tileIndexTable[currentIndex] = index

def fillBoidTable():
    global inBoidData, boidPositionTable
    for index in range(POPULATION):
        x = int(inBoidData[index,0]//GRID_CELL_SIZE) % GRID_WIDTH
        y = int(inBoidData[index,1]//GRID_CELL_SIZE) % GRID_HEIGHT
        gridIndex = x + y * GRID_WIDTH
        boidPositionTable[index,1] = gridIndex

def getBoidDataFromTile(index):
    global renderData, inBoidData, lookUpTable, boidPositionTable, tileIndexTable
    # Current  position
    x, y = inBoidData[index,0], inBoidData[index,1]
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
                agent = boidPositionTable[startIndex,0]
                startIndex += 1
                value = boidPositionTable[startIndex,1]
                ax = inBoidData[agent,0]
                ay = inBoidData[agent,1]
                adx = inBoidData[agent,2]
                ady = inBoidData[agent,3]
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
    dx = inBoidData[index,2] + dx * 3 # Weight w, ratio of 1:w with old/new direction
    dy = inBoidData[index,3] + dy * 3
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
    renderData[index,0] = rx
    renderData[index,1] = ry
    outBoidData[index,0] = x
    outBoidData[index,1] = y
    outBoidData[index,2] = dx
    outBoidData[index,3] = dy

def isNeighborEdge(x,y,ax,ay):
    mix = min(abs(x - ax - WIDTH), abs(x - ax), abs(x - ax + WIDTH))
    miy = min(abs(y - ay - HEIGHT), abs(y - ay), abs(y - ay + HEIGHT))
    if ((mix)**2 + (miy)**2) < NEIGHBOR_DIST**2:
        return True
    return False

