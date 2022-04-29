from numba import cuda
import numpy as np
import cupy as cp
import random as rd
import math

def init(settings):
    #######################################
    # Constants
    #######################################
    global POPULATION, COHESION, ALIGNMENT, SEPARATION, NEIGHBOR_DIST_SQUARED, SEPARATION_DIST_SQUARED, WRAP_AROUND
    global WIDTH, HEIGHT, HALF_WIDTH, HALF_HEIGHT, SPEED, PARALLEL, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT
    POPULATION = settings.POPULATION
    SPEED = settings.SPEED
    COHESION = settings.COHESION
    ALIGNMENT = settings.ALIGNMENT
    SEPARATION = settings.SEPARATION
    NEIGHBOR_DIST_SQUARED = settings.NEIGHBOR_DIST ** 2 # Useful for evaluating distances
    SEPARATION_DIST_SQUARED = settings.SEPARATION_DIST ** 2
    WIDTH = settings.WIDTH
    HEIGHT = settings.HEIGHT
    HALF_WIDTH = WIDTH/2
    HALF_HEIGHT = HEIGHT/2
    GRID_CELL_SIZE = settings.NEIGHBOR_DIST
    GRID_WIDTH = WIDTH//GRID_CELL_SIZE
    GRID_HEIGHT = HEIGHT//GRID_CELL_SIZE
    WRAP_AROUND = settings.WRAP_AROUND
    #######################################
    # Arrays
    #######################################
    global renderData, boidData, neighborTileData, lookUpTable, tileIndexTable, boidPositionTable
    renderData = np.zeros((POPULATION, 2), dtype=np.float32)
    neighborTileData = cuda.to_device(np.zeros((POPULATION, 27), dtype=np.float32))
    tempBoidData = np.zeros((POPULATION, 4), dtype=np.float32)
    for i in range(POPULATION):
        tempBoidData[i,0] = rd.uniform(0, WIDTH)
        tempBoidData[i,1] = rd.uniform(0, HEIGHT)
        tempBoidData[i,2] = rd.random()*2-1
        tempBoidData[i,3] = rd.random()*2-1
    boidData = cuda.to_device(tempBoidData)
    # Tables
    # Boid Table - List of every boid and their current tile, sorted by tile
    tempBoidTable = np.zeros((POPULATION,2), dtype=np.int32)
    for index, cell in enumerate(tempBoidTable):
        cell[0] = index
    # Sends array to GPU with CuPy instead of numba, to use CuPy methods
    boidPositionTable = cp.asarray(tempBoidTable)
    # Tables
    # Tile Index Table (gives start index of tile on sorted Boid Table)
    tileIndexTable = np.zeros(GRID_WIDTH*GRID_HEIGHT, dtype=np.int32)
    # Tables
    # Look-Up table (for neighbor tiles - read only)
    tempLookUpTable = np.zeros((GRID_WIDTH*GRID_HEIGHT,9), dtype=np.int32)
    tileOffset = {-1, 0, 1}
    for index, cell in enumerate(tempLookUpTable):
        x = index % GRID_WIDTH
        y = index //GRID_WIDTH
        col = 0
        for i in tileOffset:
            for j in tileOffset:
                cell[col] = (x + i)%GRID_WIDTH + (y + j)%GRID_HEIGHT * GRID_WIDTH
                col += 1
    lookUpTable = cuda.to_device(tempLookUpTable)    

#######################################
# Update
#######################################
def update():
    global renderData, neighborTileData, boidData, lookUpTable, boidPositionTable, tileIndexTable
    # Kernels are set to default stream and are executed sequentially
    # 512 threads per block, as many blocks as we need 
    nthreads = 512
    nblocks = (POPULATION + (nthreads - 1)) // nthreads
    fillBoidPositionTable[nblocks, nthreads](boidData, boidPositionTable)
    # CuPy sort, a bit faster than on CPU
    boidPositionTable = boidPositionTable[cp.argsort(boidPositionTable[:,1])]
    # New index table according to data from sorted boid table
    tileIndexTable.fill(-1)
    fillTileIndexTable[nblocks, nthreads](boidPositionTable, tileIndexTable)
    # 2D kernel. X axis is boids, Y axis is each of their 9 respective neighbor tiles
    getBoidDataFromTile[(nblocks,9), (nthreads,1)](boidData, neighborTileData, lookUpTable, tileIndexTable, boidPositionTable)
    # Once previous kernel is finished, gets data from dirBuffer and writes to renderBuffer and boidBuffer
    writeData[nblocks, nthreads](boidData, renderData, neighborTileData)


#######################################
# Parallel
#######################################

# Update Boid Table with new tile for every agent
@cuda.jit
def fillBoidPositionTable(boidData, boidPositionTable):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    x = int(boidData[index,0]//GRID_CELL_SIZE) % GRID_WIDTH
    y = int(boidData[index,1]//GRID_CELL_SIZE) % GRID_HEIGHT
    gridIndex = x + y * GRID_WIDTH
    boidPositionTable[index,1] = gridIndex

# Update Index Table to give start index for every tile on tileIndexTable
@cuda.jit
def fillTileIndexTable(boidPositionTable, tileIndexTable):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    gridIndex = boidPositionTable[index,1]
    if index > 0:
        previousGridIndex = boidPositionTable[index-1,1]
        if previousGridIndex == gridIndex:
            return
    tileIndexTable[gridIndex] = index

# For a certain agent and a certain tile in the agent's neighboring tiles
# Get neighbor data and compute new direction vector with weight
# boidData -> neighborTileData
@cuda.jit
def getBoidDataFromTile(boidData, neighborTileData, lookUpTable, tileIndexTable, boidPositionTable):
    lookUpIndex = cuda.blockIdx.y
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if index >= POPULATION:
        return
    # Current  position
    x, y = boidData[index,0], boidData[index,1]
    # Boid direction
    dx, dy = 0.0, 0.0
    numNeighbors = 0
    alignmentX, alignmentY = 0.0, 0.0
    cohesionX, cohesionY = 0.0, 0.0
    separationX, separationY = 0.0, 0.0
    numSeparation = 0
    #Check if close to edge (for distanche check)
    if WRAP_AROUND and (x < 100 or y < 100 or WIDTH - x < 100 or HEIGHT - y < 100):
        onEdge = True
    else: 
        onEdge = False
    # Get cell neighbors
    gridIndex = int(x//GRID_CELL_SIZE)%GRID_WIDTH + int(y//GRID_CELL_SIZE)%GRID_HEIGHT * GRID_WIDTH
    tile = lookUpTable[gridIndex, lookUpIndex]
    startIndex = tileIndexTable[tile]
    if startIndex != -1:
        value = tile
        while value == tile and startIndex < POPULATION:
            agent = boidPositionTable[startIndex,0]
            startIndex += 1
            if startIndex == POPULATION:
                value = -1
            else:
                value = boidPositionTable[startIndex,1]
            ax = boidData[agent,0]
            ay = boidData[agent,1]
            adx = boidData[agent,2]
            ady = boidData[agent,3]
            if onEdge:
                isNeighbor = isNeighborEdge(x,y,ax,ay)
            else: 
                isNeighbor = (((x - ax)**2 + (y - ay)**2) < NEIGHBOR_DIST_SQUARED)
            if agent != index and isNeighbor:
                numNeighbors += 1
                alignmentX += adx
                alignmentY += ady
                cohesionX += ax
                cohesionY += ay
                distX = x - ax
                distY = y - ay
                distLength = distX**2 + distY**2
                if distLength < SEPARATION_DIST_SQUARED:
                    distX /= distLength
                    distY /= distLength
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
    # Write to direction buffer at respective boid and tile index
    neighborTileData[index, lookUpIndex*3] = dx
    neighborTileData[index, lookUpIndex*3+1] = dy
    neighborTileData[index, lookUpIndex*3+2] = numNeighbors

# Reads direction vectors and weights for all neighbor tile calculations
# Gets final direction and position and writes to memory
# neighborTileData -> renderData, boidData
@cuda.jit
def writeData(boidData, renderData, neighborTileData):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    x, y = boidData[index,0], boidData[index,1]
    # Previous direction
    pdx, pdy = boidData[index,2], boidData[index,3]
    dx = 0.0
    dy = 0.0
    total = 0
    for i in range(9):
        weight = neighborTileData[index, i*3+2]
        total += weight
        dx += neighborTileData[index, i*3] * weight
        dy += neighborTileData[index, i*3+1] * weight
    if total > 0:
        dx /= total
        dy /= total
    # If Edge Wrapping is off, avoid walls
    MARGIN = 200
    if not WRAP_AROUND:
        wallX, wallY = 0.0, 0.0
        x2 = WIDTH - x 
        y2 = HEIGHT - y
        if x < MARGIN:
            wallX = MARGIN - x
            wallX /= MARGIN
            wallX = wallX ** 2
        elif x2 < MARGIN:
            wallX = MARGIN - x2
            wallX /= MARGIN
            wallX = wallX ** 2
            wallX *= -1
        if y < MARGIN:
            wallY = MARGIN - y
            wallY /= MARGIN
            wallY = wallY ** 2
        elif y2 < MARGIN:
            wallY = MARGIN - y2
            wallY /= MARGIN
            wallY = wallY ** 2
            wallY *= -1
        dx += wallX /10
        dy += wallY /10
    # Update directions
    dx = pdx + dx * 3 # Weight w, ratio of 1:w with old/new direction
    dy = pdy + dy * 3
    # Clamp speed
    l = math.sqrt(dx**2 + dy**2)
    dx, dy = dx / l, dy / l
    # Move position
    x += dx * SPEED
    y += dy * SPEED
    # Edge wrap
    if WRAP_AROUND:
        if x > WIDTH:
            x = 0
        elif x < 0 :
            x = WIDTH
        if y > HEIGHT:
            y = 0
        elif y < 0 :
            y = HEIGHT
    # Render coordinates
    rx = (x - HALF_WIDTH) / HALF_WIDTH
    ry = (y - HALF_HEIGHT) / HALF_HEIGHT
    # WRITE TO MEMORY
    # Write screen position in [-1,1]
    renderData[index,0] = rx
    renderData[index,1] = ry
    # Store data for next calculation
    boidData[index,0] = x
    boidData[index,1] = y
    boidData[index,2] = dx
    boidData[index,3] = dy

# If agent is near an edge
# Checks if other agent is a neighbor for all neighboring tile configurations (9)
# Looks for minimal toroidal distance
@cuda.jit(device=True)
def isNeighborEdge(x,y,ax,ay):
    mix = min(abs(x - ax - WIDTH), abs(x - ax), abs(x - ax + WIDTH))
    miy = min(abs(y - ay - HEIGHT), abs(y - ay), abs(y - ay + HEIGHT))
    if ((mix)**2 + (miy)**2) < NEIGHBOR_DIST_SQUARED:
        return True
    return False

