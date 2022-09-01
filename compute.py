from numba import cuda
import numpy as np
import cupy as cp
import random as rd
import math
import datetime

def init(settings):
    #Validation
    if not cuda.is_available():
        print("No CUDA device found.")
        exit()
    #######################################
    # Constants
    #######################################
    global POPULATION, SPEED, WRAP_AROUND, SPOTLIGHT
    global COHESION, ALIGNMENT, SEPARATION, NEIGHBOR_DIST, NEIGHBOR_DIST_SQUARED, SEPARATION_DIST_SQUARED
    global WIDTH, HEIGHT, HALF_WIDTH, HALF_HEIGHT, SPEED, PARALLEL, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, GRID_SIZE
    POPULATION = settings.POPULATION
    SPEED = settings.SPEED
    COHESION = settings.COHESION
    ALIGNMENT = settings.ALIGNMENT
    SEPARATION = settings.SEPARATION
    NEIGHBOR_DIST = settings.NEIGHBOR_DIST
    NEIGHBOR_DIST_SQUARED = NEIGHBOR_DIST ** 2 # Useful for evaluating distances
    SEPARATION_DIST_SQUARED = (settings.NEIGHBOR_DIST * settings.SEPARATION_DIST) ** 2
    WIDTH = settings.WIDTH
    HEIGHT = settings.HEIGHT
    HALF_WIDTH = WIDTH/2
    HALF_HEIGHT = HEIGHT/2
    GRID_CELL_SIZE = settings.NEIGHBOR_DIST
    GRID_WIDTH = WIDTH//GRID_CELL_SIZE
    GRID_HEIGHT = HEIGHT//GRID_CELL_SIZE
    GRID_SIZE = GRID_WIDTH*GRID_HEIGHT
    WRAP_AROUND = settings.WRAP_AROUND
    SPOTLIGHT = settings.SPOTLIGHT
    #######################################
    # Arrays
    #######################################
    global renderData, boidData, lookUpTable, cellIndexTable, boidPositionTable
    renderData = np.zeros((POPULATION, 3), dtype=np.float32)
    tempBoidData = np.zeros((POPULATION, 32), dtype=np.float32)
    for i in range(POPULATION):
        tempBoidData[i,0] = 0.0
        tempBoidData[i,1] = rd.uniform(0, WIDTH)
        tempBoidData[i,2] = rd.uniform(0, HEIGHT)
        tempBoidData[i,3] = rd.random()*2-1
        tempBoidData[i,4] = rd.random()*2-1
    boidData = cuda.to_device(tempBoidData)
    # Tables
    # Boid Table - List of every boid and their current cell, sorted by cell
    tempBoidTable = np.zeros((POPULATION,2), dtype=np.int32)
    for index, cell in enumerate(tempBoidTable):
        cell[0] = index
    # Sends array to GPU with CuPy instead of numba, to use CuPy methods
    boidPositionTable = cp.asarray(tempBoidTable)
    # Tables
    # Cell Index Table (gives start index of cell on sorted Boid Table)
    cellIndexTable = np.zeros(GRID_SIZE, dtype=np.int32)
    # Tables
    # Look-Up table (for neighbor cells - read only)
    tempLookUpTable = np.zeros((GRID_SIZE,9), dtype=np.int32)
    cellOffset = {-1, 0, 1}
    for index, cell in enumerate(tempLookUpTable):
        x = index % GRID_WIDTH
        y = index //GRID_WIDTH
        col = 0
        for i in cellOffset:
            for j in cellOffset:
                cell[col] = (x + i)%GRID_WIDTH + (y + j)%GRID_HEIGHT * GRID_WIDTH
                col += 1
    lookUpTable = cuda.to_device(tempLookUpTable)    

#######################################
# Update
#######################################

def update(params):
    updateParams(params)
    global renderData, boidData, lookUpTable, boidPositionTable, cellIndexTable
    # Kernels are set to default stream and are executed sequentially
    # 512 threads per block, as many blocks as we need 
    nthreads = 512
    nblocks = (POPULATION + (nthreads - 1)) // nthreads
    fillBoidPositionTable[nblocks, nthreads](boidData, boidPositionTable, renderData)
    # CuPy sort, a bit faster than on CPU
    boidPositionTable = boidPositionTable[cp.argsort(boidPositionTable[:,1])]
    # New index table according to data from sorted boid table
    cellIndexTable.fill(-1)
    fillCellIndexTable[nblocks, nthreads](boidPositionTable, cellIndexTable)
    # 2D kernel. X axis is boids, Y axis is each of their 9 respective neighbor cells
    neighborSearch[(nblocks,9), (nthreads,1)](boidData, lookUpTable, cellIndexTable, boidPositionTable, renderData, SPOTLIGHT, WRAP_AROUND, COHESION, ALIGNMENT, SEPARATION, SEPARATION_DIST_SQUARED)
    # Update positions and write to render buffer
    writeData[nblocks, nthreads](boidData, renderData, WRAP_AROUND, SPEED)

# Update parameters with GUI input
def updateParams(params):
    global POPULATION, SPEED, WRAP_AROUND, SPOTLIGHT
    global COHESION, ALIGNMENT, SEPARATION, NEIGHBOR_DIST, NEIGHBOR_DIST_SQUARED, SEPARATION_DIST_SQUARED
    POPULATION = params.POPULATION
    SPEED = params.SPEED
    COHESION = params.COHESION
    ALIGNMENT = params.ALIGNMENT
    SEPARATION = params.SEPARATION
    SEPARATION_DIST_SQUARED = (params.NEIGHBOR_DIST * params.SEPARATION_DIST) ** 2
    WRAP_AROUND = params.WRAP_AROUND
    SPOTLIGHT = params.SPOTLIGHT


#######################################
# Kernels
#######################################

# Update Boid Table with new cell for every agent
@cuda.jit
def fillBoidPositionTable(boidData, boidPositionTable, renderData):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    x = int(boidData[index,1]//GRID_CELL_SIZE) % GRID_WIDTH
    y = int(boidData[index,2]//GRID_CELL_SIZE) % GRID_HEIGHT
    gridIndex = x + y * GRID_WIDTH
    boidPositionTable[index,0] = index
    boidPositionTable[index,1] = gridIndex
    renderData[index, 2] = 0.0


# Update Index Table to give start index for every cell on boidPositionTable
@cuda.jit
def fillCellIndexTable(boidPositionTable, cellIndexTable):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    gridIndex = boidPositionTable[index,1]
    if index > 0:
        previousGridIndex = boidPositionTable[index-1,1]
        if previousGridIndex == gridIndex:
            return
    cellIndexTable[gridIndex] = index

# For a certain agent and a certain cell in the agent's neighboring cells
# Get neighbor data and compute new direction vector with weight
# boidData -> boidData
@cuda.jit
def neighborSearch(boidData, lookUpTable, cellIndexTable, boidPositionTable, renderData, SPOTLIGHT, WRAP_AROUND, COHESION, ALIGNMENT, SEPARATION, SEPARATION_DIST_SQUARED):
    lookUpIndex = cuda.blockIdx.y
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if index >= POPULATION:
        return
    # Current  position
    x, y = boidData[index,1], boidData[index,2]
    # Boid direction
    dx, dy = 0.0, 0.0
    numNeighbors = 0
    alignmentX, alignmentY = 0.0, 0.0
    cohesionX, cohesionY = 0.0, 0.0
    separationX, separationY = 0.0, 0.0
    numSeparation = 0
    #Check if close to edge (for distanche check)
    if WRAP_AROUND and (x < NEIGHBOR_DIST or y < NEIGHBOR_DIST or WIDTH - x < NEIGHBOR_DIST or HEIGHT - y < NEIGHBOR_DIST):
        onEdge = True
    else: 
        onEdge = False
    # Get cell neighbors
    gridIndex = int(x//GRID_CELL_SIZE)%GRID_WIDTH + int(y//GRID_CELL_SIZE)%GRID_HEIGHT * GRID_WIDTH
    cell = lookUpTable[gridIndex, lookUpIndex]
    startIndex = cellIndexTable[cell]
    if startIndex != -1: # True if there's at least one agent on the current cell
        # Condition to check if there are more agents to observe on the cell
        cond = cell
        while cond == cell and startIndex < POPULATION:# and numNeighbors < 20:
            agent = boidPositionTable[startIndex,0]
            startIndex += 1
            # Check if at the end of table
            if startIndex >= POPULATION:
                cond = -1
            # Or set condition to check if all agents on the cell have been observed
            else:
                cond = boidPositionTable[startIndex,1]
            # Get agent data and check if it's a neighbor
            ax = boidData[agent,1]
            ay = boidData[agent,2]
            adx = boidData[agent,3]
            ady = boidData[agent,4]
            if onEdge:
                (ax, ay) = minimalToroidalDistance(x,y,ax,ay)
            distX = x - ax
            distY = y - ay
            distLengthSquared = distX**2 + distY**2
            isNeighbor = (distLengthSquared < NEIGHBOR_DIST_SQUARED)
            isTooClose = (distLengthSquared < SEPARATION_DIST_SQUARED)
            # SPOTLIGHT
            if SPOTLIGHT and index == 0:
                renderData[index, 2] = 1.0
                if isTooClose:
                    renderData[agent, 2] = 2.0
                elif isNeighbor:
                    renderData[agent, 2] = 3.0
                else:
                    renderData[agent, 2] = 4.0
            # Collect agent data
            if agent != index and isNeighbor:
                numNeighbors += 1
                alignmentX += adx
                alignmentY += ady
                cohesionX += ax
                cohesionY += ay
                if isTooClose:
                    # distX /= distLengthSquared
                    # distY /= distLengthSquared
                    separationX += distX
                    separationY += distY
                    numSeparation += 1
    if numNeighbors > 0:
        # Apply boid rules
        # Alignment
        # Normalize alignment vector
        l = math.sqrt(alignmentX**2 + alignmentY**2)
        alignmentX, alignmentY = alignmentX / l, alignmentY / l
        # Add alignment vector to direction
        dx += alignmentX * ALIGNMENT
        dy += alignmentY * ALIGNMENT
        # Cohesion
        # Get vector towards center of mass
        cohesionX /= numNeighbors
        cohesionY /= numNeighbors
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
            # Normalize separation vector
            l = math.sqrt(separationX**2 + separationY**2)
            separationX, separationY = separationX / l, separationY / l
            # Add separation vector to direction
            dx += separationX * SEPARATION
            dy += separationY * SEPARATION
    # Write to direction buffer at respective boid and cell index
    boidData[index, lookUpIndex*3+5] = dx
    boidData[index, lookUpIndex*3+6] = dy
    boidData[index, lookUpIndex*3+7] = numNeighbors

# Reads direction vectors and weights for all neighbor cell calculations
# Gets final direction and position and writes to memory
# boidData -> renderData
@cuda.jit
def writeData(boidData, renderData, WRAP_AROUND, SPEED):
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    x, y = boidData[index,1], boidData[index,2]
    # Previous direction
    pdx, pdy = boidData[index,3], boidData[index,4]
    dx = 0.0
    dy = 0.0
    total = 0
    for i in range(9):
        weight = boidData[index, i*3+7]
        total += weight
        dx += boidData[index, i*3+5] * weight
        dy += boidData[index, i*3+6] * weight
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
        dx += wallX / 10
        dy += wallY / 10
    # Update directions
    dx = pdx + dx * 3 # Weight w, ratio of 1:w with old/new direction
    dy = pdy + dy * 3
    # Clamp speed
    l = math.sqrt(dx**2 + dy**2)
    if l > 1.0:
        dx, dy = dx / l, dy / l
    # Move position
    x += dx * SPEED
    y += dy * SPEED
    # Edge wrap
    if WRAP_AROUND:
        if x >= WIDTH:
            x = 0
        elif x < 0 :
            x = WIDTH
        if y >= HEIGHT:
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
    # renderData[index,2] = ry
    # Store data for next calculation
    boidData[index,1] = x
    boidData[index,2] = y
    boidData[index,3] = dx
    boidData[index,4] = dy

# If agent is near an edge and WrapAround is set to True
# Checks if other agent is a neighbor for all neighboring cell configurations (9)
# Looks for minimal toroidal distance and returns perceived position
# Given what is provided by Numba we have to get the minimum manually
@cuda.jit(device=True)
def minimalToroidalDistance(x,y,ax,ay):
    # X axis
    ax1 = ax - WIDTH
    ax2 = ax + WIDTH
    mix = abs(x - ax)
    mix1 = abs(x - ax1)
    mix2 = abs(x - ax2)
    if mix1 < mix:
        mix = mix1
        ax = ax1
    if mix2 < mix:
        mix = mix2
        ax = ax2
    # Y axis
    ay1 = ay - HEIGHT
    ay2 = ay + HEIGHT
    miy = abs(y - ay)
    miy1 = abs(y - ay1)
    miy2 = abs(y - ay2)
    if miy1 < miy:
        miy = miy1
        ay = ay1
    if miy2 < miy:
        miy = miy2
        ay = ay2
    return (ax,ay)

