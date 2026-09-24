from __future__ import annotations

import math
import random as rd
from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np
from numba import cuda as numbaCuda
from numba import float32, int32
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba.cuda.dispatcher import CUDADispatcher

if TYPE_CHECKING:
    from util import Parameters

# Numba's CUDA API is typed for its compiler, not for type checkers (kernel launches, cuda.grid,
# shared arrays... show as errors), so type checkers treat it as untyped
cuda: Any = numbaCuda

# Threads per block of the neighbor search, and candidates loaded in shared memory at a time
TILE_SIZE = 128
# Blocks per cell: each takes every CELL_BLOCKS-th group of TILE_SIZE agents of the cell.
# High, so that dense flocks (tens of thousands of agents in a cell) still spread over the whole GPU:
# blocks of cells with fewer agents have nothing to do and return right away
CELL_BLOCKS = 128
# Movement (SPEED, turning) is set per frame at this frame rate, and scaled for longer or shorter frames
REFERENCE_FPS = 60
# Longest time step in seconds, so a pause (e.g. dragging the window) doesn't make agents jump
MAX_TIME_STEP = 0.1
# Wander (Reynolds): radius of the circle the wander target moves on, ahead of the agent at a distance of 1.
# The target is then at most 30 degrees off the agent's direction
WANDER_RADIUS = np.float32(0.5)

def init(settings: Parameters) -> None:
    #Validation
    if not cuda.is_available():
        print("No CUDA device found.")
        exit()
    #######################################
    # Constants
    #######################################
    # (the settings that can change while running are read by updateParams)
    global POPULATION, NEIGHBOR_DIST, NEIGHBOR_DIST_SQUARED, CANDIDATE_BUDGET
    global WIDTH, HEIGHT, HALF_WIDTH, HALF_HEIGHT, GRID_CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, GRID_SIZE
    POPULATION = settings.POPULATION
    # Max agents tested per neighbor search (a random sample above it)
    CANDIDATE_BUDGET = max(1, settings.CANDIDATE_BUDGET)
    # Constants used in kernel float math are float32, otherwise Numba promotes that math to float64
    NEIGHBOR_DIST = np.float32(settings.NEIGHBOR_DIST)
    NEIGHBOR_DIST_SQUARED = np.float32(settings.NEIGHBOR_DIST ** 2) # Useful for evaluating distances
    WIDTH = np.float32(settings.WIDTH)
    HEIGHT = np.float32(settings.HEIGHT)
    HALF_WIDTH = np.float32(settings.WIDTH/2)
    HALF_HEIGHT = np.float32(settings.HEIGHT/2)
    GRID_CELL_SIZE = np.float32(settings.NEIGHBOR_DIST)
    # Grid dimensions stay ints (array sizes, modulo, indexing)
    GRID_WIDTH = settings.WIDTH//settings.NEIGHBOR_DIST
    GRID_HEIGHT = settings.HEIGHT//settings.NEIGHBOR_DIST
    GRID_SIZE = GRID_WIDTH*GRID_HEIGHT
    #######################################
    # Arrays
    #######################################
    global boids, sortedBoids, steering, frameCounter, timeStep, cursor, wanderAngles, lookUpTable, cellIndexTable, cellEndTable, boidPositionTable, unsortedBoidPositionTable
    # Boids - (x, y, dx, dy) for every agent, in agent order
    tempBoids = np.zeros((POPULATION, 4), dtype=np.float32)
    for i in range(POPULATION):
        tempBoids[i,0] = rd.uniform(0, settings.WIDTH)
        tempBoids[i,1] = rd.uniform(0, settings.HEIGHT)
        tempBoids[i,2] = rd.random()*2-1
        tempBoids[i,3] = rd.random()*2-1
    boids = cp.asarray(tempBoids)
    # Sorted Boids - copy of Boids in cell order, rebuilt every frame so that
    # agents of the same cell are contiguous in memory for the neighbor search
    sortedBoids = cp.empty_like(boids)
    # Steering - (x, y) direction given by the three rules for every agent, indexed by sorted slot
    steering = cp.zeros((POPULATION, 2), dtype=cp.float32)
    # Frame counter, seeds the random sampling of candidates (kernel arguments are fixed in the graph)
    frameCounter = cp.zeros(1, dtype=cp.int32)
    # Time step, in frames at REFERENCE_FPS (in memory too, as kernel arguments are fixed in the graph)
    timeStep = cp.ones(1, dtype=cp.float32)
    # Cursor - (x, y, mode) in the simulated space, mode 1 attracts, -1 repels, 0 off (in memory too)
    cursor = cp.zeros(3, dtype=cp.float32)
    # Wander Angles - where each agent's wander target is on its circle, in agent order
    wanderAngles = cp.zeros(POPULATION, dtype=cp.float32)
    # Tables
    # Boid Table - List of every boid and their current cell, sorted by cell
    # (filled unsorted, then sorted into a second table so both keep their memory)
    unsortedBoidPositionTable = cp.zeros((POPULATION,2), dtype=cp.int32)
    boidPositionTable = cp.zeros((POPULATION,2), dtype=cp.int32)
    # Tables
    # Cell Index Tables (give start and end index of cell on sorted Boid Table)
    cellIndexTable = cp.zeros(GRID_SIZE, dtype=cp.int32)
    cellEndTable = cp.zeros(GRID_SIZE, dtype=cp.int32)
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
    # Numba views of the CuPy arrays, made once: converting CuPy arrays on every launch is slow
    global nbBoids, nbSortedBoids, nbSteering, nbFrameCounter, nbTimeStep, nbCursor, nbWanderAngles, nbCellIndexTable, nbCellEndTable, nbBoidPositionTable, nbUnsortedBoidPositionTable
    nbBoids = cuda.as_cuda_array(boids)
    nbSortedBoids = cuda.as_cuda_array(sortedBoids)
    nbSteering = cuda.as_cuda_array(steering)
    nbFrameCounter = cuda.as_cuda_array(frameCounter)
    nbTimeStep = cuda.as_cuda_array(timeStep)
    nbCursor = cuda.as_cuda_array(cursor)
    nbWanderAngles = cuda.as_cuda_array(wanderAngles)
    nbCellIndexTable = cuda.as_cuda_array(cellIndexTable)
    nbCellEndTable = cuda.as_cuda_array(cellEndTable)
    nbBoidPositionTable = cuda.as_cuda_array(boidPositionTable)
    nbUnsortedBoidPositionTable = cuda.as_cuda_array(unsortedBoidPositionTable)
    #######################################
    # Kernels
    #######################################
    # Compiled here, as they read the constants above. Fixed signatures: other scalar types are
    # converted to these, arrays of other types raise, and no slower (e.g. float64) version is compiled
    compileKernel(fillBoidPositionTable, "void(float32[:,::1], int32[:,::1], float32[:,::1])")
    compileKernel(fillCellIndexTable, "void(int32[:,::1], int32[::1], int32[::1])")
    compileKernel(neighborSearch, "void(float32[:,::1], float32[:,::1], int32[:,::1], int32[::1], int32[::1], int32[:,::1], float32[:,::1], int32[::1], "
        + "boolean, boolean, float32, float32, float32, float32, float32)")
    compileKernel(writeData, "void(float32[:,::1], float32[:,::1], float32[:,::1], int32[:,::1], float32[:,::1], float32[::1], int32[::1], float32[::1], "
        + "float32[::1], boolean, float32, float32, float32, float32, float32, float32)")
    #######################################
    # Stream and CUDA graph
    #######################################
    # All simulation work runs on one non-blocking stream, so a frame can be recorded
    # as a CUDA graph (the default stream can't be recorded)
    global stream, nbStream, graph, graphKey, graphPool
    stream = cp.cuda.Stream(non_blocking=True)
    nbStream = cuda.external_stream(stream.ptr)
    graph, graphKey = None, None
    # Temporary arrays of a recorded frame (the sort) come from this pool, which nothing
    # else uses, so their memory is never handed to other arrays while the graph uses it
    graphPool = cp.cuda.MemoryPool()
    # The arrays above were filled on the default stream
    cp.cuda.Device().synchronize()

def compileKernel(kernel: CUDADispatcher, signature: str) -> None:
    _ = kernel.compile(signature)
    kernel.disable_compile()

#######################################
# Update
#######################################

# renderData - (x, y, color) for every agent, the mapped OpenGL vertex buffer
# dt - time since the last update, in seconds
# cursorInput - (x, y, mode) of the cursor in the simulated space, mode 1 attracts, -1 repels, 0 off (default)
def update(params: Parameters, renderData: cp.ndarray, dt: float, cursorInput: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    global graph, graphKey
    updateParams(params)
    # Kernel arguments and array addresses are recorded in the graph,
    # so it is recorded again when a parameter or the buffer address changes
    key = (speed, cohesion, alignment, separation, separationDistSquared, separationFalloff, wrapAround, spotlight, jitter, wander, wanderRate, cursorAttention, cursorRadius, renderData.data.ptr)
    if key != graphKey:
        nbRenderData = cuda.as_cuda_array(renderData, sync=False)
        with stream, cp.cuda.using_allocator(graphPool.malloc):
            stream.begin_capture()
            try:
                recordFrame(nbRenderData)
            finally:
                graph = stream.end_capture()
        graphKey = key
    with stream:
        timeStep.fill(min(dt, MAX_TIME_STEP) * REFERENCE_FPS)
        cursor.set(np.array(cursorInput, dtype=np.float32), stream=stream)
    assert graph is not None # recorded above on the first update
    graph.launch(stream)

# Launches one simulation step on the stream (recorded into the graph, not executed)
def recordFrame(renderData: DeviceNDArray) -> None:
    # 512 threads per block, as many blocks as we need
    nthreads = 512
    nblocks = (POPULATION + (nthreads - 1)) // nthreads
    cp.add(frameCounter, 1, out=frameCounter)
    fillBoidPositionTable[nblocks, nthreads, nbStream](nbBoids, nbUnsortedBoidPositionTable, renderData)
    # CuPy sort, a bit faster than on CPU
    order = cp.argsort(unsortedBoidPositionTable[:,1])
    cp.take(unsortedBoidPositionTable, order, axis=0, out=boidPositionTable)
    # Reorder agent data to match the sorted table
    cp.take(boids, order, axis=0, out=sortedBoids)
    # New index tables according to data from sorted boid table (empty cells stay at start = end = 0)
    cellIndexTable.fill(0)
    cellEndTable.fill(0)
    fillCellIndexTable[nblocks, nthreads, nbStream](nbBoidPositionTable, nbCellIndexTable, nbCellEndTable)
    # CELL_BLOCKS blocks per cell
    neighborSearch[(GRID_SIZE, CELL_BLOCKS), TILE_SIZE, nbStream](nbSortedBoids, nbSteering, lookUpTable, nbCellIndexTable, nbCellEndTable, nbBoidPositionTable, renderData,
        nbFrameCounter, spotlight, wrapAround, cohesion, alignment, separation, separationDistSquared, separationFalloff)
    # Update positions and write to render buffer
    writeData[nblocks, nthreads, nbStream](nbBoids, nbSortedBoids, nbSteering, nbBoidPositionTable, renderData, nbTimeStep, nbFrameCounter,
        nbWanderAngles, nbCursor, wrapAround, speed, jitter, wander, wanderRate, cursorAttention, cursorRadius)

# Update parameters with GUI input (settings that can change while running)
def updateParams(params: Parameters) -> None:
    global speed, cohesion, alignment, separation, separationDistSquared, separationFalloff, wrapAround, spotlight
    global jitter, wander, wanderRate, cursorAttention, cursorRadius
    # Kernel arguments are float32, Python floats would arrive as float64
    speed = np.float32(params.SPEED)
    cohesion = np.float32(params.COHESION)
    alignment = np.float32(params.ALIGNMENT)
    separation = np.float32(params.SEPARATION)
    separationDistSquared = np.float32((params.NEIGHBOR_DIST * params.SEPARATION_DIST) ** 2)
    separationFalloff = np.float32(params.SEPARATION_FALLOFF)
    wrapAround = params.WRAP_AROUND
    spotlight = params.SPOTLIGHT
    jitter = np.float32(params.JITTER)
    wander = np.float32(params.WANDER)
    wanderRate = np.float32(math.radians(params.WANDER_RATE))
    cursorAttention = np.float32(min(max(params.CURSOR_ATTENTION, 0.0), 1.0))
    cursorRadius = np.float32(max(1.0, params.CURSOR_RADIUS))


#######################################
# Kernels
#######################################

# Update Boid Table with new cell for every agent
@cuda.jit(fastmath=True)
def fillBoidPositionTable(boids: DeviceNDArray, boidPositionTable: DeviceNDArray, renderData: DeviceNDArray) -> None:
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    x = int(boids[index,0]//GRID_CELL_SIZE) % GRID_WIDTH
    y = int(boids[index,1]//GRID_CELL_SIZE) % GRID_HEIGHT
    gridIndex = x + y * GRID_WIDTH
    boidPositionTable[index,0] = index
    boidPositionTable[index,1] = gridIndex
    renderData[index, 2] = 0.0


# Update Index Tables to give start and end (exclusive) index for every cell on boidPositionTable
@cuda.jit(fastmath=True)
def fillCellIndexTable(boidPositionTable: DeviceNDArray, cellIndexTable: DeviceNDArray, cellEndTable: DeviceNDArray) -> None:
    index = cuda.grid(1)
    if index >= POPULATION:
        return
    gridIndex = boidPositionTable[index,1]
    # First agent of its cell
    if index == 0 or boidPositionTable[index-1,1] != gridIndex:
        cellIndexTable[gridIndex] = index
    # Last agent of its cell
    if index == POPULATION-1 or boidPositionTable[index+1,1] != gridIndex:
        cellEndTable[gridIndex] = index + 1

# One block per cell: its threads take the agents of the cell (TILE_SIZE at a time) and
# go through the agents of the 9 neighbor cells together, seen as one list of candidates.
# Those are loaded in shared memory one tile at a time, and every thread tests its own agent
# against the tile. All threads of a block read the same candidate at the same time and loop
# the same number of times. The three rules are then applied to all of an agent's neighbors at once.
# When there are more than CANDIDATE_BUDGET candidates, only every stride-th one is tested,
# from a random offset that changes every frame: every candidate has the same chance to be
# tested whatever its cell or position, so the sample isn't biased in any direction
# sortedBoids -> steering
@cuda.jit(fastmath=True)
def neighborSearch(sortedBoids: DeviceNDArray, steering: DeviceNDArray, lookUpTable: DeviceNDArray, cellIndexTable: DeviceNDArray,
        cellEndTable: DeviceNDArray, boidPositionTable: DeviceNDArray, renderData: DeviceNDArray, frameCounter: DeviceNDArray,
        SPOTLIGHT: bool, WRAP_AROUND: bool, COHESION: np.float32, ALIGNMENT: np.float32, SEPARATION: np.float32,
        SEPARATION_DIST_SQUARED: np.float32, SEPARATION_FALLOFF: np.float32) -> None:
    cell = cuda.blockIdx.x
    thread = cuda.threadIdx.x
    cellStart, cellEnd = cellIndexTable[cell], cellEndTable[cell]
    # Blocks without agents of their own (the whole block returns together)
    if cellStart + cuda.blockIdx.y * TILE_SIZE >= cellEnd:
        return
    # Candidate list: the agents of the 9 neighbor cells, one after the other.
    # listStart[k] is where cell k starts in the list, cellFirst[k] where it starts in sortedBoids
    listStart = cuda.shared.array(10, int32)
    cellFirst = cuda.shared.array(9, int32)
    if thread == 0:
        listStart[0] = 0
        for k in range(9):
            neighborCell = lookUpTable[cell, k]
            cellFirst[k] = cellIndexTable[neighborCell]
            listStart[k+1] = listStart[k] + cellEndTable[neighborCell] - cellIndexTable[neighborCell]
    cuda.syncthreads()
    numCandidates = listStart[9]
    # Sample every stride-th candidate from a random offset when over the budget
    stride = 1
    offset = 0
    if numCandidates > CANDIDATE_BUDGET:
        stride = (numCandidates + CANDIDATE_BUDGET - 1) // CANDIDATE_BUDGET
        offset = randomHash(cell, cuda.blockIdx.y, frameCounter[0]) % stride
    numSampled = (numCandidates - offset + stride - 1) // stride
    # Candidates (x, y, dx, dy) and their slots, shared by the threads of the block
    tile = cuda.shared.array((TILE_SIZE, 4), float32)
    tileSlot = cuda.shared.array(TILE_SIZE, int32)
    # Agents of the cell, TILE_SIZE at a time, shared between the blocks of the cell
    # (the whole block skips the loop together when it has no agents)
    for first in range(cellStart + cuda.blockIdx.y * TILE_SIZE, cellEnd, TILE_SIZE * CELL_BLOCKS):
        slot = first + thread
        # Threads past the end of the cell still help loading tiles
        active = slot < cellEnd
        index = -1
        x, y, dx, dy = float32(0.0), float32(0.0), float32(0.0), float32(0.0)
        if active:
            # Agent id, only needed for the spotlight
            index = boidPositionTable[slot,0]
            # Current  position
            x, y = sortedBoids[slot,0], sortedBoids[slot,1]
            dx, dy = sortedBoids[slot,2], sortedBoids[slot,3]
        # Squared length of the direction, for the field of view test
        directionSquared = dx**2 + dy**2
        # Separation radius
        separationDist = math.sqrt(SEPARATION_DIST_SQUARED)
        # The linear falloff (1, the default) skips pow, which is slow: this runs for most candidates in dense flocks
        powFalloff = SEPARATION_FALLOFF != float32(1.0)
        # Boid direction
        resX, resY = float32(0.0), float32(0.0)
        numNeighbors = 0
        alignmentX, alignmentY = float32(0.0), float32(0.0)
        cohesionX, cohesionY = float32(0.0), float32(0.0)
        separationX, separationY = float32(0.0), float32(0.0)
        numSeparation = 0
        #Check if close to edge (for distanche check)
        if WRAP_AROUND and (x < NEIGHBOR_DIST or y < NEIGHBOR_DIST or WIDTH - x < NEIGHBOR_DIST or HEIGHT - y < NEIGHBOR_DIST):
            onEdge = True
        else:
            onEdge = False
        for tileStart in range(0, numSampled, TILE_SIZE):
            # Load a tile of candidates, one per thread
            i = tileStart + thread
            if i < numSampled:
                # Position in the candidate list, then in sortedBoids
                m = offset + i * stride
                k = 0
                while listStart[k+1] <= m:
                    k += 1
                candidate = cellFirst[k] + m - listStart[k]
                tileSlot[thread] = candidate
                for j in range(4):
                    tile[thread, j] = sortedBoids[candidate, j]
            cuda.syncthreads()
            if active:
                for t in range(min(TILE_SIZE, numSampled - tileStart)):
                    other = tileSlot[t]
                    # Get agent data and check if it's a neighbor
                    if other != slot:
                        ax = tile[t,0]
                        ay = tile[t,1]
                        adx = tile[t,2]
                        ady = tile[t,3]
                        if onEdge:
                            (ax, ay) = minimalToroidalDistance(x,y,ax,ay)
                        distX = x - ax
                        distY = y - ay
                        distLengthSquared = distX**2 + distY**2
                        isTooClose = (distLengthSquared < SEPARATION_DIST_SQUARED)
                        isNeighbor = (distLengthSquared < NEIGHBOR_DIST_SQUARED) and isInFOV(dx,dy,distX,distY,directionSquared,distLengthSquared)
                        # SPOTLIGHT
                        if SPOTLIGHT and index == 0:
                            # Agent spotlight
                            agent = boidPositionTable[other,0]
                            renderData[index, 2] = 1.0
                            if isTooClose:
                                renderData[agent, 2] = 2.0
                            elif isNeighbor:
                                renderData[agent, 2] = 3.0
                            else:
                                renderData[agent, 2] = 4.0
                        # Collect agent data
                        if isNeighbor:
                            numNeighbors += 1
                            alignmentX += adx
                            alignmentY += ady
                            cohesionX += ax
                            cohesionY += ay
                            if isTooClose:
                                # Push away from the other agent, stronger the closer it is: 1 when touching,
                                # fading to 0 at the separation radius. SEPARATION_FALLOFF sets how fast
                                # (0: same push at any distance, 1: linear, higher: mostly at close range)
                                # (distLengthSquared > 0 here, as agents at the same position aren't neighbors)
                                dist = math.sqrt(distLengthSquared)
                                push = float32(1.0) - dist / separationDist
                                if powFalloff:
                                    push = math.pow(push, SEPARATION_FALLOFF)
                                push /= dist # (distX, distY) to unit length
                                separationX += distX * push
                                separationY += distY * push
                                numSeparation += 1
            # The tile is overwritten next
            cuda.syncthreads()
        if numNeighbors > 0:
            # Apply boid rules
            # Alignment
            # Normalize alignment vector
            l = math.sqrt(alignmentX**2 + alignmentY**2)
            alignmentX, alignmentY = alignmentX / l, alignmentY / l
            # Add alignment vector to direction
            resX += alignmentX * ALIGNMENT
            resY += alignmentY * ALIGNMENT
            # Cohesion
            # Get vector towards center of mass
            cohesionX /= float32(numNeighbors)
            cohesionY /= float32(numNeighbors)
            cohesionX -= x
            cohesionY -= y
            # Normalize cohesion vector
            l = math.sqrt(cohesionX**2 + cohesionY**2)
            cohesionX, cohesionY = cohesionX / l, cohesionY / l
            # Add cohesion vector to direction
            resX += cohesionX * COHESION
            resY += cohesionY * COHESION
            # Separation
            if numSeparation > 0:
                # The pushes add up, so closer and more crowded agents push harder, up to a length
                # of 1 (like the other rules): larger pushes made agents flip back and forth.
                # When only one candidate out of stride was tested, the sum is scaled back up first
                separationX *= float32(stride)
                separationY *= float32(stride)
                l = math.sqrt(separationX**2 + separationY**2)
                if l > float32(1.0):
                    separationX, separationY = separationX / l, separationY / l
                # Add separation vector to direction
                resX += separationX * SEPARATION
                resY += separationY * SEPARATION
        # Write to steering buffer at sorted slot
        if active:
            steering[slot, 0] = resX
            steering[slot, 1] = resY

# Reads the steering direction given by the rules
# Gets final direction and position and writes to memory (back in agent order)
# sortedBoids, steering -> boids, renderData
@cuda.jit(fastmath=True)
def writeData(boids: DeviceNDArray, sortedBoids: DeviceNDArray, steering: DeviceNDArray, boidPositionTable: DeviceNDArray,
        renderData: DeviceNDArray, timeStep: DeviceNDArray, frameCounter: DeviceNDArray, wanderAngles: DeviceNDArray,
        cursor: DeviceNDArray, WRAP_AROUND: bool, SPEED: np.float32, JITTER: np.float32, WANDER: np.float32,
        WANDER_RATE: np.float32, CURSOR_ATTENTION: np.float32, CURSOR_RADIUS: np.float32) -> None:
    slot = cuda.grid(1)
    if slot >= POPULATION:
        return
    index = boidPositionTable[slot,0]
    x, y = sortedBoids[slot,0], sortedBoids[slot,1]
    # Previous direction
    dx, dy = sortedBoids[slot,2], sortedBoids[slot,3]
    # Direction given by the rules
    steerX, steerY = steering[slot,0], steering[slot,1]
    # Frames at REFERENCE_FPS since the last update. The rule below is made for one such frame:
    # it's applied once per whole frame, and partly for what remains, so that agents turn and
    # move the same way whatever the frame rate (the rules themselves are evaluated once per frame)
    remaining = timeStep[0]
    MARGIN = float32(200.0)
    # Sub-step number, for the random numbers
    subStep = 0
    wanderAngle = wanderAngles[index]
    cursorX, cursorY, cursorMode = cursor[0], cursor[1], cursor[2]
    while remaining > float32(0.0):
        part = min(remaining, float32(1.0))
        remaining -= float32(1.0)
        subStep += 1
        # If Edge Wrapping is off, avoid walls
        wallX, wallY = float32(0.0), float32(0.0)
        if not WRAP_AROUND:
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
                wallX = -wallX
            if y < MARGIN:
                wallY = MARGIN - y
                wallY /= MARGIN
                wallY = wallY ** 2
            elif y2 < MARGIN:
                wallY = MARGIN - y2
                wallY /= MARGIN
                wallY = wallY ** 2
                wallY = -wallY
        sx = steerX + wallX / float32(10.0)
        sy = steerY + wallY / float32(10.0)
        # Wander (Reynolds): steer towards a target on a circle ahead of the agent, which drifts
        # a little around the circle every frame (random walk, WANDER_RATE radians of standard
        # deviation, sqrt(part) of it on part of a frame). The target moves with the agent's
        # direction, so an agent keeps turning the same way for a while (course commitment)
        if WANDER > float32(0.0):
            wanderAngle += WANDER_RATE * math.sqrt(part) * randomGaussian(index, frameCounter[0], 3 * subStep)
            if wanderAngle > float32(math.pi):
                wanderAngle -= float32(2.0 * math.pi)
            elif wanderAngle < float32(-math.pi):
                wanderAngle += float32(2.0 * math.pi)
            length = math.sqrt(dx**2 + dy**2)
            if length > float32(0.0):
                hx, hy = dx / length, dy / length
                cosAngle, sinAngle = math.cos(wanderAngle), math.sin(wanderAngle)
                tx = hx + WANDER_RADIUS * (hx * cosAngle - hy * sinAngle)
                ty = hy + WANDER_RADIUS * (hx * sinAngle + hy * cosAngle)
                tl = math.sqrt(tx**2 + ty**2)
                # Desired direction (at full speed) minus current direction
                sx += WANDER * (tx / tl - dx)
                sy += WANDER * (ty / tl - dy)
        # Update directions
        ndx = dx + sx * float32(3.0) # Weight w, ratio of 1:w with old/new direction
        ndy = dy + sy * float32(3.0)
        # Clamp speed
        l = math.sqrt(ndx**2 + ndy**2)
        if l > float32(1.0):
            ndx, ndy = ndx / l, ndy / l
        # Cursor: takes a share of the agent's attention, CURSOR_ATTENTION next to the cursor and fading
        # to 0 at CURSOR_RADIUS. The new direction is that share of "straight towards the cursor (or away
        # from it, cursorMode -1), at full speed", and the rest of what the rules gave
        if cursorMode != float32(0.0):
            toCursorX, toCursorY = cursorX - x, cursorY - y
            if WRAP_AROUND:
                # Shortest way, over the edges
                if toCursorX > HALF_WIDTH:
                    toCursorX -= WIDTH
                elif toCursorX < -HALF_WIDTH:
                    toCursorX += WIDTH
                if toCursorY > HALF_HEIGHT:
                    toCursorY -= HEIGHT
                elif toCursorY < -HALF_HEIGHT:
                    toCursorY += HEIGHT
            cursorDist = math.sqrt(toCursorX**2 + toCursorY**2)
            if cursorDist < CURSOR_RADIUS and cursorDist > float32(0.0):
                attention = CURSOR_ATTENTION * (float32(1.0) - cursorDist / CURSOR_RADIUS)
                towards = cursorMode / cursorDist
                ndx += (toCursorX * towards - ndx) * attention
                ndy += (toCursorY * towards - ndy) * attention
        # Part of a frame: only part of the way to the new direction
        if part < float32(1.0):
            ndx = dx + (ndx - dx) * part
            ndy = dy + (ndy - dy) * part
        # Jitter: a new random push every frame (Gaussian in x and y, weighted like the rules;
        # sqrt(part) of it on part of a frame, so that the pushes add up the same way whatever the frame rate)
        if JITTER > float32(0.0):
            kick = float32(3.0) * JITTER * math.sqrt(part)
            ndx += kick * randomGaussian(index, frameCounter[0], 3 * subStep + 1)
            ndy += kick * randomGaussian(index, frameCounter[0], 3 * subStep + 2)
            # Clamp speed
            l = math.sqrt(ndx**2 + ndy**2)
            if l > float32(1.0):
                ndx, ndy = ndx / l, ndy / l
        dx, dy = ndx, ndy
        # Move position
        move = SPEED * part
        x += dx * move
        y += dy * move
        # Edge wrap
        # (keeps the part of the move past the edge)
        if WRAP_AROUND:
            if x >= WIDTH:
                x -= WIDTH
            elif x < float32(0.0):
                x += WIDTH
            if y >= HEIGHT:
                y -= HEIGHT
            elif y < float32(0.0):
                y += HEIGHT
    if WANDER > float32(0.0):
        wanderAngles[index] = wanderAngle
    # Render coordinates
    rx = (x - HALF_WIDTH) / HALF_WIDTH
    ry = (y - HALF_HEIGHT) / HALF_HEIGHT
    # WRITE TO MEMORY
    # Write screen position in [-1,1]
    renderData[index,0] = rx
    renderData[index,1] = ry
    # renderData[index,2] = ry
    # Store data for next calculation
    boids[index,0] = x
    boids[index,1] = y
    boids[index,2] = dx
    boids[index,3] = dy

# If agent is near an edge and WrapAround is set to True
# Checks if other agent is a neighbor for all neighboring cell configurations (9)
# Looks for minimal toroidal distance and returns perceived position
# Given what is provided by Numba we have to get the minimum manually
@cuda.jit(device=True, fastmath=True)
def minimalToroidalDistance(x: np.float32, y: np.float32, ax: np.float32, ay: np.float32) -> tuple[np.float32, np.float32]:
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

# True if the other agent is outside the blind spot, i.e. cos(angle between the direction and
# the vector from the other agent) < 0.8. Without sqrt or division: when the dot product is
# negative the cosine is too, otherwise both sides can be squared
@cuda.jit(device=True, fastmath=True)
def isInFOV(dx: np.float32, dy: np.float32, distX: np.float32, distY: np.float32, directionSquared: np.float32, distLengthSquared: np.float32) -> np.bool_:
    dot = dx * distX + dy * distY
    return dot < float32(0.0) or dot**2 < float32(0.64) * directionSquared * distLengthSquared

# Integer hash ("lowbias32"), gives a well mixed non-negative number from three integers
@cuda.jit(device=True)
def randomHash(a: int, b: int, c: int) -> int:
    h = (a * 73856093 + b * 19349663 + c * 83492791) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x7feb352d) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x846ca68b) & 0xFFFFFFFF
    h ^= h >> 16
    return h

# Standard normal random number (Box-Muller transform of two uniform numbers from randomHash)
@cuda.jit(device=True, fastmath=True)
def randomGaussian(a: int, b: int, c: int) -> float:
    seed = randomHash(a, b, c)
    u1 = float32((randomHash(seed, 1, 0) >> 8) + 1) / float32(16777216.0) # (0, 1], so the log is finite
    u2 = float32(randomHash(seed, 2, 0) >> 8) / float32(16777216.0)       # [0, 1)
    return math.sqrt(float32(-2.0) * math.log(u1)) * math.cos(float32(2.0 * math.pi) * u2)
