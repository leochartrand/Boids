import util

# Grille de partitionnment spatial
size = 100

def init(w, h, l):
    global size, cells, width, height, boidList
    size = 100
    cells = {}
    width = w//size
    height = h//size
    boidList = l

def getCell(pos):
    global size
    return (pos[0]//size, pos[1]//size)

def add(index, cell):
    global cells
    if cell in cells:
        cells[cell].append(index)
    else:
        cells[cell] = [index]

def remove(index, cell):
    global cells
    if cell in cells:
        if index in cells[cell]:
            cells[cell].remove(index)

def getCellNeighbors(index, cell):
    global cells, width, height, boidList
    friends, strangers, predators = [], [], []
    tileOffset = {-1, 0, 1}
    if cell in cells:
        for i in tileOffset:
            for i in tileOffset:
                x = (cell[0] + i)%width
                y = (cell[1] + i)%height
                # neighbors += cells.get((x,y),[])
                for agent in cells.get((x,y),[]):
                    if agent != index and boidList[agent].position.distance_to(boidList[index].position) < 100:
                        if boidList[agent].species == boidList[index].species:
                            friends.append(boidList[agent])
                        elif agent.species == util.Species.PREDATOR:
                            predators.append(boidList[agent])
                        else:
                            strangers.append(boidList[agent])
    return friends, strangers, predators

