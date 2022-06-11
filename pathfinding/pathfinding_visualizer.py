import time

import cv2 as cv
import pygame
from queue import Queue, PriorityQueue
import random

#implement more maze generating algorithms?
#implement multi agent pathfinding algorithms?
#implement weighted nodes(otherwise, breadth first search is dijkstra)
#in dfs(and bfs and others) implement what if no path is found
#optimize bfs(and others) perhaps by rather than checking and skipping a cur if_visited, how about check if a Node is in queue before adding it to the queue
#optimize all by examining necessary vs unneccesary or improvable if statements etc
#optimize by implementing a for neigbour in curr.neighbours style instead of repeated checks
#optimize making final path
#imp note: people use a visited_set to keep track of what is already in queue, but whether element is in queue or not can be directly checked, and time complexity seems to be the same but space used is less, hence i use this method

WINDOW_WIDTH = 800
tile_width = 800//50
WIN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_WIDTH))
pygame.display.set_caption("Pathfinding Visualizer")

RED = (255, 0, 0) #menu line color
WHITE = (255, 255, 255) #default background
BLACK = (0, 0, 0) #wall
GREEN = (0, 255, 0) #final path color
# BLUE = (0, 0, 255) end color for visited
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128) #start color for visited
ORANGE = (255, 165, 0) #start node
GREY = (128, 128, 128) #walls/lines of grid
TURQUOISE = (64, 224, 208) #end node

timepause = 0.000001

class Node:
    def __init__(self, row, col):
        self.x = col * tile_width
        self.y = (49-row) * tile_width
        self.color = WHITE
        self.width = tile_width
        self.row = row
        self.col = col
        self.visited = False

    def visit(self):
        self.visited = True
        self.color = PURPLE

    def is_visited(self):
        return self.visited

    def make_start(self):
        self.color = ORANGE
        self.visited = False

    def is_start(self):
        return self.color == ORANGE

    def make_end(self):
        self.color = TURQUOISE
        self.visited = False

    def make_maze_path(self):
        self.color = RED

    def is_end(self):
        return self.color == TURQUOISE

    def make_path(self):
        self.color = GREEN
        self.visited = False

    def is_path(self):
        return self.color == GREEN

    def make_wall(self):
        self.color = BLACK
        self.visited = False

    def is_wall(self):
        return self.color == BLACK

    def reset(self):
        self.color = WHITE
        self.visited = False

    def get_pos(self):
        return self.col,self.row

    def draw_itself(self, window):
        if self.visited:
            r,g,b = self.color
            if r>1:
                self.color = r-2, g, b
            if b<254:
                self.color = r,g,b+2
        pygame.draw.rect(window, self.color, (self.x, self.y, tile_width, tile_width))

def make_grid():
    grid = []
    for rownumber in range(50):
        grid.append([])
        for colnumber in range(50):
            tile_var = Node(rownumber,colnumber)
            grid[rownumber].append(tile_var)
    return grid

def draw_grid_lines(window):
    for i in range(50):
        pygame.draw.line(window, GREY, (0, i * tile_width), (WINDOW_WIDTH, i * tile_width))
        pygame.draw.line(window, GREY, (i * tile_width, 0), (i * tile_width, WINDOW_WIDTH))

def draw(window, grid):
    for row in grid:
        for tile_var in row:
            tile_var.draw_itself(window)
    draw_grid_lines(window)
    pygame.display.update()

def coords_to_tileno(x,y):
    x = x//tile_width
    y = y//tile_width
    y = 49-y
    return x,y

def retain_and_reset_grid(grid):
    for row in grid:
        for tile_var in row:
            if not tile_var.is_wall() and not tile_var.is_start() and not tile_var.is_end():
                tile_var.reset()

def heuristic(x1,y1,x2,y2):
    return abs(x1-x2)+abs(y1-y2)

def execute_depthfs(draw, grid, start, end):
    global timepause
    retain_and_reset_grid(grid)
    path = []
    path.append(start)
    curr = start
    while True:
        time.sleep(timepause)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_j:
                    timepause = 0.000001
                if event.key == pygame.K_k:
                    timepause = 0.003
                if event.key == pygame.K_l:
                    timepause = 0.01
        if curr == end:
            break
        if not curr == start and not curr.is_visited():
            curr.visit()
            path.append(curr)
        draw()
        x,y = curr.get_pos()
        if x+1<50 and (not grid[y][x+1].is_wall()) and (not grid[y][x+1].is_visited()) and (not grid[y][x+1].is_start()):
            curr = grid[y][x+1]
        elif y-1>=0 and (not grid[y-1][x].is_wall()) and (not grid[y-1][x].is_visited()) and (not grid[y-1][x].is_start()):
            curr = grid[y-1][x]
        elif x-1>=0 and (not grid[y][x-1].is_wall()) and (not grid[y][x-1].is_visited()) and (not grid[y][x-1].is_start()):
            curr = grid[y][x-1]
        elif y+1<50 and (not grid[y+1][x].is_wall()) and (not grid[y+1][x].is_visited()) and (not grid[y+1][x].is_start()):
            curr = grid[y+1][x]
        else:
            path.remove(path[-1])
            if len(path) == 0:
                return False
            curr = path[-1]

    path.remove(start)
    for node_var in path:
        node_var.make_path()
        draw()
    return True

def execute_breadthfs_dijkstra(draw, grid, start, end):
    global timepause
    retain_and_reset_grid(grid)
    source = {}
    q = Queue()
    q.put(start)
    while not q.empty():
        time.sleep(timepause)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_j:
                    timepause = 0.000001
                if event.key == pygame.K_k:
                    timepause = 0.003
                if event.key == pygame.K_l:
                    timepause = 0.01
        curr = q.get()
        x,y = curr.get_pos()
        if curr == end:
            break
        if not curr == start:
            curr.visit()
        draw()
        if x+1<50 and (not grid[y][x+1].is_wall()) and (not grid[y][x+1].is_visited()) and (not grid[y][x+1].is_start()) and (not grid[y][x+1] in q.queue):
            q.put(grid[y][x+1])
            source[grid[y][x+1]] = curr
        if y-1>=0 and (not grid[y-1][x].is_wall()) and (not grid[y-1][x].is_visited()) and (not grid[y-1][x].is_start()) and (not grid[y-1][x] in q.queue):
            q.put(grid[y-1][x])
            source[grid[y-1][x]] = curr
        if x-1>=0 and (not grid[y][x-1].is_wall()) and (not grid[y][x-1].is_visited()) and (not grid[y][x-1].is_start()) and (not grid[y][x-1] in q.queue):
            q.put(grid[y][x-1])
            source[grid[y][x-1]] = curr
        if y+1<50 and (not grid[y+1][x].is_wall()) and (not grid[y+1][x].is_visited()) and (not grid[y+1][x].is_start()) and (not grid[y+1][x] in q.queue):
            q.put(grid[y+1][x])
            source[grid[y+1][x]] = curr
    path = Queue()
    if not curr == end:
        return False
    if curr == end:
        curr = source[curr]
        while not curr == start:
            path.put(curr)
            curr = source[curr]
        while not path.empty():
            curr = path.get()
            curr.make_path()
            draw()
    return True

def execute_greedy_bestfs(draw, grid, start, end):
    global timepause
    retain_and_reset_grid(grid)
    q = PriorityQueue()
    curr = start
    xend, yend = end.get_pos()
    order = 0
    source = {}
    q.put((0,order,start))
    while not q.empty():
        time.sleep(timepause)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_j:
                    timepause = 0.000001
                if event.key == pygame.K_k:
                    timepause = 0.003
                if event.key == pygame.K_l:
                    timepause = 0.01
        curr = q.get()[2]
        if curr == end:
            break
        if not curr == start and not curr.is_visited():
            curr.visit()
        draw()
        x,y = curr.get_pos()
        if x+1<50 and (not grid[y][x+1].is_wall()) and (not grid[y][x+1].is_visited()) and (not grid[y][x+1].is_start()) and (not any(grid[y][x+1] in item for item in q.queue)):
            order+=1
            q.put((heuristic(x+1,y,xend,yend),order,grid[y][x+1]))
            source[grid[y][x+1]] = curr
        if y-1>=0 and (not grid[y-1][x].is_wall()) and (not grid[y-1][x].is_visited()) and (not grid[y-1][x].is_start()) and (not any(grid[y-1][x] in item for item in q.queue)):
            order+=1
            q.put((heuristic(x,y-1,xend,yend),order,grid[y-1][x]))
            source[grid[y-1][x]] = curr
        if x-1>=0 and (not grid[y][x-1].is_wall()) and (not grid[y][x-1].is_visited()) and (not grid[y][x-1].is_start()) and (not any(grid[y][x-1] in item for item in q.queue)):
            order+=1
            q.put((heuristic(x-1,y,xend,yend),order,grid[y][x-1]))
            source[grid[y][x-1]] = curr
        if y+1<50 and (not grid[y+1][x].is_wall()) and (not grid[y+1][x].is_visited()) and (not grid[y+1][x].is_start()) and (not any(grid[y+1][x] in item for item in q.queue)):
            order+=1
            q.put((heuristic(x,y+1,xend,yend),order,grid[y+1][x]))
            source[grid[y+1][x]] = curr

    if not curr == end:
        return False
    curr = source[curr]
    path = Queue()
    while not curr == start:
        path.put(curr)
        curr = source[curr]
    while not path.empty():
        curr = path.get()
        curr.make_path()
        draw()
    return True

def remove(q,element1):
    newq = PriorityQueue()
    for (f_score, heurist, order, nodevar) in q.queue:
        if nodevar == element1:
            continue
        newq.put((f_score, heurist, order, nodevar))
    return newq

def execute_A_star(draw, grid, start, end):
    retain_and_reset_grid(grid)
    order = 0
    q = PriorityQueue()
    global timepause
    source = {}
    g_score = {tile_var: float("inf") for row in grid for tile_var in row}
    g_score[start] = 0
    f_score = {tile_var: float("inf") for row in grid for tile_var in row}
    x,y = start.get_pos()
    xend, yend = end.get_pos()
    f_score[start] = heuristic(x,y,xend,yend)
    q.put((g_score[start],heuristic(x,y,xend,yend), order, start))
    while not q.empty():
        time.sleep(timepause)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_j:
                    timepause = 0.000001
                if event.key == pygame.K_k:
                    timepause = 0.003
                if event.key == pygame.K_l:
                    timepause = 0.01
        #implement time delay and ability to change time delay or quit programme in middle of programme execution?
        curr = q.get()[3]
        if curr == end:
            break
        x,y = curr.get_pos()
        new_g_score = g_score[curr] + 1
        if x+1<50 and (not grid[y][x+1].is_wall()):
            if new_g_score < g_score[grid[y][x+1]]:
                source[grid[y][x+1]] = curr
                g_score[grid[y][x+1]] = new_g_score
                f_score[grid[y][x+1]] = new_g_score+heuristic(x+1,y,xend,yend)
                if not grid[y][x+1].is_visited() and not grid[y][x+1].is_start():
                    if not any(grid[y][x+1] in item for item in q.queue):
                        order+=1
                        q.put((f_score[grid[y][x+1]],heuristic(x+1,y,xend,yend),order,grid[y][x+1]))
                    else: #this removes and readds neighbour from queue and to queue to account for updated f_score
                        order+=1
                        q = remove(q, grid[y][x+1])
                        q.put((f_score[grid[y][x+1]],heuristic(x+1,y,xend,yend),order,grid[y][x+1]))
        if y-1>=0 and (not grid[y-1][x].is_wall()):
            if new_g_score < g_score[grid[y-1][x]]:
                source[grid[y-1][x]] = curr
                g_score[grid[y-1][x]] = new_g_score
                f_score[grid[y-1][x]] = new_g_score+heuristic(x,y-1,xend,yend)
                if not grid[y-1][x].is_visited() and not grid[y-1][x].is_start():
                    if not any(grid[y-1][x] in item for item in q.queue):
                        order+=1
                        q.put((f_score[grid[y-1][x]],heuristic(x,y-1,xend,yend),order,grid[y-1][x]))
                    else:
                        order+=1
                        q = remove(q, grid[y-1][x])
                        q.put((f_score[grid[y-1][x]],heuristic(x,y-1,xend,yend),order,grid[y-1][x]))
        if x-1>=0 and (not grid[y][x-1].is_wall()):
            if new_g_score < g_score[grid[y][x-1]]:
                source[grid[y][x-1]] = curr
                g_score[grid[y][x-1]] = new_g_score
                f_score[grid[y][x-1]] = new_g_score+heuristic(x-1,y,xend,yend)
                if not grid[y][x-1].is_visited() and not grid[y][x-1].is_start():
                    if not any(grid[y][x-1] in item for item in q.queue):
                        order+=1
                        q.put((f_score[grid[y][x-1]],heuristic(x-1,y,xend,yend),order,grid[y][x-1]))
                    else:
                        order+=1
                        q = remove(q, grid[y][x-1])
                        q.put((f_score[grid[y][x-1]],heuristic(x-1,y,xend,yend),order,grid[y][x-1]))
        if y+1<50 and (not grid[y+1][x].is_wall()):
            if new_g_score < g_score[grid[y+1][x]]:
                source[grid[y+1][x]] = curr
                g_score[grid[y+1][x]] = new_g_score
                f_score[grid[y+1][x]] = new_g_score+heuristic(x,y+1,xend,yend)
                if not grid[y+1][x].is_visited() and not grid[y+1][x].is_start():
                    if not any(grid[y+1][x] in item for item in q.queue):
                        order+=1
                        q.put((f_score[grid[y+1][x]],heuristic(x,y+1,xend,yend),order,grid[y+1][x]))
                    else:
                        order+=1
                        q = remove(q, grid[y+1][x])
                        q.put((f_score[grid[y+1][x]],heuristic(x,y+1,xend,yend),order,grid[y+1][x]))
        draw()
        if not curr == start:
            curr.visit()
    if not curr == end:
        return False
    curr = source[curr]
    path = Queue()
    while not curr == start:
        path.put(curr)
        curr = source[curr]
    while not path.empty():
        curr = path.get()
        curr.make_path()
        draw()
    return True

def generate_neighbours(grid, tile_var, visited):
    x,y = tile_var.get_pos()
    neighbours = []
    if x + 2 < 50 and not grid[y][x + 2] in visited:
        neighbours.append(grid[y][x+2])
    if y - 2 >= 0 and not grid[y - 2][x] in visited:
        neighbours.append(grid[y-2][x])
    if x - 2 >= 0 and not grid[y][x - 2] in visited:
        neighbours.append(grid[y][x-2])
    if y + 2 < 50 and not grid[y + 2][x] in visited:
        neighbours.append(grid[y+2][x])
    return neighbours

def tile_between(tile1, tile2, grid):
    x1,y1 = tile1.get_pos()
    x2,y2 = tile2.get_pos()
    if x1==x2:
        x,y =  x1,((y1+y2)//2)
    else:
        x,y = ((x1+x2)//2),y1
    return grid[y][x]

def make_recursive_backtracking_maze(window):
    grid = make_grid()
    global timepause
    # for rownumber in range(50):
    #     for col in range(50):
    #         if (rownumber+col)%2==1:
    #             grid[rownumber][col].make_wall()
    for rownumber in range(50):
        for col in range(50):
            if rownumber%2==1 or col%2==1:
                grid[rownumber][col].make_wall()
    visited = []
    starter = grid[0][0]
    curr = starter
    path = []
    path.append(curr)
    visited.append(curr)
    curr.make_maze_path()
    while True:
        time.sleep(timepause)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_j:
                    timepause = 0.000001
                if event.key == pygame.K_k:
                    timepause = 0.003
                if event.key == pygame.K_l:
                    timepause = 0.01
        while len(generate_neighbours(grid,curr,visited))==0:
            if curr == starter:
                break
            else:
                path[-1].reset()
                tile_between(path[-1],path[-2],grid).reset()
                path.remove(path[-1])
                curr = path[-1]
                draw(window, grid)
        if curr == starter and len(generate_neighbours(grid,curr,visited))==0:
            break
        temp = random.choice(generate_neighbours(grid,curr,visited))
        tile_between(curr, temp , grid).make_maze_path()
        curr = temp
        path.append(curr)
        curr.make_maze_path()
        visited.append(curr)
        draw(window, grid)
    starter.reset()
    draw(window,grid)
    return grid

#main
grid = make_grid()
running = True
xcord = 0
ycord = 0
start = None
end = None

draw(WIN,grid)
img = cv.imread("welcome.png")
cv.imshow("welcome!",img)
cv.waitKey(0)
cv.destroyAllWindows()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        xcord, ycord = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed()[0]:
            #left button down
            x,y = coords_to_tileno(xcord,ycord)
            tile_under_mouse = grid[y][x]
            if not start and tile_under_mouse != end:
                start = tile_under_mouse
                start.make_start()
            elif not end and tile_under_mouse != start:
                end = tile_under_mouse
                end.make_end()
            elif tile_under_mouse != end and tile_under_mouse != start:
                tile_under_mouse.make_wall()
        elif pygame.mouse.get_pressed()[2]:
            #right button down
            x,y = coords_to_tileno(xcord,ycord)
            tile_under_mouse = grid[y][x]
            if tile_under_mouse == start:
                start = None
            elif tile_under_mouse == end:
                end = None
            tile_under_mouse.reset()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1 and end and start:
                execute_depthfs(lambda: draw(WIN, grid), grid, start, end)
            if event.key == pygame.K_2 and end and start:
                execute_breadthfs_dijkstra(lambda: draw(WIN, grid), grid, start, end)
            if event.key == pygame.K_3 and end and start:
                execute_greedy_bestfs(lambda: draw(WIN, grid), grid, start, end)
            if event.key == pygame.K_4 and end and start:
                execute_A_star(lambda: draw(WIN, grid), grid, start, end)
            if event.key == pygame.K_5:
                start = None
                end = None
                grid = make_recursive_backtracking_maze(WIN)
            if event.key == pygame.K_r:
                start = None
                end = None
                grid = make_grid()
            if event.key == pygame.K_j:
                timepause = 0.000001
            if event.key == pygame.K_k:
                timepause = 0.003
            if event.key == pygame.K_l:
                timepause = 0.01
    draw(WIN,grid)
pygame.quit()
