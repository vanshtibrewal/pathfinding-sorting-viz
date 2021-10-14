import time
import pygame
import random

#improve the colors maybe(like green for final pos, and maybe orange for currently being compared? or green for both? idk
#implement a scaling timepause for all 4 algorithms and try a variety to see what works, including re evaluating bubble sort timepause

WINDOW_WIDTH = 800
WIN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_WIDTH))
pygame.display.set_caption("Sorting Visualizer")

WHITE = (255, 255, 255) #default background
GREEN = (0, 255, 0) #color of the 2 bars being compared
BLUE = (0, 0, 255) #default color of a bar
PURPLE = (128, 0, 128) #color all bars turn after the whole sorting is done
ORANGE = (255, 165, 0) #color of bar when it's in final position
YELLOW = (255,255,0) #pivot

orgmergarr = []
orgquickarr = []

n = 100

bubbletimepause = 10/(n*n)
mergetimepause = 0.01
quicktimepause = 0.005
heaptimepause = 0.01

constfactor = WINDOW_WIDTH // n
leftover = WINDOW_WIDTH % n

class Bar:
    def __init__(self,val):
        self.color = BLUE
        self.val = val

    def set_val(self,val):
        self.val = val

    def get_val(self):
        return self.val

    def highlight(self):
        self.color = GREEN

    def reset(self):
        self.color = BLUE

    def make_final(self):
        self.color = PURPLE

    def get_color(self):
        return self.color

    def make_placed(self):
        self.color = ORANGE

    def make_pivot(self):
        self.color = YELLOW

def make_array():
    global n
    array = []
    for i in range(n):
        array.append(Bar(i))
    random.shuffle(array)
    return array

def draw(window, array):
    global n
    global constfactor
    global leftover
    window.fill(WHITE)
    for i in range(len(array)):
        bar = array[i]
        pygame.draw.rect(window, bar.get_color(),(i*constfactor+(leftover//2),0,constfactor//2,(bar.get_val()+1)*constfactor))
    pygame.display.update()

def execute_bubble_sort(window, array):
    global n
    global bubbletimepause
    for i in range(n):
        completed = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        for j in range(n-i-1):
            array[j].highlight()
            array[j+1].highlight()
            draw(window, array)
            time.sleep(bubbletimepause)
            if array[j].get_val()>array[j+1].get_val():
                completed = False
                array[j], array[j+1] = array[j+1], array[j]
            draw(window,array)
            time.sleep(bubbletimepause)
            array[j].reset()
            array[j+1].reset()
        array[n-i-1].make_placed()
        if completed:
            break
    for bar in array:
        bar.make_final()

def drawmerge(window, mergedarray, L, R, pos):
    global n
    global constfactor
    global leftover
    global orgmergarr
    window.fill(WHITE)
    if not len(orgmergarr)==0:
        orgmergarr[pos:pos+len(mergedarray)] = [j for j in mergedarray]
    if not len(L)==0:
        orgmergarr[pos+len(mergedarray):pos+len(mergedarray)+len(L)] = [j for j in L]
    if not len(R)==0:
        orgmergarr[pos + len(mergedarray)+len(L):pos + len(mergedarray) + len(L)+len(R)] = [j for j in R]
    for i in range(len(orgmergarr)):
        bar = orgmergarr[i]
        pygame.draw.rect(window, bar.get_color(),(i*constfactor+(leftover//2),0,constfactor//2,(bar.get_val()+1)*constfactor))
    pygame.display.update()

def execute_merge_sort(window, array):
    if len(array) > 1:

        mid = len(array) // 2

        L = array[:mid]

        R = array[mid:]

        execute_merge_sort(window,L)

        execute_merge_sort(window,R)
        finalsort = False
        if len(array)==len(orgmergarr):
            finalsort = True

        i = j = k = 0
        pos = orgmergarr.index(L[0])
        Lcopy = [j for j in L]
        Rcopy = [j for j in R]
        jointarr = []
        while i < len(L) and j < len(R):
            L[i].highlight()
            R[j].highlight()
            drawmerge(window, jointarr, Lcopy, Rcopy, pos)
            time.sleep(mergetimepause)
            if L[i].get_val() < R[j].get_val():
                array[k] = L[i]
                Lcopy.remove(L[i])
                jointarr.append(L[i])
                drawmerge(window, jointarr, Lcopy, Rcopy, pos)
                time.sleep(mergetimepause)
                L[i].reset()
                if finalsort:
                    array[k].make_placed()
                i += 1
            else:
                array[k] = R[j]
                if finalsort:
                    array[k].make_placed()
                Rcopy.remove(R[j])
                jointarr.append(R[j])
                drawmerge(window, jointarr, Lcopy, Rcopy, pos)
                time.sleep(mergetimepause)
                R[j].reset()
                if finalsort:
                    array[k].make_placed()
                j += 1
            k += 1
        if i<len(L):
            L[i].reset()
        else:
            L[i-1].reset()
        if j<len(R):
            R[j].reset()
        else:
            R[j-1].reset()
        # Checking if any element was left
        while i < len(L):
            array[k] = L[i]
            if finalsort:
                array[k].make_placed()
            i += 1
            k += 1

        while j < len(R):
            array[k] = R[j]
            if finalsort:
                array[k].make_placed()
            j += 1
            k += 1
        if finalsort:
            for bar in array:
                bar.make_final()


def partition(start, end, array, window):
    pivot_index = start
    pivot = array[pivot_index]
    pivot.make_pivot()
    while start < end:
        while start < len(array) and array[start].get_val() <= pivot.get_val():
            array[start].highlight()
            pivot.make_pivot()
            draw(window,array)
            time.sleep(quicktimepause)
            if start>0:
                array[start-1].reset()
            start += 1


        while array[end].get_val() > pivot.get_val():
            array[end].highlight()
            draw(window,array)
            time.sleep(quicktimepause)
            if end<len(array)-1:
                array[end+1].reset()
            end -= 1


        if (start < end):
            array[start], array[end] = array[end], array[start]
            draw(window,array)
            time.sleep(quicktimepause)

        if len(array)>start>=0:
            array[start].reset()
        if 0<=end < len(array):
            array[end].reset()
        draw(window, array)
        time.sleep(quicktimepause)
    array[end], array[pivot_index] = array[pivot_index], array[end]
    draw(window,array)
    time.sleep(quicktimepause)
    pivot.reset()
    return end

def execute_quick_sort(start, end, array, window):
    if (start < end):
        p = partition(start, end, array, window)
        execute_quick_sort(start, p - 1, array, window)
        execute_quick_sort(p + 1, end, array, window)
    if start == 0 and end == len(array)-1:
        for bar in array:
            bar.make_final()
    else:
        for i in range(end+1):
            array[i].make_placed()



def heapify(array, n, i,window):
    largest = i
    left = 2*i + 1
    right = 2*i + 2
    if left<n or right<n:
        array[largest].highlight()
    if left<n:
        array[left].highlight()
        draw(window, array)
        time.sleep(heaptimepause)
        array[left].reset()
    if right<n:
        array[right].highlight()
        draw(window, array)
        time.sleep(heaptimepause)
        array[right].reset()
    array[largest].reset()
    if left < n and array[largest].get_val() < array[left].get_val():
        largest = left

    if right < n and array[largest].get_val() < array[right].get_val():
        largest = right

    if largest != i:
        array[i], array[largest] = array[largest], array[i]  # swap
        draw(window, array)
        time.sleep(heaptimepause)
        heapify(array, n, largest, window)


def execute_heap_sort(window, array):
    n = len(array)

    #for loop iterates through each node with children, starting from the lowest node with children, heapifying each tree with that node as a root along the way
    for i in range(n//2 -1, -1, -1):
        heapify(array, n, i, window)

    for i in range(n - 1, 0, -1):
        array[i], array[0] = array[0], array[i]
        array[i].make_placed()
        draw(window, array)
        time.sleep(heaptimepause)
        heapify(array, i, 0, window)
    for bar in array:
        bar.make_final()
    draw(window, array)
    time.sleep(heaptimepause)

#main
array = make_array()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_j:
                orgmergarr = array
                execute_merge_sort(WIN, array)
            if event.key == pygame.K_h:
                execute_bubble_sort(WIN, array)
            if event.key == pygame.K_k:
                orgquickarr = array
                execute_quick_sort(0, len(array)-1,array, WIN)
            if event.key == pygame.K_l:
                execute_heap_sort(WIN, array)
            if event.key == pygame.K_1:
                n = 10
                constfactor = WINDOW_WIDTH // n
                leftover = WINDOW_WIDTH % n
                bubbletimepause = 10 / (n * n)
                array = make_array()
            if event.key == pygame.K_2:
                n = 20
                constfactor = WINDOW_WIDTH // n
                leftover = WINDOW_WIDTH % n
                bubbletimepause = 10 / (n * n)
                array = make_array()
            if event.key == pygame.K_3:
                n = 30
                constfactor = WINDOW_WIDTH // n
                leftover = WINDOW_WIDTH % n
                bubbletimepause = 10 / (n * n)
                array = make_array()
            if event.key == pygame.K_4:
                n = 40
                constfactor = WINDOW_WIDTH // n
                leftover = WINDOW_WIDTH % n
                bubbletimepause = 10 / (n * n)
                array = make_array()
            if event.key == pygame.K_5:
                n = 50
                constfactor = WINDOW_WIDTH // n
                leftover = WINDOW_WIDTH % n
                bubbletimepause = 10 / (n * n)
                array = make_array()
            if event.key == pygame.K_6:
                n = 60
                constfactor = WINDOW_WIDTH // n
                leftover = WINDOW_WIDTH % n
                bubbletimepause = 10 / (n * n)
                array = make_array()
            if event.key == pygame.K_7:
                n = 70
                constfactor = WINDOW_WIDTH // n
                leftover = WINDOW_WIDTH % n
                bubbletimepause = 10 / (n * n)
                array = make_array()
            if event.key == pygame.K_8:
                n = 80
                constfactor = WINDOW_WIDTH // n
                leftover = WINDOW_WIDTH % n
                bubbletimepause = 10 / (n * n)
                array = make_array()
            if event.key == pygame.K_9:
                n = 90
                constfactor = WINDOW_WIDTH // n
                leftover = WINDOW_WIDTH % n
                bubbletimepause = 10 / (n * n)
                array = make_array()
            if event.key == pygame.K_0:
                n = 100
                constfactor = WINDOW_WIDTH // n
                leftover = WINDOW_WIDTH % n
                bubbletimepause = 10 / (n * n)
                array = make_array()
            # if event.key == pygame.K_r:

    draw(WIN,array)
pygame.quit()
