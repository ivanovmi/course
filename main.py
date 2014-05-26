__author__ = 'michael & evgeny'

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

graph = {}
deleted=[]


matplotlib.rcParams["toolbar"]="None"
cf = pylab.gcf()
cf.set_facecolor('w')
ax = pylab.gca()


def create_random():
    nodes = int(np.random.randint(2, 10))
    edges = int(np.random.randint(1, nodes))


    for i in xrange(1, nodes+1):
        graph[i] = []

    if edges == 1:
        edges += 1
    for i in xrange(edges):
        graph[int(np.random.randint(1, edges))].append(int(np.random.randint(0, edges)))

    draw(graph)


def create_from_file():
    f = open("graph.txt", 'r')
    nodes = int(f.readline())
    edges = int(f.readline())

    for i in xrange(1, nodes+1):
        graph[i] = []

    for i in xrange(1, int(edges/2)-1):
        arr = (f.read())
        edge = arr.split("\n")
        for i in xrange(len(edge)):
            a = edge[i].split(" ")
            graph[int(a[0])].append(int(a[1]))

    draw(graph)


def create_graph():
    nodes = int(raw_input("Enter the count of nodes: "))
    edges = int(raw_input("Enter the count of edges: "))

    for i in xrange(1, nodes+1):
        graph[i] = []

    for i in xrange(edges):
        edge = raw_input("Enter first and second (space): ")
        a = edge.split(" ")
        if int(a[1]) == 0:
            graph[int(a[0])].append(int(a[0]))
        else:
            graph[int(a[0])].append(int(a[1]))

    draw(graph)


def draw(graph):
    xy = []
    for i in xrange(len(graph)):
        x = np.random.random()
        y = np.random.random()
        xy.append([x, y])
        node_collection = ax.scatter(x, y, s=300, c='r', marker='s', zorder=2)

    pos = []

    for i in graph:
            for j in xrange(len(graph[i])):
                    pos.append((i, graph[i][j]))

    for i in xrange(len(pos)):
        if [pos[i][0],pos[i][1]] not in deleted:
            plt.plot([xy[pos[i][0]-1][0], xy[pos[i][1]-1][0]], [xy[pos[i][0]-1][1], xy[pos[i][1]-1][1]], 'k-', zorder=1)

    for i in graph:
            t = ax.text(xy[i-1][0]-0.007, xy[i-1][1]-0.01, i, zorder=3)

    plt.axis("off")
    #cf.set_history_buttons()
    cf.canvas.set_window_title("Graph vizualization")
    plt.ion()
    plt.show()


def find_shortest_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not graph.has_key(start):
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = find_shortest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest



choice = int(raw_input("How you would like to create graph?\
                  1 - manually\
                  2-random\
                  3-from file: "))
if choice == 1:
    create_graph()
elif choice == 2:
    create_random()
elif choice == 3:
    create_from_file()


flag=0
while(1):

    choice = raw_input("Would you like to continue(Y/N)? ")


    if choice in ["Y", "y", "YES", "yes", "Yes"]:
        plt.cla()
        plt.clf()
        plt.close()
        flag=1
        break
    elif choice in ["N", "n", "no", "NO", "No"]:
        plt.cla()
        plt.clf()
        plt.close()
        break

if flag:
    count=int(raw_input("How much edges you would like to remove?: "))
    for i in xrange(count):
        rem=raw_input("Enter the edges to remove (space): ")
        a=rem.split(" ")
        deleted.append([int(a[0]),int(a[1])])
    matplotlib.rcParams["toolbar"]="None"
    cf = pylab.gcf()
    cf.set_facecolor('w')
    ax = pylab.gca()
    draw(graph)