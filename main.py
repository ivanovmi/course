__author__ = 'michael & evgeny'

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.cbook as cb
from matplotlib.collections import LineCollection
import numpy
import random
import networkx as nx


'''
nodes=int(raw_input("Enter the count of nodes: "))
edges=int(raw_input("Enter the count of edges: "))

graph={}

for i in xrange(1,nodes+1):
        graph[i]=[]

for i in xrange(edges):
        edge=raw_input("Enter first and second (space): ")
        a=edge.split(" ")
        graph[int(a[0])].append(int(a[1]))

print graph.items()
'''

graph = {1: [2, 3],
         2: [3, 4],
         3: [4],
         4: [3],
         5: [6],
         6: [3]}

cf=pylab.gcf()
cf.set_facecolor('w')
ax=None
nodelist=None
edgelist=None
x=random.randint(0,1)
y=random.randint(0,1)
if ax is None:
        ax=pylab.gca()

nodelist=graph.keys()


plt.plot([0.1, 0.5],[0.1, 0.7], 'k-', zorder=1)
plt.plot([0.1,0.2],[0.1,0.5],'k-',zorder=1)
t=ax.text(0.094,0.091, "X", zorder=3)
node_collection=ax.scatter(0.1,0.1,s=300,c='r',marker='o', zorder=2)
node_collection=ax.scatter(0.2,0.5,s=300,c='r',marker='o', zorder=2)
node_collection=ax.scatter(0.3,0.1,s=300,c='r',marker='o')
node_collection=ax.scatter(0.4,0.2,s=300,c='r',marker='o')
node_collection=ax.scatter(0.5,0.7,s=300,c='r',marker='o')
node_collection=ax.scatter(0.6,0.4,s=300,c='r',marker='o')

edgelist=graph.values()

edge_pos=[]
pos=[]

for i in graph:
        for j in xrange(len(graph[i])):
                pos.append((i, graph[i][j]))

edge_pos=numpy.asarray([(pos[0][0],pos[0][1])])

                

def find_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not graph.has_key(start):
            return None
        for node in graph[start]:
            if node not in path:
                newpath = find_path(graph, node, end, path)
                if newpath: return newpath
        return None

#print find_path(graph,'A','C')

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

#print find_shortest_path(graph,1,4)


plt.show()
