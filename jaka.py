import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()

nodes=int(raw_input("Enter the count of nodes: "))
edges=int(raw_input("Enter the count of edges: "))

for i in xrange(1,nodes+1):
    G.add_node(i)

for i in xrange(edges):
    edge=raw_input("Enter first and second (space): ")
    a=edge.split(" ")
    G.add_edge(int(a[0]),int(a[1]))




nx.draw(G)
#nx.draw_random(G)
#nx.draw_circular(G)

plt.show()
