import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)
G.add_node(6)
G.add_edge(1,2)

nx.draw(G)
#nx.draw_random(G)
#nx.draw_circular(G)

plt.show()