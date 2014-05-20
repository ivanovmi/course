__author__ = 'michael'

import networkx as nx
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
from random import random
from functools import partial

G=nx.Graph()
'''
class InitScreen(GridLayout):
    def __init__(self, **kwargs):
        super(InitScreen,self).__init__(**kwargs)
        self.cols = 2
        self.add_widget(Label(text = "Enter the count of nodes:"))
        self.nodes = TextInput(multiline = False)
        self.add_widget(self.nodes)
        self.add_widget(Label(text = "Enter the count of edges:"))
        self.edges = TextInput(multiline = False)
        self.add_widget(self.edges)

        #self.do = Button(on_press=partial(NodeCreate(self.nodes),Widget(),'add'))
        #self.add_widget(self.do)
        a=self.edges
        if a == 1:
            print "Fuck the king!"'''
nodes=int(raw_input("Enter the count of nodes: "))
edges=int(raw_input("Enter the count of edges: "))

#def NodeCreate(nodes):
for i in xrange(1,nodes+1):
    G.add_node(i)

#def EdgeCreate(edges):
for i in xrange(edges):
    edge=raw_input("Enter first and second (space): ")
    a=edge.split(" ")
    G.add_edge(int(a[0]),int(a[1]))

'''
class Application(App):
    def build(self):
        return InitScreen()

if __name__ == '__main__':
    Application().run()

if nx.is_connected(G)==True:
    print "Yeah, it's connected!"
else:
    print "Not connected"'''
c=G.edges()
print c
print type(c)
nx.draw(G)
#nx.draw_random(G)
#nx.draw_circular(G)

print G.edges()

plt.show()