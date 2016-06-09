import networkx as nx
# from networkx.drawing.nx_agraph import graphviz_layout # in order to use graphviz_layout we need a hack due to bug http://stackoverflow.com/questions/35279733/what-could-cause-networkx-pygraphviz-to-work-fine-alone-but-not-together
import pygraphviz as PG
import matplotlib.pyplot as plt
from pre_defined_road_nets import roadnet1

g = roadnet1(edglen=2)

# convert to pygraphviz for better graphviz support
g_pg = nx.nx_agraph.to_agraph(g)

# add graph property for desired dpi
g_pg.graph_attr['dpi'] = 300

# output graph file
g_pg.draw('name.png',format='png',prog='neato')
