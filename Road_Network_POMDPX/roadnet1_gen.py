import networkx as nx
from scipy import misc
import numpy as np
# from networkx.drawing.nx_agraph import graphviz_layout # in order to use graphviz_layout we need a hack due to bug http://stackoverflow.com/questions/35279733/what-could-cause-networkx-pygraphviz-to-work-fine-alone-but-not-together
import pygraphviz as PG
import matplotlib.pyplot as plt
from pre_defined_road_nets import roadnet1
from toPBM import writePBM

g = roadnet1(edglen=2)

# convert to pygraphviz for better graphviz support
g_pg = nx.nx_agraph.to_agraph(g)

# add graph property for desired dpi
g_pg.graph_attr['dpi'] = 300

# output graph file
fname = 'roadnet1'
g_pg.draw(fname+'.png',format='png',prog='neato')

# load image to convert to .pbm
g_img = misc.imread(fname+'.png')[:,:,0]
print(g_img.shape)

# make square for gazebo code
min_sze = min(g_img.shape)
max_sze = max(g_img.shape)
min_ax = g_img.shape.index(min_sze)
g_img_sq = np.append(g_img,np.zeros((max_sze-min_sze,max_sze),dtype=int),axis=min_ax)

# squash colors out
blk = g_img==0
wht = g_img>0
g_img[blk] = 1
g_img[wht] = 0

writePBM(g_img,fname)
