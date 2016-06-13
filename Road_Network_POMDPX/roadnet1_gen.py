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
g_pg.graph_attr['dpi'] = 100
fmt = 'png'
make_pbm = True

if fmt is 'svg':
    # output graph file
    fname = 'roadnet1'
    g_pg.draw(fname+'.'+fmt,format=fmt,prog='neato')
elif fmt is 'png':
    # output graph file
    fname = 'roadnet1'
    g_pg.draw(fname+'.'+fmt,format=fmt,prog='neato')

if make_pbm and fmt is 'png':
    # load image to convert to .pbm
    g_img = misc.imread(fname+'.'+fmt)

    g_img_flat = np.sum(g_img,axis=2)/4
    misc.imshow(g_img_flat)
    print(g_img_flat)

    # make square for gazebo code
    min_sze = min(g_img_flat.shape)
    max_sze = max(g_img_flat.shape)
    min_ax = g_img_flat.shape.index(min_sze)
    g_img_sq = np.append(g_img_flat,np.zeros((max_sze-min_sze,max_sze),dtype=int),axis=min_ax)
    # squash colors out
    blk = g_img_sq<=254
    wht = g_img_sq>254
    g_img_sq[blk] = 0
    g_img_sq[wht] = 1

    writePBM(g_img_sq,fname)
