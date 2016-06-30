#!/usr/bin/env python
from __future__ import division
import sys
import networkx as nx
from scipy import misc
from scipy import ndimage
import numpy as np
# from networkx.drawing.nx_agraph import graphviz_layout # in order to use graphviz_layout we need a hack due to bug http://stackoverflow.com/questions/35279733/what-could-cause-networkx-pygraphviz-to-work-fine-alone-but-not-together
# import matplotlib.pyplot as plt
import pre_defined_road_nets
from toPBM import writePBM
import json

"""Outputs occupancy grid and a .png for a road network

"""
__author__ = "Brett Israelsen"
__copyright__ = "Copyright 2016, Cohrint"
__credits__ = ["Brett Israelsen"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Brett Israelsen"
__email__ = "brett.israelsen@colorado.edu"
__status__ = "Development"

__modifier__= "Sierra Williams"


def genImg(net, res):
    roadnet_gen = getattr(pre_defined_road_nets, net)
    g = roadnet_gen(edglen=2)

    #Use layout to grab nodes pos
    position = nx.nx_agraph.graphviz_layout(g)

    #To convert to gazebo measurements will need to multiply non key values by constant
    # This line will be changed


    for key, value in position.iteritems():
        position[key] = (value[0]*.206, value[1]*.206) 

    # Writes node positions to txt file with json
    with open('node_position.json','w') as outfile:
        json.dump(position, outfile, sort_keys=True, indent=4)

    # convert to pygraphviz for better graphviz support
    g_pg = nx.nx_agraph.to_agraph(g)

    # add graph property for desired dpi
    g_pg.graph_attr['dpi'] = res
    fmt = 'png'

    #fmt = 'svg'
    make_pbm = True
    fname = net

    if fmt is 'svg':
        # output graph file
        g_pg.draw(fname+'.'+fmt, format=fmt, prog='neato')
    elif fmt is 'png':
        # output graph file
        g_pg.draw(fname+'.'+fmt, format=fmt, prog='neato')

    if make_pbm and fmt is 'png':
        # load image to convert to .pbm
        g_img = misc.imread(fname+'.'+fmt)

        # sum layers, divide by 4 becasue we summed 4 =ers
        g_img_flat = np.sum(g_img, axis=2)/4

        # make square for gazebo code
        min_sze = min(g_img_flat.shape)
        max_sze = max(g_img_flat.shape)
        min_ax = g_img_flat.shape.index(min_sze)
        if min_ax == 0:
            append_ary = np.ones((max_sze-min_sze, max_sze), dtype=int)
        else:
            append_ary = np.ones((max_sze, max_sze-min_sze), dtype=int)

        g_img_sq = np.append(g_img_flat, 255*append_ary, axis=min_ax)
        #g_sobel = ndimage.sobel(g_img_sq)
        # print(g_sobel)
        # misc.imshow(g_sobel)
        # squash colors out
        blk = g_img_sq <= 254
        wht = g_img_sq > 254
        # blk = np.abs(g_sobel) <= 0
        # wht = np.abs(g_sobel) > 0
        g_img_sq[blk] = 0
        g_img_sq[wht] = 1

        writePBM(g_img_sq, fname)
        g_img_sq_rot90 = np.rot90(g_img_sq,k=2)
        writePBM(g_img_sq_rot90, fname+'_rot180')


def main():
    net = sys.argv[1]
    res = sys.argv[2]
    genImg(net, res)

if __name__ == '__main__':
    main()
