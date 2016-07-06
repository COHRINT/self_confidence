#!/usr/bin/env python
from __future__ import division
import sys
import networkx as nx
from scipy import misc
from scipy import ndimage
import numpy as np
# from networkx.drawing.nx_agraph import graphviz_layout # in order to use graphviz_layout we need a hack due to bug http://stackoverflow.com/questions/35279733/what-could-cause-networkx-pygraphviz-to-work-fine-alone-but-not-together
# import matplotlib.pyplot as plt
import pre_defined_road_nets # file for road net constructors
from get_image_size import get_image_size # for getting the size of an image
from toPBM import writePBM # writing out a binary occupancy grid image
import json # exporting data
import ipdb
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
__modifier__ = "Sierra Williams"


def genImg(net, res):
    roadnet_gen = getattr(pre_defined_road_nets, net)
    g = roadnet_gen(edglen=2)

    # convert to pygraphviz for better graphviz support
    g_pg = nx.nx_agraph.to_agraph(g)
    __loc_garbage = nx.nx_agraph.graphviz_layout(g, prog='neato', args='') # If I take this away then it segfaults when we use g_pg.graph_attr below... I don't want to use this because it doesn't take into account all of the scaling, and resolutions stuff.

    # add graph property for desired dpi
    g_pg.graph_attr['dpi'] = res # `dots per inch` default for a .png is 96
    g_pg.graph_attr['size'] = 10.0 # in inches, given that there is only one value, both should be equal to this.
    g_pg.graph_attr['ratio'] = 'fill' # this means that the dimensions will need to be scaled in some way in order to match the size

    fmt = 'png'
    # fmt = 'svg'
    make_pbm = True
    fname = net

    if fmt is 'svg':
        # output graph file
        g_pg.draw(fname+'.'+fmt, format=fmt, prog='neato')
    elif fmt is 'png':
        # output graph file
        g_pg.layout(prog='neato',args='')
        g_pg.draw(fname+'.'+fmt, format=fmt)

        # get positions for use in Gazebo world creation
        pos = {}
        for i in g_pg.nodes_iter():
            n = g_pg.get_node(i)
            pos[n.get_name()] = [ float(i) for i in n.attr['pos'].split(',')]

        # graphviz bounding box layout
        bb = g_pg.graph_attr['bb']
        bb_num = [float(i) for i in bb.split(',')]
        # find the scaling between layout and png
        image_size = get_image_size(fname+"."+fmt)

        bb_x =  bb_num[2]-bb_num[0]
        bb_y = bb_num[3]-bb_num[1]

        scale = np.divide(image_size, [bb_x, bb_y])
        pixel_pos = {}
        for key in pos:
            pixel_pos[key] = [pos[key][0]*scale[0], pos[key][1]*scale[1]]

        node_atts = {}
        for key in g.nodes():
            node_atts[key] = g.node[key]['feature']

        with open(fname+".json","w") as outfile:
           json.dump({'pixel_positions':pixel_pos,'original_positions':pos,'feature':node_atts}, outfile, indent=4)

    if make_pbm and fmt is 'png':
        # load image to convert to .pbm
        g_img = misc.imread(fname+'.'+fmt)

        # sum layers, divide by 4 becasue we summed 4 =ers
        g_img_flat = np.sum(g_img, axis=2)/3

        blk = g_img_flat <= 254
        wht = g_img_flat > 254

        g_img_flat[blk] = 0
        g_img_flat[wht] = 1

        writePBM(g_img_flat, fname)

def main():
    net = sys.argv[1]
    res = sys.argv[2]
    genImg(net, res)

if __name__ == '__main__':
    main()
