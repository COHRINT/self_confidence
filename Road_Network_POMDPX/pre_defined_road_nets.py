#!/usr/bin/env python
from __future__ import division

"""Functions to produce various road networks

"""
__author__ = "Brett Israelsen"
__copyright__ = "Copyright 2016, Cohrint"
__credits__ = ["Brett Israelsen"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Brett Israelsen"
__email__ = "brett.israelsen@colorado.edu"
__status__ = "Development"

import networkx as nx

def test_roadnetwork():
    rn = nx.Graph(name='roadnetwork')
    rn.add_edge(1, 2, weight=5)
    rn.add_edge(1, 10, weight=5)
    rn.add_edge(2, 3, weight=5)
    rn.add_edge(2, 9, weight=5)
    rn.add_edge(3, 4, weight=5)
    rn.add_edge(4, 5, weight=5)
    rn.add_edge(4, 7, weight=5)
    rn.add_edge(5, 6, weight=5)
    rn.add_edge(6, 7, weight=5)
    rn.add_edge(6, 8, weight=5)
    rn.add_edge(7, 8, weight=5)
    rn.add_edge(7, 11, weight=5)
    rn.add_edge(8, 12, weight=5)
    rn.add_edge(8, 13, weight=5)
    rn.add_edge(9, 10, weight=5)
    rn.add_edge(9, 11, weight=5)
    rn.add_edge(10, 12, weight=5)
    rn.add_edge(11, 12, weight=5)
    rn.add_edge(12, 13, weight=5)
    rn.node[13]['feature'] = 'exit'
    rn.node[1]['feature'] = 'sensor'
    rn.node[3]['feature'] = 'sensor'
    rn.node[5]['feature'] = 'sensor'
    rn.node[6]['feature'] = 'sensor'
    rn.node[7]['feature'] = 'sensor'
    rn.node[8]['feature'] = 'sensor'
    rn.node[9]['feature'] = 'sensor'
    rn.node[10]['feature'] = 'sensor'
    rn.node[12]['feature'] = 'sensor'

    def mapping(x):
        return str(x)

    rn = nx.relabel_nodes(rn, mapping)

    return rn

def roadnet1(edglen=1):
    rn = nx.Graph(name='roadnetwork')
    rn.add_edge(1, 2)
    rn.add_edge(1, 10)
    rn.add_edge(2, 3)
    rn.add_edge(2, 9)
    rn.add_edge(3, 4)
    rn.add_edge(4, 5)
    rn.add_edge(4, 7)
    rn.add_edge(5, 6)
    rn.add_edge(6, 7)
    rn.add_edge(6, 8)
    rn.add_edge(7, 8)
    rn.add_edge(7, 11)
    rn.add_edge(8, 12)
    rn.add_edge(8, 13)
    rn.add_edge(9, 10)
    rn.add_edge(9, 11)
    rn.add_edge(10, 12)
    rn.add_edge(11, 12)
    rn.add_edge(12, 13)
    rn.node[13]['feature'] = 'exit'
    rn.node[1]['feature'] = 'sensor'
    rn.node[3]['feature'] = 'sensor'
    rn.node[5]['feature'] = 'sensor'
    rn.node[6]['feature'] = 'sensor'
    rn.node[7]['feature'] = 'sensor'
    rn.node[8]['feature'] = 'sensor'
    rn.node[9]['feature'] = 'sensor'
    rn.node[10]['feature'] = 'sensor'
    rn.node[12]['feature'] = 'sensor'

    # Add some edge attributes
    for e in rn.edge.keys():
        for edg in rn.edge[e]:
            # Edge lengths
            rn.edge[e][edg]['len'] = str(edglen)

            # Edge appearance
            rn.edge[e][edg]['penwidth'] = '5'

    # Add some node attributes
    for n in rn.nodes_iter():
        # Node shape
        rn.node[n]['shape'] = 'circle'
        rn.node[n]['fixedsize'] = 'true'
        rn.node[n]['width'] = '0.3'

        # Node appearance
        rn.node[n]['style'] = 'filled'
        rn.node[n]['label'] = ''
        if rn.node[n].get('feature') is 'sensor':
            rn.node[n]['color'] = 'red'
            rn.node[n]['fillcolor'] = 'red'
        elif rn.node[n].get('feature') is 'exit':
            rn.node[n]['color'] = 'green'
            rn.node[n]['fillcolor'] = 'green'
        else:
            rn.node[n]['color'] = 'black'
            rn.node[n]['fillcolor'] = 'black'

    def mapping(x):
        return str(x)

    rn = nx.relabel_nodes(rn, mapping)

    return rn
