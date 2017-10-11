#!/usr/bin/env python
from __future__ import division

"""Provides functunality for a Road Network

"""
__author__ = "Matthew Aitken"
__copyright__ = "Copyright 2016, Cohrint"
__credits__ = ["Matthew Aitken"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Matthew Aitken"
__email__ = "matthew@raitken.net"
__status__ = "Development"


import numpy as np
import logging
import copy
import networkx as nx
from lxml import etree
from pre_defined_road_nets import test_roadnetwork

logger = logging.getLogger(__name__)


def pomdpx_write(road_network,
                 ugv_transition_function,
                 pursuer_transition_function,
                 description_text=None,
                 filename='road_network.pomdpx',
                 problem_id='pursuitEvasion',
                 discount_factor=0.95):
    """Generate a pomdpx file for the road network problem
        road_network: nx graphical model with intersections and weights
        transition_function: function takes states and returns a probability
        description: optional text to describe the problem

        Future Work:
        Add sorted node list
    """

    rn = road_network
    actions = ['f', 'b', 'l', 'r', 's']
    ugv_iter = list(rn.nodes())
    ugv_iter.append('T')

    xml = "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>"
    ns = "http://www.w3.org/2001/XMLSchema-instance"
    location_attribute = '{%s}noNameSpaceSchemaLocation' % ns
    pomdpx = etree.Element("pomdpx", version="0.1", id=problem_id)
    pomdpx.set(location_attribute, "pomdpx.xsd")

    # Description
    if description_text is not None:
        description = etree.SubElement(pomdpx, "Description")
        description.text = description_text

    # Discount
    discount = etree.SubElement(pomdpx, "Discount")
    discount.text = str(discount_factor)

    # Variable
    variable = etree.SubElement(pomdpx, "Variable")

    #   State Variables
    ugv_attr = {"vnamePrev": "ugv_0", "vnameCurr": "ugv_1", "fullyObs": "true"}
    ugv_state = etree.SubElement(variable, "StateVar", ugv_attr)
    ugv_values = etree.SubElement(ugv_state, "ValueEnum")
    ugv_values.text = ' '.join(sorted(rn.nodes(), key=neighbor_key)) + ' T'

    pusuer_attr = {"vnamePrev": "pursuer_0", "vnameCurr": "pursuer_1"}
    pursuer_state = etree.SubElement(variable, "StateVar", pusuer_attr)
    pursuer_values = etree.SubElement(pursuer_state, "ValueEnum")
    pursuer_values.text = ' '.join(sorted(rn.nodes(), key=neighbor_key))

    #   Observation Variables
    val_enum = {}
    obs_var = {}
    obs_vnames = []
    for node_name in (n for n in rn if 'feature' in rn.node[n]
                      and rn.node[n]['feature'] == 'sensor'):
        vname = 'Node_{}_Sensor'.format(node_name)
        obs_vnames.append(vname)
        obs_var[node_name] = etree.SubElement(variable, "ObsVar", vname=vname)
        val_enum[node_name] = etree.SubElement(obs_var[node_name], "ValueEnum")
        val_enum[node_name].text = 'True False'
    # If not caught, pursuer wasn't there
    obs_vnames.append('UGV_Sensor')
    obs_var_ugv = etree.SubElement(variable, "ObsVar", vname='UGV_Sensor')
    val_enum_ugv = etree.SubElement(obs_var_ugv, "ValueEnum")
    val_enum_ugv.text = 'True False'

    #   Action Variables
    act_var = etree.SubElement(variable, "ActionVar", vname="action_ugv")
    act_val = etree.SubElement(act_var, "ValueEnum")
    act_val.text = ' '.join(actions)

    #   Reward Variables
    reward_var = etree.SubElement(variable, "RewardVar", vname="reward_ugv")

    # Initial State Belief
    initial_state_belief = etree.SubElement(pomdpx, "InitialStateBelief")
    ugv_icp = etree.SubElement(initial_state_belief, "CondProb")
    ugv_icp_var = etree.SubElement(ugv_icp, "Var")
    ugv_icp_var.text = 'ugv_0'
    ugv_icp_parent = etree.SubElement(ugv_icp, "Parent")
    ugv_icp_parent.text = 'null'
    ugv_icp_param = etree.SubElement(ugv_icp, "Parameter", type='TBL')
    ugv_icp_entry = etree.SubElement(ugv_icp_param, "Entry")
    ugv_icp_instance = etree.SubElement(ugv_icp_entry, "Instance")
    ugv_icp_instance.text = '4'
    ugv_icp_probtable = etree.SubElement(ugv_icp_entry, "ProbTable")
    ugv_icp_probtable.text = '1.0'

    pursuer_icp = etree.SubElement(initial_state_belief, "CondProb")
    pursuer_icp_var = etree.SubElement(pursuer_icp, "Var")
    pursuer_icp_var.text = 'pursuer_0'
    pursuer_icp_parent = etree.SubElement(pursuer_icp, "Parent")
    pursuer_icp_parent.text = 'null'
    pursuer_icp_param = etree.SubElement(pursuer_icp, "Parameter", type='TBL')
    pursuer_icp_entry = etree.SubElement(pursuer_icp_param, "Entry")
    pursuer_icp_instance = etree.SubElement(pursuer_icp_entry, "Instance")
    pursuer_icp_instance.text = '-'
    pursuer_icp_probtable = etree.SubElement(pursuer_icp_entry, "ProbTable")
    pursuer_icp_probtable.text = 'uniform'

    # State Transition Function
    stfunc = etree.SubElement(pomdpx, "StateTransitionFunction")
    ugv_cond_prob = etree.SubElement(stfunc, "CondProb")
    ugv_cond_prob_var = etree.SubElement(ugv_cond_prob, "Var")
    ugv_cond_prob_var.text = 'ugv_1'
    ugv_cond_prob_parent = etree.SubElement(ugv_cond_prob, "Parent")
    ugv_cond_prob_parent.text = 'action_ugv ugv_0 pursuer_0'
    ugv_cond_prob_param = etree.SubElement(ugv_cond_prob, "Parameter", type="TBL")
    ugv_entry_list = ugv_transition_function(actions, rn)
    for entry in ugv_entry_list:
        ugv_cond_prob_param.append(entry)

    pursuer_cond_prob = etree.SubElement(stfunc, "CondProb")
    pursuer_cond_prob_var = etree.SubElement(pursuer_cond_prob, "Var")
    pursuer_cond_prob_var.text = 'pursuer_1'
    pursuer_cond_prob_parent = etree.SubElement(pursuer_cond_prob, "Parent")
    pursuer_cond_prob_parent.text = 'pursuer_0 ugv_0'
    pursuer_cond_prob_param = etree.SubElement(pursuer_cond_prob, "Parameter", type="TBL")
    pursuer_entry_list = pursuer_transition_function(rn)
    for entry in pursuer_entry_list:
        pursuer_cond_prob_param.append(entry)

    # Observation Function
    true_positive = 0.85
    false_positive = 0.15
    obsfunc = etree.SubElement(pomdpx, "ObsFunction")
    for sensor in obs_vnames:
        obs_cond_prob = etree.SubElement(obsfunc, "CondProb")
        obs_cond_prob_var = etree.SubElement(obs_cond_prob, "Var")
        obs_cond_prob_var.text = sensor
        obs_cond_prob_parent = etree.SubElement(obs_cond_prob, "Parent")
        if sensor == 'UGV_Sensor':
            obs_cond_prob_parent.text = 'pursuer_1 ugv_1'
        else:
            obs_cond_prob_parent.text = 'pursuer_1'
        obs_cond_prob_param = etree.SubElement(obs_cond_prob, "Parameter", type="TBL")

        for state in rn.nodes():
            if sensor == 'UGV_Sensor':
                for ugv_state in ugv_iter:
                    entry = etree.SubElement(obs_cond_prob_param, "Entry")
                    instance = etree.SubElement(entry, "Instance")
                    instance.text = state + ' ' + ugv_state + ' -'
                    probtable = etree.SubElement(entry, "ProbTable")
                    if ugv_state == state:
                        text = "{:.2f} {:.2f}".format(1.0, 0.0)
                    else:
                        text = "{:.2f} {:.2f}".format(0.0, 1.0)
                    probtable.text = text
            else:
                entry = etree.SubElement(obs_cond_prob_param, "Entry")
                instance = etree.SubElement(entry, "Instance")
                instance.text = state + ' -'
                probtable = etree.SubElement(entry, "ProbTable")
                node = sensor.split('_')[1]
                if state == node:
                    text = "{:.2f} {:.2f}".format(true_positive, (1 - true_positive))
                else:
                    text = text = "{:.2f} {:.2f}".format(false_positive, (1 - false_positive))
                probtable.text = text

    # Reward Function
    movement_reward = -1
    caught_reward = -1000
    escape_reward = 1000
    terminal_reward = 0

    rewfunc = etree.SubElement(pomdpx, "RewardFunction")
    rfunc = etree.SubElement(rewfunc, "Func")
    rfunc_var = etree.SubElement(rfunc, "Var")
    rfunc_var.text = 'reward_ugv'
    rfunc_parent = etree.SubElement(rfunc, "Parent")
    rfunc_parent.text = 'ugv_0 action_ugv pursuer_0'
    rfunc_param = etree.SubElement(rfunc, "Parameter")
    for ugv_i in ugv_iter:
        if ugv_i == 'T':
            entry = etree.SubElement(rfunc_param, "Entry")
            instance = etree.SubElement(entry, "Instance")
            instance.text = 'T * *'
            valuetable = etree.SubElement(entry, "ValueTable")
            valuetable.text = '{}'.format(terminal_reward)
        else:
            for action in actions:
                for pursuer_i in rn.nodes():
                    ugv_e = ugv_update_state(ugv_i, action, rn)
                    if ugv_i == pursuer_i or ugv_e == pursuer_i:
                        entry = etree.SubElement(rfunc_param, "Entry")
                        instance = etree.SubElement(entry, "Instance")
                        instance.text = ugv_i + ' ' + action + ' ' + pursuer_i
                        valuetable = etree.SubElement(entry, "ValueTable")
                        valuetable.text = '{}'.format(caught_reward)
                    elif 'feature' in rn.node[ugv_e] and rn.node[ugv_e]['feature'] == 'exit':
                        entry = etree.SubElement(rfunc_param, "Entry")
                        instance = etree.SubElement(entry, "Instance")
                        instance.text = ugv_i + ' ' + action + ' ' + pursuer_i
                        valuetable = etree.SubElement(entry, "ValueTable")
                        valuetable.text = '{}'.format(escape_reward)
                    else:
                        entry = etree.SubElement(rfunc_param, "Entry")
                        instance = etree.SubElement(entry, "Instance")
                        instance.text = ugv_i + ' ' + action + ' ' + pursuer_i
                        valuetable = etree.SubElement(entry, "ValueTable")
                        valuetable.text = '{}'.format(movement_reward)

    f = open('test.pomdpx', 'w')
    f.write(xml + '\n')
    f.write(etree.tostring(pomdpx, pretty_print=True))


def expand_road_network(road_network, discritization):
    """Discritize a simple road_network
        Takes a simple road network with nodes at features and intersections
        and edges with weights between nodes and add nodes along the edges
    """
    rn_old = road_network
    df = discritization  # nodes per unit weight

    # Find shortest paths and path lengths
    paths = nx.all_pairs_dijkstra_path(rn_old)
    path_lengths = nx.all_pairs_dijkstra_path_length(rn_old)

    # Create new graph
    rn = nx.Graph()
    rn.add_nodes_from(rn_old.nodes(data=True))

    for old_edge in rn_old.edges(data=True):
        beg = old_edge[0]
        end = old_edge[1]
        if int(beg) > int(end):
            beg, end = end, beg

        num_nodes = int(round(old_edge[2]['weight'] * df) - 1)

        old_node_name = beg
        for node in range(num_nodes):
            new_node_name = '{}.{}.{}'.format(beg, end, node)
            if node == num_nodes - 1:
                rn.add_edge(new_node_name, end)
            rn.add_edge(old_node_name, new_node_name)

            old_node_name = new_node_name

    return rn, paths, path_lengths


def neighbor_key(name):
            return tuple(int(part) for part in name.split('.'))


def ugv_update_state(state, action, rn):
    if action == 's':
        return state

    if '.' in state:  # True- on a line between to main nodes, actions = 'f','b','s'
        if action not in ['f', 'b']:
            return state

        neighbors = rn[state].keys()
        if len(neighbors) > 2:
            # logger.warn
            raise ValueError('{} has too many neighbors'.format(state))

        state_list = state.split('.')
        if state_list[-1] is not '0':
            backward = '.'.join(state_list[0:-1]) + '.' + str(int(state_list[-1]) - 1)
        else:
            backward = state_list[0]

        for state in neighbors:
            if state is not backward:
                forward = state

        if action == 'f':
            return forward
        elif action == 'b':
            return backward

    else:   # False - main node, actions full fblrs
        neighbors = rn[state].keys()
        if len(neighbors) > 4:
            # logger.warn
            raise ValueError('{} has too many neighbors'.format(state) +
                             '\nCode can not handle more than a 4 way intersection')

        neighbors = sorted(neighbors, key=neighbor_key)
        # clockwise = ascending order

        try:
            idx = 'fblrs'.index(action)
            return neighbors[idx]
        except:
            return state


def ugv_transition_function(actions, road_network):
    """Take the roadnetwork and return nonzero transition probabilities.
        Independant of ugv actions.
       Return a list of <Entry/> elements
    """
    rn = road_network
    entry_list = []
    for state in rn.nodes():
        for action in actions:
            new_state = ugv_update_state(state, action, rn)
            for pstate in rn.nodes():
                if pstate == new_state or pstate == state:
                    new_entry = etree.Element("Entry")
                    instance = etree.SubElement(new_entry, "Instance")
                    instance.text = action + ' ' + state + ' ' + pstate + ' T'
                    probtable = etree.SubElement(new_entry, "ProbTable")
                    probtable.text = '1.0'
                elif 'feature' in rn.node[new_state] and \
                     rn.node[new_state]['feature'] == 'exit':
                    new_entry = etree.Element("Entry")
                    instance = etree.SubElement(new_entry, "Instance")
                    instance.text = action + ' ' + state + ' ' + pstate + ' T'
                    probtable = etree.SubElement(new_entry, "ProbTable")
                    probtable.text = '1.0'
                else:
                    new_entry = etree.Element("Entry")
                    instance = etree.SubElement(new_entry, "Instance")
                    instance.text = action + ' ' + state + ' ' + pstate + ' ' + new_state
                    probtable = etree.SubElement(new_entry, "ProbTable")
                    probtable.text = '1.0'

                entry_list.append(new_entry)

    # Handle the terminal state
    new_entry = etree.Element("Entry")
    instance = etree.SubElement(new_entry, "Instance")
    instance.text = '*' + ' ' + 'T' + ' ' + '*' + ' ' + 'T'
    probtable = etree.SubElement(new_entry, "ProbTable")
    probtable.text = '1.0'
    entry_list.append(new_entry)

    return entry_list


def pursuer_random_walk(state, rn):
    neighbors = rn[state].keys()
    neighbors.append(state)

    n = len(neighbors)
    prob = "%.3f" % (1 / n)
    s_prob = "%.3f" % (float(prob) + 1 - n * float(prob))
    entry_list = []
    for new_state in neighbors:
        new_entry = etree.Element("Entry")
        instance = etree.SubElement(new_entry, "Instance")
        instance.text = state + ' * ' + new_state
        probtable = etree.SubElement(new_entry, "ProbTable")
        if state == new_state:
            probtable.text = s_prob
        else:
            probtable.text = prob
        entry_list.append(new_entry)

    return entry_list


def pursuer_transition_function(road_network):
    """Take the roadnetwork and return nonzero transition probabilities.
        Independant of ugv actions.
       Return a list of <Entry/> elements
    """
    rn = road_network
    entry_list = []
    for state in rn.nodes():
        state_entry_list = pursuer_random_walk(state, rn)
        entry_list.extend(state_entry_list)

    return entry_list


def main():
    rn = test_roadnetwork()
    rn, paths, path_lengths = expand_road_network(rn, 1)
    description_text = 'The pursuer is attempting to capture the UGV ' + 'before it can reach the exit.'
    pomdpx_write(rn, ugv_transition_function, pursuer_transition_function,
                 description_text=description_text)

if __name__ == '__main__':
    main()
