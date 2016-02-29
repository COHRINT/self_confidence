#!/usr/bin/env python
from __future__ import division

"""Augmented Markov Decision Process Solver

"""
__author__ = "Matthew Aitken"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Matthew Aitken"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Matthew Aitken"
__email__ = "matthew@raitken.net"
__status__ = "Development"


import numpy as np
import logging
import copy

logging.basicConfig(level=logging.INFO)
# np.set_printoptions(precision=4)


def get_actions(bbar):
        if bbar[0] == 0:
            actions = [0, 1]
        elif bbar[0] in [1, 3]:
            actions = [-1, 0, 1]
        elif bbar[0] == 5:
            actions = [-3, 0, 1]
        elif bbar[0] == 2:
            actions = [-1, 0, 1, 3]
        elif bbar[0] == 't':
            actions = ['t']
        return actions


def f(b, bbars, precision=1):
    # Generates bbar from b
    # Precision must exactly match bbars
    vehicle_state = b[0]
    b = b[1:]
    branch_02 = round(sum(b[0:3]), precision)
    branch_34 = round(sum(b[3:5]), precision)
    branch_56 = round(sum(b[5:]), precision)

    total = branch_02 + branch_34 + branch_56
    if total != 1:
        logging.debug('Total != 1')
        branch_56 = 1 - branch_02 - branch_34
        branch_56 = round(branch_56, precision)

    belief = [branch_02, branch_34, branch_56]
    b = [vehicle_state] + belief

    return bbars.index(b)


def finv(bbar):
    logging.debug('bbar = {}'.format(bbar))
    # Generates b from bbar
    vehicle_state = bbar[0]
    # branch_02 = np.random.random(3)
    # branch_02 *= bbar[1] / branch_02.sum()

    # branch_34 = np.random.random(2)
    # branch_34 *= bbar[2] / branch_34.sum()

    # branch_56 = np.random.random(2)
    # branch_56 *= bbar[3] / branch_56.sum()

    branch_02 = bbar[1] / 3 * np.array([1, 1, 1])
    branch_34 = bbar[2] / 2 * np.array([1, 1])
    branch_56 = bbar[3] / 2 * np.array([1, 1])

    belief = branch_02.tolist() + branch_34.tolist() + branch_56.tolist()
    b = [vehicle_state] + belief
    return b


def sample_state(belief):
    state = np.random.choice(7, 1, p=belief)
    return state


def sample_movement(old_state, T):
    # T is the transition matrix
    p = np.reshape(T[:, old_state], len(T[:, old_state]))
    p = p / sum(p)
    new_state = np.random.choice(7, 1, p=p)
    return new_state


def sample_measurement(state, truepos, falsepos):
    # truepos is the chance of a true detection
    # falsepos is the chance of a false detection
    if state == 3:
        detector3 = np.random.choice(2, 1, p=[1 - truepos, truepos])
        detector3 = bool(detector3)
    else:
        detector3 = np.random.choice(2, 1, p=[1 - falsepos, falsepos])
        detector3 = bool(detector3)
    if state == 5:
        detector5 = np.random.choice(2, 1, p=[1 - truepos, truepos])
        detector5 = bool(detector5)
    else:
        detector5 = np.random.choice(2, 1, p=[1 - falsepos, falsepos])
        detector5 = bool(detector5)

    measurements = [detector3, detector5]
    return measurements


def belief_update(b, z, B, T):
    # B is the observation matrix

    vehicle_state = b[0]
    b = np.array([b[1:]])
    new_b = T .dot(np.transpose(b))
    logging.debug('T = {}'.format(T))
    logging.debug('b {}'.format(new_b))

    if z[0]:
        py3 = np.diag(B[:, 0])
    else:
        py3 = np.diag(B[:, 1])
    if z[1]:
        py5 = np.diag(B[:, 2])
    else:
        py5 = np.diag(B[:, 3])

    new_b = py3.dot(py5.dot(new_b))
    new_b = new_b / np.linalg.norm(new_b, 1)

    new_b = np.squeeze(new_b)
    belief = [vehicle_state] + new_b.tolist()
    logging.debug('belief_update {}'.format(belief))
    return belief


def get_reward(vehicle_state, intruder_state, action):
    if vehicle_state == intruder_state or (vehicle_state + action) == intruder_state:
        reward = -1000
    elif vehicle_state + action in [4, 6]:
        reward = 1000
    else:
        reward = -1
    return reward


def AMDP(bbars, T, B, tp, fp):
    '''Learn Model'''
    P = []
    R = []
    n = 100
    for i, bbar in enumerate(bbars):
        P_i = []
        R_i = []
        actions = get_actions(bbar)
        for u, action in enumerate(actions):
            P_iu = []
            for j in range(len(bbars)):
                P_iu.append(0)
            R_i.append(0)

            if bbar == 't':
                P_iu[len(bbars)-1] = 1
                R_i[u] = 0
            else:
                for k in range(n):
                    # generate b with f(b) = bbar
                    b = finv(bbar)
                    # sample x ~ b(x)
                    intruder_state = sample_state(b[1:])
                    vehicle_state = b[0]
                    # get reward
                    reward = get_reward(vehicle_state, intruder_state, action)
                    if reward != -1:
                        P_iu[len(bbars)-1] += 1/n
                        R_i[u] += reward / n
                    else:
                        # update vehicle state
                        new_vehicle_state = vehicle_state + action
                        # sample x' ~ p(x'|u,x)
                        new_intruder_state = sample_movement(intruder_state, T)
                        # sample z ~ p(x|x')
                        measurements = sample_measurement(new_intruder_state, tp, fp)
                        # calculate b' = B(b,u,z)
                        new_b = belief_update(b, measurements, B, T)
                        new_b[0] = new_vehicle_state
                        # calculate b'bar = f(b')
                        new_bbar_idx = f(new_b, bbars)
                        logging.debug('index = {}'.format(new_bbar_idx))
                        P_iu[new_bbar_idx] += 1 / n
                        R_i[u] += reward / n

            P_i.append(P_iu)
        P.append(P_i)
        R.append(R_i)

    print 'Doing value iteration...'
    '''Value Iteration'''
    V = []
    W = []
    count = 0
    rmin = -1000
    gamma = 0.95

    # Initialize
    for i in range(len(bbars)):
        V.append(rmin)
        W.append(None)

    converged = False
    while not converged:
        V_old = copy.copy(V)
        for i, bbar in enumerate(bbars):
            U = []
            for u in range(len(R[i])):
                J = 0
                for j in range(len(bbars)):
                    J += V[j] * P[i][u][j]
                U.append(R[i][u] + J)

            V[i] = gamma * max(U)
            W[i] = U.index(max(U))

        if V_old == V:
            converged = True
        else:
            count += 1
            logging.debug('count = {}'.format(count))
    return V, P, R, W


def main(save=False):
    # augmented belief space (vehicle pos, probability of each branch (3))
    states = [0, 1, 2, 3, 5]
    precision_p = 10

    bbars = []
    for vehicle_state in states:
        for b02 in range(0, precision_p + 1):
            for b34 in range(0, precision_p - b02 + 1):
                b56 = precision_p - b02 - b34
                prob_branch_02 = b02 / precision_p
                prob_branch_34 = b34 / precision_p
                prob_branch_56 = b56 / precision_p
                bbars.append([vehicle_state, prob_branch_02, prob_branch_34, prob_branch_56])
    bbars.append('t')
    # print len(bbars)

    t = 0.2
    T = [[1, 0, 0, 0, 0, 0, 0],
         [t, 1-t, 0, 0, 0, 0, 0],
         [0, t, 1-t, 0, 0, 0, 0],
         [0, 0, t, 1-t, 0, 0, 0],
         [0, 0, 0, t, 1-t, 0, 0],
         [0, 0, t, 0, 0, 1-t, 0],
         [0, 0, 0, 0, 0, t, 1-t]]
    T = np.array(T)
    T = np.transpose(T)

    fp = 0.01
    tp = 0.95

    B = [[fp, 1-fp, fp, 1-fp],
         [fp, 1-fp, fp, 1-fp],
         [fp, 1-fp, fp, 1-fp],
         [tp, 1-tp, fp, 1-fp],
         [fp, 1-fp, fp, 1-fp],
         [fp, 1-fp, tp, 1-tp],
         [fp, 1-fp, fp, 1-fp]]
    B = np.array(B)

    V, P, R, W = AMDP(bbars, T, B, tp, fp)

    if save:
        folder = '/home/matt/Dropbox/Research/SavedResults/7nodeamdp/'
        np.save(folder + 'V', V)
        np.save(folder + 'P', P)
        np.save(folder + 'R', R)
        np.save(folder + 'W', W)
        np.save(folder + 'T', T)
        np.save(folder + 'B', B)
        np.save(folder + 'bbars', bbars)


def load_vars():
    try:
        folder = '/home/matt/Dropbox/Research/SavedResults/7nodeamdp/'
        V = np.load(folder + 'V.npy').tolist()
        P = np.load(folder + 'P.npy').tolist()
        R = np.load(folder + 'R.npy').tolist()
        W = np.load(folder + 'W.npy').tolist()
        T = np.load(folder + 'T.npy')
        B = np.load(folder + 'B.npy')
        bbars = np.load(folder + 'bbars.npy').tolist()
    except:
        print 'One or more variables are missing'
        pass

    return V, P, R, W, T, B, bbars


def run_trial(V, P, R, W, T, B, bbars, print_states=False):
    fp = 0.01
    tp = 0.95

    belief = [0, 0, 0, 0, .5, 0, .5]
    vehicle_state = 0
    intruder_state = sample_state(belief)
    bbar = [vehicle_state] + [0, .5, .5]
    idx = bbars.index(bbar)

    done = False
    reward = 0
    while not done:
        # Move vehicle
        action_idx = W[idx]
        actions = get_actions([vehicle_state])
        old_vehicle_state = vehicle_state
        old_intruder_state = intruder_state
        vehicle_state += actions[action_idx]
        # Move intruder
        intruder_state = sample_movement(intruder_state, T)
        # Sample Measurements
        measurements = sample_measurement(intruder_state, tp, fp)
        # Calculate belief
        belief = belief_update([vehicle_state] + belief, measurements, B, T)
        new_reward = get_reward(old_vehicle_state, old_intruder_state, actions[action_idx])
        if new_reward == -1:
            # Find bbar idx
            idx = f(belief, bbars)
            belief = belief[1:]
        if print_states:
            print 'Vehicle = {}'.format(old_vehicle_state)
            print 'Action = {}'.format(actions[action_idx])
            print 'Intruder = {}'.format(intruder_state)
            # print 'measurents = {}'.format(measurements)
            # print 'belief = {}'.format(belief)

        reward += new_reward
        if new_reward == 1000:
            done = True
            end = 'escaped'
        elif new_reward == -1000:
            done = True
            end = 'caught'

        # if vehicle_state == 1 and intruder_state == 0 and not done:
        #     print new_reward
        #     raw_input('wut')
    return reward, end


def run_trials(n=1000, print_states=False):
    rewards = []
    caught = 0
    escaped = 0
    V, P, R, W, T, B, bbars = load_vars()

    for i in range(n):
        reward, end = run_trial(V, P, R, W, T, B, bbars, print_states)
        rewards.append(reward)
        if end == 'caught':
            caught += 1
            print 'caught'
        else:
            escaped += 1
            print 'escaped'
        if print_states:
            raw_input("PRESS ENTER TO CONTINUE.")

    mean_reward = np.mean(rewards)
    expected_reward = V[bbars.index([0, 0, .5, .5])]

    print 'mean reward = {}'.format(mean_reward)
    print 'expected reward = {}'.format(expected_reward)
    print 'Caught = {}%'.format(caught/n*100)

if __name__ == '__main__':
    # main(save=True)
    run_trials(print_states=False)

    V, P, R, W, T, B, bbars = load_vars()

    # b = [1, 0.01, 0.02, 0.02, 0.91, 0.01, 0.02, 0.01]
    # idx = f(b, bbars)
    # print bbars[idx]
    # print V[idx]
    # action_idx = W[idx]
    # actions = get_actions([bbars[idx][0]])
    # print action_idx
    # print actions
    # print actions[action_idx]

    
    # for i in range(0,7):
    #     b = [i, 0.001, 0.002, 0.002, 0.99, 0.001, 0.002, 0.002]
    #     idx = f(b, bbars)
    #     actions = get_actions([bbars[idx][0]])
    #     action_idx = W[idx]
    #     print actions[action_idx]
