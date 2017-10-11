from __future__ import division

import numpy as np
import logging
import copy


def get_actions(bbar):
        if bbar[0] == 0:
            actions = [0, 1, 2]
        elif bbar[0] == 't':
            actions = ['t']
        return actions


def f(b, bs, precision=1):
    # Generates bbar from b
    # Precision must exactly match bbars
    vehicle_state = b[0]
    b = b[1:]
    b0 = round(b[0], precision)
    b1 = round(b[1], precision)
    b2 = round(b[2], precision)

    total = b0 + b1 + b2
    if total != 1:
        logging.debug('Total != 1')
        b0 = 1 - b1 - b2
        b0 = round(b0, precision)

    belief = [b0, b1, b2]
    b = [vehicle_state] + belief

    return bs.index(b)


def sample_state(belief):
    state = np.random.choice(3, 1, p=belief)
    return state


def sample_movement(old_state, T):
    # T is the transition matrix
    p = np.reshape(T[:, old_state], len(T[:, old_state]))
    p = p / sum(p)
    new_state = np.random.choice(3, 1, p=p)
    return new_state


def sample_measurement(state, truepos, falsepos):
    # truepos is the chance of a true detection
    # falsepos is the chance of a false detection
    if state == 1:
        detector1 = np.random.choice(2, 1, p=[1 - truepos, truepos])
        detector1 = bool(detector1)
    else:
        detector1 = np.random.choice(2, 1, p=[1 - falsepos, falsepos])
        detector1 = bool(detector1)
    if state == 2:
        detector2 = np.random.choice(2, 1, p=[1 - truepos, truepos])
        detector2 = bool(detector2)
    else:
        detector2 = np.random.choice(2, 1, p=[1 - falsepos, falsepos])
        detector2 = bool(detector2)

    measurements = [detector1, detector2]
    return measurements


def belief_update(b, z, B, T):
    # B is the observation matrix

    vehicle_state = b[0]
    b = np.array([b[1:]])
    new_b = T .dot(np.transpose(b))
    logging.debug('T = {}'.format(T))
    logging.debug('b {}'.format(new_b))

    if z[0]:
        py1 = np.diag(B[:, 0])
    else:
        py1 = np.diag(B[:, 1])
    if z[1]:
        py2 = np.diag(B[:, 2])
    else:
        py2 = np.diag(B[:, 3])

    new_b = py1.dot(py2.dot(new_b))
    new_b = new_b / np.linalg.norm(new_b, 1)

    new_b = np.squeeze(new_b)
    belief = [vehicle_state] + new_b.tolist()
    logging.debug('belief_update {}'.format(belief))
    return belief


def get_reward(vehicle_state, intruder_state, action):
    if vehicle_state == intruder_state or (vehicle_state + action) == intruder_state:
        reward = -1000
    elif vehicle_state + action in [1, 2]:
        reward = 1000
    else:
        reward = -1
    return reward


def MDP(bs, T, B, tp, fp):
    '''Learn Model'''
    P = []
    R = []
    PO = []
    for i, b in enumerate(bs):
        P_i = []
        R_i = []
        PO_i = []
        actions = get_actions(b)
        for u, action in enumerate(actions):
            P_iu = []
            for j in range(len(bs)):
                P_iu.append(0)
            R_i.append(0)

            if b == 't':
                P_iu[len(bs)-1] = 1
                R_i[u] = 0
            else:
                p_s = np.array([b[1:]])
                p_s = np.transpose(p_s)
                p_sp = T.dot(p_s)
                # Calculate R
                if action == 1:
                    R_i[u] = 1000*p_s[2, 0] + -1000*p_s[1, 0] + -1000*p_s[0, 0]
                elif action == 2:
                    R_i[u] = 1000*p_s[1, 0] + -1000*p_s[2, 0] + -1000*p_s[0, 0]
                else:
                    R_i[u] = -1*p_s[2, 0] + -1*p_s[1, 0] + -1000*p_s[0, 0]

                if action != 0:
                    P_iu[len(bs)-1] = 1
                else:
                    observation_set = [[True,True],[False,True],[True,False],[False,False]]
                    for o, observation in enumerate(observation_set):
                        # find observation probability
                        m1 = 0 + (1 - observation[0])
                        m2 = 2 + (1 - observation[1])
                        o_prob = 0
                        for l in range(3):
                            o_prob += B[l,m1]*B[l,m2]*p_sp[l]
                        # find b'
                        new_b = belief_update(b, observation, B, T)
                        # get idx
                        b_idx = f(new_b, bs)
                        PO_i.append([o_prob, b_idx])

                        P_iu[b_idx] = o_prob[0]
            P_iu = [x / np.linalg.norm(P_iu, 1) for x in P_iu]
            P_i.append(P_iu)
        PO.append(PO_i)
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
    for i in range(len(bs)):
        V.append(rmin)
        W.append(None)

    converged = False
    while not converged:
        V_old = copy.copy(V)
        for i, b in enumerate(bs):
            U = []
            for u in range(len(R[i])):
                J = 0
                for j in range(len(bs)):
                    J += V[j] * P[i][u][j]
                U.append(R[i][u] + J)

            V[i] = gamma * max(U)
            W[i] = U.index(max(U))

        if V_old == V:
            converged = True
        else:
            count += 1
            logging.debug('count = {}'.format(count))
    return V, P, R, W, PO


def main(save=False):
    t = 0.2
    T = [[1, 0, 0],
         [t, 1-t, 0],
         [t, 0, 1-t]]
    T = np.array(T)
    T = np.transpose(T)

    fp = 0.01
    tp = 0.95

    B = [[fp, 1-fp, fp, 1-fp],
         [tp, 1-tp, fp, 1-fp],
         [fp, 1-fp, tp, 1-tp]]
    B = np.array(B)

    bs = []
    for j in range(0, 11):
        for k in range(0, 11-j):
            b = [0, j/10, k/10, (10-j-k)/10]
            bs.append(b)
    bs.append('t')

    V, P, R, W, PO = MDP(bs, T, B, tp, fp)

    if save:
        folder = 'SavedResults/3nodeexact/'
        np.save(folder + 'V', V)
        np.save(folder + 'P', P)
        np.save(folder + 'R', R)
        np.save(folder + 'W', W)
        np.save(folder + 'T', T)
        np.save(folder + 'B', B)
        np.save(folder + 'bs', bs)
        np.save(folder + 'PO', PO)


def load_vars():
    try:
        folder = 'SavedResults/3nodeexact/'
        V = np.load(folder + 'V.npy').tolist()
        P = np.load(folder + 'P.npy').tolist()
        R = np.load(folder + 'R.npy').tolist()
        W = np.load(folder + 'W.npy').tolist()
        T = np.load(folder + 'T.npy')
        B = np.load(folder + 'B.npy')
        bs = np.load(folder + 'bs.npy').tolist()
    except:
        print 'One or more variables are missing'
        pass

    return V, P, R, W, T, B, bs


def run_trial(V, P, R, W, T, B, bs, print_states=False):
    fp = 0.01
    tp = 0.95

    belief = [0, .5, .5]
    vehicle_state = 0
    intruder_state = sample_state(belief)
    b = [vehicle_state] + [0, .5, .5]
    idx = bs.index(b)

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
        # Find bbar idx
        if vehicle_state == 0:
            idx = f(belief, bs)
            belief = belief[1:]
        if print_states:
            print 'Action = {}'.format(actions[action_idx])
            print 'Vehicle = {}'.format(vehicle_state)
            print 'Intruder = {}'.format(intruder_state)
            print 'measurents = {}'.format(measurements)
            print 'belief = {}'.format(belief)

        new_reward = get_reward(old_vehicle_state, old_intruder_state, actions[action_idx])
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
    V, P, R, W, T, B, bs = load_vars()

    for i in range(n):
        reward, end = run_trial(V, P, R, W, T, B, bs, print_states)
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
    expected_reward = V[bs.index([0, 0, .5, .5])]

    print 'mean reward = {}'.format(mean_reward)
    print 'expected reward = {}'.format(expected_reward)
    print 'Caught = {}%'.format(caught/n*100)
    # print V

if __name__ == '__main__':
    main(save=True)
    run_trials(print_states=False)
