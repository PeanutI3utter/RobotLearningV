import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


##
def genGridWorld():
    O = -1e5  # Dangerous places to avoid
    D = 35    # Dirt
    W = -100  # Water
    C = -3000 # Cat
    T = 1000  # Toy
    grid_list = {0:'', O:'O', D:'D', W:'W', C:'C', T:'T'}
    grid_world = np.array([[0, O, O, 0, 0, O, O, 0, 0, 0],
        [0, 0, 0, 0, D, O, 0, 0, D, 0],
        [0, D, 0, 0, 0, O, 0, 0, O, 0],
        [O, O, O, O, 0, O, 0, O, O, O],
        [D, 0, 0, D, 0, O, T, D, 0, 0],
        [0, O, D, D, 0, O, W, 0, 0, 0],
        [W, O, 0, O, 0, O, D, O, O, 0],
        [W, 0, 0, O, D, 0, 0, O, D, 0],
        [0, 0, 0, D, C, O, 0, 0, D, 0]])
    return grid_world, grid_list


##
def showWorld(grid_world, tlt):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(tlt)
    ax.set_xticks(np.arange(0.5,10.5,1))
    ax.set_yticks(np.arange(0.5,9.5,1))
    ax.grid(color='b', linestyle='-', linewidth=1)
    ax.imshow(grid_world, interpolation='nearest', cmap='copper')
    return ax


##
def showTextState(grid_world, grid_list, ax):
    for x in range(grid_world.shape[0]):
        for y in range(grid_world.shape[1]):
            if grid_world[x,y] >= -3000:
                ax.annotate(grid_list.get(grid_world[x,y]), xy=(y,x), horizontalalignment='center')


##
def showPolicy(policy, ax):
    for x in range(policy.shape[0]):
        for y in range(policy.shape[1]):
            if policy[x,y] == 0:
                ax.annotate(r'$\downarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 1:
                ax.annotate(r'$\rightarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 2:
                ax.annotate(r'$\uparrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 3:
                ax.annotate(r'$\leftarrow$', xy=(y,x), horizontalalignment='center')
            elif policy[x,y] == 4:
                ax.annotate(r'$\perp$', xy=(y,x), horizontalalignment='center')


height = 9
width = 10

actions = {
    0: {'y': -1, 'x': 0},
    1: {'y': 0, 'x': 1},
    2: {'y': 1, 'x': 0},
    3: {'y': 0, 'x': -1},
    4: {'y': 0, 'x': 0}
}


def legal(y, x, action):
    """
    Returns:
        True if given action is legal
    """
    return (y + actions[action]['y'] >= 0 and y + actions[action]['y'] < height
            and x + actions[action]['x'] >= 0 and x + actions[action]['x'] < width)


def deterministic_trans(y, x, action):
    """
    Deterministic transition function
    Args:
        y: y coordinate of current state
        x: x coordinate of current state
        action: action to be taken
    Returns:
        A dictionary containing the next state after taking given real action
        (action that has really happened) and its probability
    """
    assert(legal(y, x, action))
    return {action: 1}


##
def ValIter(R, discount, maxSteps, infHor, probModel=None):
    # YOUR CODE HERE
    # generate possible actions
    if probModel is None:
        probModel = deterministic_trans
    V = np.zeros((height, width, maxSteps))
    for t in range(maxSteps):
        # update value functions for all states
        for h in range(height):
            for w in range(width):
                # search for action maximizing value function for current state
                vmax = None
                for i in range(5):
                    if legal(h, w, i):
                        rap = probModel(h, w, i)
                        vnew = 0
                        for real_action in rap:
                            new_state = {
                                'y': h + actions[real_action]['y'],
                                'x': w + actions[real_action]['x']
                            }
                            vnew += (
                                rap[real_action]
                                * (R[new_state['y'], new_state['x']]
                                    + discount * V[new_state['y'], new_state['x'], maxSteps - t - 1])
                            )
                        if vmax is None or vmax < vnew:
                            vmax = vnew
                V[h, w, maxSteps - t - 2] = vmax
    return V


##
def maxAction(V, R, discount, probModel=None):
    # YOUR CODE HERE
    pass

##
def findPolicy(V, probModel=None):
    # YOUR CODE HERE
    if probModel is None:
        probModel = deterministic_trans
    p = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            a_max = None
            a_val = None
            for a in range(5):
                if legal(h, w, a):
                    rap = probModel(h, w, a)
                    v = 0
                    for real_action in rap:
                        new_state = {
                            'y': h + actions[real_action]['y'],
                            'x': w + actions[real_action]['x']
                        }
                        v += (
                            rap[real_action]
                            * (
                                V[new_state['y'], new_state['x']])
                        )
                    if a_max is None or a_val < v:
                        a_max = a
                        a_val = v
            p[h, w] = a_max
    return p
############################

saveFigures = True

data = genGridWorld()
grid_world = data[0]
grid_list = data[1]

# YOUR CODE HERE
probModel = ...

ax = showWorld(grid_world, 'Environment')
showTextState(grid_world, grid_list, ax)
if saveFigures:
    plt.savefig('gridworld.pdf')

# Finite Horizon
V = ValIter(grid_world, 1, 15, False)
V = V[:,:,0];
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon')
if saveFigures:
    plt.savefig('value_Fin_15.pdf')

policy = findPolicy(V)
ax = showWorld(grid_world, 'Policy - Finite Horizon')
showPolicy(policy, ax)
if saveFigures:
    plt.savefig('policy_Fin_15.pdf')

# Infinite Horizon
V = ValIter(...)
showWorld(np.maximum(V, 0), 'Value Function - Infinite Horizon')
if saveFigures:
    plt.savefig('value_Inf_08.pdf')

policy = findPolicy(...);
ax = showWorld(grid_world, 'Policy - Infinite Horizon')
showPolicy(policy, ax)
if saveFigures:
    plt.savefig('policy_Inf_08.pdf')

# Finite Horizon with Probabilistic Transition
V = ValIter(...)
V = V[:,:,0];
showWorld(np.maximum(V, 0), 'Value Function - Finite Horizon with Probabilistic Transition')
if saveFigures:
    plt.savefig('value_Fin_15_prob.pdf')

policy = findPolicy(...)
ax = showWorld(grid_world, 'Policy - Finite Horizon with Probabilistic Transition')
showPolicy(policy, ax)
if saveFigures:
    plt.savefig('policy_Fin_15_prob.pdf')
