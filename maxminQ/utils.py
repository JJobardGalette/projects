import numpy as np


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x, taken from
        KTH RL labs
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y