import numpy as np

# This one is biased (like np.std(ddof=0))
# To get the unbiased one: DescrStatsW(values, weights, ddof=1).std
def weighted_std(values, weights, axis):

    average = np.average(values, weights=weights, axis=axis)
    variance = np.average((values-average)**2, weights=weights, axis=axis)

    return np.sqrt(variance)

