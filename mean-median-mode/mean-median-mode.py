from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    #calculate mean
    mean=float(np.mean(x))
    #sort and calculate median
    median=float(np.median(x))
    #find mode
    counts=Counter(x)
    max_freq=max(counts.values())
    mode=float(min(key for key,freq in counts.items() if freq==max_freq))

    dict={'mean':mean,'median':median,'mode':mode}
    return dict
    
    
    pass