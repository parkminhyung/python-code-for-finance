import numpy as np

def find_extrema(x, m, mode='peak'):
    """
    Detect peaks (local maxima) or valleys (local minima) in a given 1D array x
    based on m points to the left and right.
    
    Parameters:
        x : 1D numpy array
        m : int, number of points to compare on each side
        mode : str, 'peak' to detect peaks, 'valley' to detect valleys
    
    Returns:
        extrema : numpy array, indices of detected extrema (peaks or valleys)
    """
    x = np.asarray(x)
    dx = np.diff(x)
    sign_dx = np.sign(dx)
    shape = np.diff(sign_dx)
    
    if mode == 'peak':
        candidate_indices = np.where(shape < 0)[0]  # Peak candidates (negative change)
    elif mode == 'valley':
        candidate_indices = np.where(shape > 0)[0]  # Valley candidates (positive change)
    else:
        raise ValueError("mode must be either 'peak' or 'valley'.")
    
    extrema = []
    for i in candidate_indices:
        idx = i + 1  # Candidate index in x
        left_start = max(idx - m, 0)
        left_window = x[left_start:idx]
        right_end = min(idx + m + 1, len(x))
        right_window = x[idx+1:right_end]
        
        if mode == 'peak':
            condition = np.all(left_window <= x[idx]) and np.all(right_window <= x[idx])
        else:  # mode == 'valley'
            condition = np.all(left_window >= x[idx]) and np.all(right_window >= x[idx])
        
        if condition:
            extrema.append(idx)
    
    return np.array(extrema)
