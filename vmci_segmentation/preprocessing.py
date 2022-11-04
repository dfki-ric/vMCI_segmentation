'''
Methods to pre-process data before segmentation using vMCI.

:Authors: Lisa Gutzeit (lisa.gutzeit@uni-bremen.de)
:Last update: 2022-06
'''
import numpy as np

def normalize_differences(data, data_vel=None):
    """
    Pre-processes data for segmentation so that the variances of the first order
    differences is one. If position and velocity data is given, the velocity
    is scaled in order that it's in the same range as the (position-) data.
    
    Parameters
    -----------
    data : array-like of shape (T, d)
        Original data, e.g. sensor positions.
    
    data_vel : array-like of shape (T, num_sensors), optional
        Velocity of each sensor. Defaults ``None``.
            
    Returns
    -------
    data_proc : array-like of shape (T, d)
        Pre-processed data.
        
    data_vel_proc : array-like of shape (T, num_sensors)
        Pre-processed velocity.
        
    """    
    if data.ndim==1:
        data = data.reshape(-1, 1)
    
    d = np.shape(data)[1]
    
    # set mean to zero
    data_proc = data - np.mean(data, 0).reshape(1, d)
    
    # set first order variances to 1
    data_proc = data_proc / np.sqrt(np.var(np.diff(data_proc, axis=0), 0).reshape(1, d))
    
    data_proc[np.isinf(data_proc)] = 0 # data is constant the whole time
    
    
    if data_vel is not None:
        # scale velocity in order that it is in the same range as positions
        data_vel_proc = (np.max(data_proc)/np.max(data_vel))*data_vel 
    else:
        data_vel_proc = None
        
    return data_proc, data_vel_proc
