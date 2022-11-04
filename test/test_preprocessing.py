import numpy as np
from vmci_segmentation.preprocessing import normalize_differences
from numpy.testing import assert_almost_equal



def test_normalize():
    data = np.array([[1,3,5,6,8], # artificial positsion
                         [0.2, 0.4, 0.5, 0.4, 0.2]]).T # artificial velocity
        
    pos_normalized, vel_normalized = normalize_differences(data[:, 0],
                                                               data[:, 1])
        
    # check if normalized values have zero mean
    assert_almost_equal(np.mean(pos_normalized),  0)
        
    # check if normalized values have first order variance of one
    assert_almost_equal(np.var(np.diff(pos_normalized, axis=0)), 1)
        
        # check if shape of velocity has not changed
    assert_almost_equal(np.shape(vel_normalized),
                         np.shape(data[:, 1]))
        


   
    