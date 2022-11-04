import numpy as np
from vmci_segmentation.lrm import ARBasis, RBFBasisVel
from numpy.testing import assert_almost_equal


def test_ARBasis():
    # reconstruct AR function with weight beta=2
    data = np.array([[2,4,8,16]]).T
    beta = 2
    
    basis = ARBasis(1, 1,np.array([2,1]))
    x = np.array([0,1,2,3])
    
    for i, y in enumerate(data):
        basis.update(x[i], y, i)
    
    H = basis.evaluate(x, len(x)-1)
    assert_almost_equal((H*beta), data)


def test_RBFBasis():
    # reconstruct Gaussian curve with center = 2  and width r= 1 
    
    y = np.array([[np.exp(-1), 1, np.exp(-1)]]).T
    x = np.array([1, 2, 3])
    basis = RBFBasisVel(r=1, num_centers=1, center=1)
    
    for i, y_t in enumerate(y):
        basis.update(x[i], y_t, i)
    
    H = basis.evaluate(x, len(x-1))
    assert_almost_equal(np.dot(H, np.array([[1, 0]]).T), y)

