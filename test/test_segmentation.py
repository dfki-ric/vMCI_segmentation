import numpy as np
from numpy.testing import assert_almost_equal

from vmci_segmentation.vmci import VMCI
from vmci_segmentation.preprocessing import normalize_differences

def test_segmentation():
    # check correct segmentation on simple artificial data
    pos = np.concatenate((np.ones((15,3)), 2*np.ones((10,3))))
    vel = np.concatenate((np.zeros(1),np.diff(pos[:, 0], axis=0)))
    t = np.linspace(0, 2, 25)
    
    pos, vel = normalize_differences(pos,
                                     vel.reshape(-1, 1))

    vmci = VMCI(1, verbose=0)
    changepoints, segments = vmci.segment(pos, vel, t)
            
    assert_almost_equal(changepoints,
                     [0, 16, 24])
   





