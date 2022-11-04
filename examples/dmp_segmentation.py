'''
Example use of vMCI segmentation on artificial generated data (two 
sequenced Dynamical Movement Primitives (DMPs)).

:Author: Lisa Gutzeit (Lisa.Gutzeit@uni-bremen.de)
:Last update: 2022-06
'''
import os
import pylab as pl
import numpy as np
from vmci_segmentation.vmci import VMCI
from vmci_segmentation.preprocessing import normalize_differences


data = np.genfromtxt(os.path.join('data', 'sequenced_dmps.csv'), delimiter = ',')
time = data[:, 0]
position = data[:, 1]
velocity = data[:, 2]
subgoal = [0.4, 0.3]
   
pl.figure()
pl.title('2 sequenced DMPs')
pl.ylabel('position')
pl.xlabel('time')
pl.plot(time, position, label='position')
pl.plot(time, velocity, label='velocity')
pl.scatter(subgoal[0], subgoal[1], color='g', label='subgoal (true segment border)')
pl.legend(numpoints=1, loc=4)


# pre-process data to zero mean and variance of first order differences
# equal to one
position, vel = normalize_differences(position.reshape(-1, 1),
                                      velocity.reshape(-1, 1))

# segment data using vMCI algorithm
vmci = VMCI(1)
changepoints, segments = vmci.segment(position, vel, time, 
                                      demo_name='sequenced DMPs'
                                      )

vmci.plot(segments, 2, 'sequenced DMPs')

pl.show()