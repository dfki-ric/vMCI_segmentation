# vmci_segmentation

Implementation of the velocity-based Multiple Change-point Inference (vMCI) algorithm 
to segment human manipulation movement into movement building blocks based on the velocity
of the hand. The algorithm is introduced in (Senger et al. 2014, https://dl.acm.org/doi/abs/10.1109/ICPR.2014.781).

## Installation

After cloning the repository, install it with

    python setup.py install
	
OR by using pip:
	
	pip install -e .
	
To install the packages needed to build the documentation and/or to run tests use:

	pip install -e .[doc,test]
	
## Documentation

The docmentation of this project can be found in the directory `doc`. To
build the documentation, run on windows:

	cd doc
	make.bat html

And on linux:
	
	cd doc
	make html

The HTML documentation is now located at `doc/build/html/index.html`.

## Example

The following example segments synthetical data into units with a bell-shaped velocity profile. 
The example can be found in the folder `examples`, the resulting images are shown below.

```python
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
```
Synthetic data which should be segmented. The true segmentation point is shown as green dot:
<img src="/examples/dmp_segmentation_00.png?raw=true" height="300px"/>

Result of unsupervised vMCI segmentation. The determined segmentation point is shown as red vertical line:
<img src="/examples/dmp_segmentation_01.png?raw=true" height="400px"/>

## Contributing
If you wish to report bugs, please create an issue.

Directly pushing to the main branch (master) is not allowed. If you want to add new features, documentation or bug fixes, please open a pull/merge request.

## Testing
To run test use:

	pytest

## Funding

This library has been developed at the Robotics Group of the University of Bremen.
The work was supported through two grants of the German
Federal Ministry of Economic Affairs and Energy (BMWi, FKZ 50 RA 1217 and FKZ 50 RA 1703).
