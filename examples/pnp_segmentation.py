'''
Example use of vMCI segmentation on real human motion data. The data consists
of pick-and-place movements recorded using the Qualisys motion
capture system as described in detail in :cite:p:`Gutzeit2022`.

:Author: Lisa Gutzeit (Lisa.Gutzeit@uni-bremen.de)
:Created: 2022-11
'''
import pandas, os, json
import pylab as pl
import numpy as np

from vmci_segmentation.vmci import VMCI
from vmci_segmentation.preprocessing import normalize_differences


# select demonstration which should be segmented
# (currently selected movement demonstration corresponds to vMCI result
# in Figure 8 in :cite:p:`Gutzeit2022`)
filename = r'data\pick_and_place_mocap_data\subject1_run6.tsv'

# load ground truth
segments_manually = json.load(open(r'data\pick_and_place_mocap_data\subject1_run6_manually_segmented.json', 'r'))
true_changepoints = []
for segment in segments_manually:
    true_changepoints.extend([segment['start index'], segment['end index']])
    
true_changepoints = list(set(true_changepoints)) # remove double entries
true_changepoints.sort()

# down-sample to 20 Hz and start at 0
true_changepoints = [int(np.round(cp/3)) for cp in true_changepoints]
true_changepoints = [cp-true_changepoints[0] for cp in true_changepoints]

true_changepoints[-1] = true_changepoints[-1] - 1 


# load data recorded at 60 Hz
hand_position = pandas.read_csv(filename, sep='\t', 
                               skiprows=10,
                               na_values=['null'], 
                               usecols=['Hand top X', 'Hand top Y', 'Hand top Z']).values

# down-ample to 20 Hz
hand_position = hand_position[::3,:]

# read time component and down sample
t = pandas.read_csv(filename, sep='\t', 
                               skiprows=10,
                               na_values=['null'], 
                               usecols=['Time']).values[::3,:]

# calculate absolute velocity
hand_velocity = np.concatenate(([0],
                                np.sqrt(np.sum(np.square(np.diff(hand_position, axis=0)), 1)/(t[1]-t[0]))
                                )).reshape(-1, 1)

# pre-process data to zero means and variance of first order differences
# equal to one
position, velocity = normalize_differences(hand_position, hand_velocity)

# segment data using vMCI algorithm
vmci = VMCI(1)
changepoints, segments = vmci.segment(position, velocity, t)

# plot results
fig = pl.figure(figsize=(8, 2))
ax = fig.gca()

demo_name = os.path.split(filename)[-1][:-4]
pl.title(f'vMCI segmentation ({demo_name})')

#scale velocity to range of position
velocity_scaled = (np.max(hand_position)/np.max(hand_velocity))*hand_velocity

ax.plot(t, velocity_scaled, 'k', lw=2, label='vel. right hand')
ax.plot(t, hand_position[:, 0], '--k', lw=2, label='x position')
ax.plot(t, hand_position[:, 1], '-.k', lw=2, label='y position')
ax.plot(t, hand_position[:, 2], ':k', lw=2, label='z position')

# plot manually determined change points
ax.axvline(t[true_changepoints[0]], lw=2, color='g', label='true segment borders')
for s in true_changepoints[1::]:
    ax.axvline(t[s], lw=2, color='g')

# plot vMCI change points
ax.axvline(t[changepoints[0]], lw=2, color='r', label='segment borders')
for s in changepoints[1::]:
    ax.axvline(t[s], lw=2, color='r')
    
ax.set_xlabel('time (s)', fontsize=10)
ax.set_ylabel('position/velocity', fontsize=10)

ax.set_yticks([])
ax.set_ylim(0, np.max(velocity_scaled))

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
pl.tight_layout()

pl.show()