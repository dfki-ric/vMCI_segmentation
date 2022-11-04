User guide
=====================

There are multiple ways to use the methods provided in this python repository in order to segment
human movement data into building blocks with a bell-shaped velocity profile using the vMCI
algorithm. To run the segmentation, position and velocity of at 
least one sensor is needed. In the ideal case, this sensor should have recorded the position and velocity of the hand. Before 
segmentation, the data needs to be preproced to a zero mean and first order variance of one. The easiest way is to used the class :class:`.vmci.VMCI`, in shich most parameters that need to be defined for inference are calculated from the data per default. An example usage on artificial data
can be fould below.


Example: Segmenting concatenation of synthetic DMPs
-----------------------------------------------------

.. plot:: ../../examples/dmp_segmentation.py
    :include-source:
	
	
Example: Segmenting pick-and-place movements
-----------------------------------------------------

.. plot:: ../../examples/pnp_segmentation.py
    :include-source: