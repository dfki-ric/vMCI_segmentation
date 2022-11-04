'''
:Author: Lisa Gutzeit (Lisa.Gutzeit@uni-bremen.de)
:Last update: 2022-06
'''
from scipy.linalg import block_diag
from vmci_segmentation.lrm import *

class Segment():
    """ Represents a segment calculated by the MCI/vMCI algorithm implemented
    in :class:`.inference.ChangePointInference`.
    
    Attributes
    -----------
    data : array, shape (T, d)
        Observed data of this segment without velocities.
        
    data_vel : array, shape (T, num_sensors)
        Velocity of observed data. 
    
    t : array, shape (T, )
        Sample time points.
        
    start : int
        Start index of the segment in original unsegmented data.
        
    end : int
        End index of the segment in original unsegmented data.
        
    basis : instance of subclass of :class:`.lrm.Basis`
        Basis of the segment.
        
    weights : array, shape (model_order, d)
        Weights beta of the linear regression model.
        
    segmentation_algrithm : string, optional
        Name of segmentation algorithm the segment has been determined with.
        Defaults ``None``.
        
    segmentation_algrithm : string, optional
        Name of classification algorithm the segment label has been determined with.
        Defaults ``None``.
        
    label : int or string
        Class label of the segment.
                
    """
    def __init__(self, data, t, start, end,
                 basis,
                 weights,
                 segmentation_algorithm=None,
                 clustering_algorithm=None,                 
                 data_vel = [],
                 take_dims = []
                 ):
        self.data = data
        self.data_vel = data_vel
        self.data_orig = data
        self.t = t
        self.model = []
        self.start = start
        self.end = end
        self.basis = basis
        self.weights = weights
        self.label = -1
        self.label_prob = []
        self.true_labels = []
        self.annotation = ''
        self.segmentation_algorithm = segmentation_algorithm
        self.clustering_algorithm = clustering_algorithm
        
        self.demonstration_name = '' # gives the demonstration this segment belongs to
        
        if len(take_dims) == 0:
            self.take_dims = range(np.shape(self.data)[1])
        else:
            self.take_dims = take_dims
        self.take_dims_vel = [int(i/3.) for i in self.take_dims[0:-1:3]]
        
        if self.basis is not None:
            self.__calc_approximation_error()
    
    def set_label(self, l):
        """ Set label of the segment to `l`."""
        self.label = l
        
    def __calc_approximation_error(self):
        """
        Calculates the mean absolute distance between true and estimated
        trajectory.
        """
        self.data_est = np.dot(self.basis.evaluate(self.t, len(self.t)-1), self.weights) # H*beta
        
        if len(self.data_vel) == 0:
            diff = self.data[:, self.take_dims] - self.data_est
        else:
            diff = np.concatenate((self.data[:, self.take_dims], self.data_vel[:, self.take_dims_vel]), 1) - self.data_est
        self.appr_error = np.sqrt(np.mean(diff.flatten()**2))
        
    def _update_basis(self,basis):
        self.basis = basis
        

    def append(self, seg, delta):
        """
        Appends `seg` to segment (including update of all attributes).
        
        Parameters
        -----------
        
        seg : instance of :class:`.Segment`
            Segment that should be append to current segment instance.
            
        delta : array, shape (num_basis_func, ) 
            Delta (prior, variance of model parameter along columns) of segment 
            that should be appended
        """         
        if seg.start != self.end:
            raise RuntimeError("Segments should be consecutive.")
                
        # update basis
        for tt in range(len(seg.t)):
            self.basis.update(seg.t[tt], seg.data[tt, seg.take_dims], len(self.t)+tt)
        
        self.data = np.concatenate((self.data, seg.data))
        self.t = np.concatenate((self.t, seg.t))
        if len(self.data_vel) > 0:
            self.data_vel = np.concatenate((self.data_vel, seg.data_vel))
        self.end = seg.end
        
        if isinstance(self.basis, SplitBasis):
            H1 = self.basis.basis1.evaluate(self.t, len(self.t)-1)
            H2 = self.basis.basis2.evaluate(self.t, len(self.t)-1)
            H = np.concatenate((H1, H2), axis=1)
            
            D = np.diag(np.multiply(delta[0:self.basis.num_basis_func], delta[0:self.basis.num_basis_func]))
        
            if isinstance(self.basis.basis1, ARBasis):
                D1 = np.kron(np.diag(np.multiply(delta[0:self.basis.basis1.num_basis_func], delta[0:self.basis.basis1.num_basis_func])), 
                             np.eye(len(self.take_dims)))
                
                D2 = np.diag(np.multiply(delta[self.basis.basis1.num_basis_func::], delta[self.basis.basis1.num_basis_func::]))
                
                D = block_diag(D1, D2)
        else:
            H = self.basis.evaluate(self.t, len(self.t)-1)
            
            D = np.diag(np.multiply(delta[0:self.basis.num_basis_func], delta[0:self.basis.num_basis_func]))
        
            if isinstance(self.basis, ARBasis):
                D = np.kron(D, np.eye(len(self.take_dims))) 
                
        D_inv = np.linalg.inv(D)
                
        # beta = (H^T H + D^-1)^-1 H^T Y
        seg.weights = np.dot(np.linalg.inv(np.dot(H.transpose(), H)+D_inv), np.dot(H.transpose(), self.data[:, self.take_dims]))
        
        
        self.data_est = np.dot(self.basis.evaluate(self.t, len(self.t)-1), self.weights)
        
        if len(self.data_vel) == 0:
            diff = self.data[:, self.take_dims] - self.data_est
        else:
            diff = np.concatenate((self.data[:, self.take_dims], self.data_vel[:, self.take_dims_vel]), axis=1)
        self.appr_error = np.sqrt(np.mean(diff.flatten()**2))
