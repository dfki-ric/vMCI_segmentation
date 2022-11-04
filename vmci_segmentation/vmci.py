# -*- encoding: utf-8 -*-
'''
:Author: Lisa Gutzeit (Lisa.Gutzeit@uni-bremen.de)
:Last update: 2022-06
'''
import numpy as np

from .inference import ChangePointInference
from .lrm import LRM
from .visualizations import plot_changepoints_time, plot_changepoints_space 

class VMCI(object):
    r"""Velocity-based Multiple Change-point Inference :cite:p:`Senger2014`.
    
    Class to automatically segment human manipulation movements into
    building blocks with a bell-shaped velocity of the hand. Example
    usage can be found in experiments/dmp_segmentation. Details about
    the algorithm can be found in :cite:p:`Senger2014`.
    
    In this implementation an autoregressive basis function of order 1 is used as basis
    for the data except the velocity. The velocity dimensions are modeled with
    a special radial basis functions as described in :cite:p:`Senger2014` with 3 possible 
    centers and model order fixed to 2. The hyper-parameters are set as un-informative priors.
    The 'SRC' particle filter with N=100 and alpha = 1e-20 is used to speed up the 
    inference, see :cite:p:`Fearnhead2007` for details about the implemented filtering algorithms.
    
    The data (without velocity) is assumed to have shape (T, d). d = 3*num_sensors
    if the sensor data is in Euclidean space. Velocity data is assumed
    to have shape (t, num_sensors).
    
    .. Note:: For `num_sensors` > 1 segment borders are assigned to positions
            where **all** sensors have a local velocity minima at the same
            time.
       
    Attributes
    ----------
    
    num_sensors : int
        Number of sensors of the movement data.
        
    p : float, optional
        Prior of segments length. If ``None`` it is calculated from the data and
        set to *1/T*. Defaults ``None``.
        
    delta : array, shape (q+2, ) optional
        Prior, variance of model parameters along columns. delta[:q]
        refers to prior of AR basis; delta[-2::]
        refers to prior of velocity basis. If ``None`` it is 
        calculated from the data and set to *10*ones(q)*
        for AR basis and *sqrt(max(data_vel)/2)*ones(2)* for velocity basis. 
        Defaults ``None``.
    
    nu : double, optional
        Prior, Inverse-Wishart noise prior parameter. If ``None`` it is 
        calculated from the data and set to *d*. Defaults ``None``.
        
    gamma : array, shape (d, d), optional
        Prior, Inverse-Wishart noise prior parameter. If ``None`` it is 
        calculated from the data and set to *diag(diag(cov(diff(data))))*. 
        Gamma for the velocity dimension is calculated the same way using the
        velocity data.
        Defaults ``None``.
    
    take_dims : list of int, optional
        Specifies dimensions (columns of data that should be segmented) to be 
        considered during segmentation. If ``None`` all dimensions are used. 
        Defaults ``None``.
        
    verbose : int, optional
        Verbosity level. Defaults ``1``.
        
    basis_func_vel : ['rbfv', 3]
        Defines used basis function and model order for velocity data.
        ``['rbfv', 1]`` corresponds to velocity radial basis function as
        defined in :cite:p:`Senger2014` with 3 possible center positions.
                    
    """
    
    def __init__(self, num_sensors, p=None, delta=None, nu=None, gamma=None,
                 take_dims=None, verbose=1, basis_func_vel = ['rbfv', 3] ):
        
        self.num_sensors = num_sensors
        
        self.basis_func = ['ar', 1]
        self.basis_func_vel = basis_func_vel
        self.q = self.basis_func[1]
        
        self.p = p
        
        self.nu = nu
        self.nu_vel = num_sensors + 2
        
        if gamma is None:
            self.gamma = None
            self.gamma_vel = None 
        else:
            self.gamma_vel = self.gamma[-self.num_sensors::, :][:, -self.num_sensors::]
            self.gamma = self.gamma[0:-self.num_sensors, :][:, 0:-self.num_sensors] 
        
        if delta is None: 
            self.delta = None
            self.delta_vel = None
        else:
            self.delta_vel = delta[-2::]
            self.delta = delta[0:-2] 
                
        self.verbose = verbose
        self.take_dims = take_dims
        
        self.inference = []
        
    
    def segment(self, data, data_vel, time, seed=None, filter_alg=['SRC', 1e-20], 
                N=100, demo_name = None):
        """
        Segments the data into sequences with a bell-shaped velocity. Each
        resulting segment has hat least a length of 2.
        
        
        Parameters
        ----------
        data : array-like, shape=(T, d)
            Observed data without velocity.
            
        data_vel : array-like, shape=(T, num_sensors)
            Velocity of the observed data. 
            
        time : array_like, shape=(T,)
            Sample points of the data.
            
        seed : int, optional
            Random seed.
            
        filter_alg : list, optional
            Defines algorithm to filter particles during inference. 
            Must be one of *['SCR', a]* or *['SOR', M]*, with *a* and *M*
            parameters of the algorithm (int).
            Defaults: ``['SCR',  1e-20]``.
        
        N : int, optional
            Maximal number of particles. Defaults ``100``.
            
        demo_name : string, optional
            Name of the demonstration. 
            If ``None``, demonstrations name is set to *'demo 1'*. 
            Defaults ``None``.
            
        Returns
        -------
        changepoints : list of ints
            Indices of segment borders/change-points.
        
        segments : list of :class:`.segments.Segment`
            Resulting segments.
        
        """
        if data_vel is not None:
            data_vel = np.array(data_vel, ndmin=2)
            
        if len(time.shape) > 1:
            time = time[:, 0]
        
        if demo_name is None:
            demo_name = 'demo 1'
        
        # set which columns of given data should be used for inference (default: use all dimensions)
        self._set_takedims(np.shape(data)[1])
                
        # determine hyper-parameters if not given
        self._determine_parameters(data, data_vel)
        
        # initialize inference
        T = len(time)
        self._init_inference(N, filter_alg, seed)
        
        # run MCI inference
        if self.verbose:
            print("Inference of " + demo_name)
        for t in range(T):
            data_t = self._determine_datat(data, t, data_vel)
            self.inference.evaluate_time_point(t, time[t], data_t)
            
            if (t%100==0) and self.verbose:
                print("t: %i/%i "%(t, T))
        
        changepoints = self._read_changepoints()
        
        segments = self.inference.get_segments(data, time, data_vel,
                                          self.take_dims, demo_name)
        
        # merge segments of length 1   
        num_segments = len(segments)
        s = 0
        
        if data_vel is not None:
            delta = np.concatenate((self.delta, self.delta_vel))
        else:
            delta=self.delta
            
        while s < num_segments:
            segment = segments[s]
            
            if segment.end - segment.start == 1:
                # merge with with previous 
                if s != 0 :
                    segments[s-1].append(segment, delta)
                    segments.remove(segment)
                # if its the first, merge with second
                else:
                    segment.append(segments[s+1], delta)
                
                    segments.remove(segments[s+1])
                    
                num_segments -= 1
            s += 1
            
    
        if self.verbose:
            print(self.inference.write_changepoints())
            
        return changepoints, segments
        
    def plot(self, segments, fig_number=1, demo_name=None):
        """
        Generates subplot visualizing the segments and corresponding segment borders and
        subplot showing the posterior probability of a segment border for each time point. 
        If the data is in Euclidean Space the 3D position of the borders are visualized 
        in an additional figure.
        
        Parameters
        ----------
        segments : list of :class:`.segments.Segment`
            Segmented demonstrations as returned by :meth:`.VMCI.segment`. 
        
        fig_number : int, optional
            Number of figure to plot results of first demonstration in.
            Further demonstrations are plotted in separate figures with
            successive numbers, *fig_number+1, ...*. Defaults ``1``.
            
        demo_names : string, optional
            Name of the demonstration used to generate figure titles. 
            If ``None``, demonstration name is set to *'demo 1'*. Defaults ``None``.
        """
        
        if demo_name is None:
            demo_name = "demo 1" 
        
        T = segments[-1].end
        
        # calculate posterior of change point position and segment length
        number_of_cp_simulations = 1000
        cp_post = self.inference.calculate_changepoint_posterior(T, 
                                                                 number_of_cp_simulations)[0]
        
        title = "vMCI results: " + demo_name
        plot_changepoints_time(fig_number, title, segments,
                               changepoint_posterior=cp_post, 
                               num_simulations=number_of_cp_simulations)
        
        # if data is given in Euclidean coordinates plot 3D
        if np.shape(segments[0].data)[1]/3 == self.num_sensors:
            plot_changepoints_space(fig_number+1, "MCI results " + demo_name,
                                    segments)
          
    def _init_inference(self, N, filter_alg, seed):
        # initialize inference    
        lrm = LRM(self.nu, 
                  self.gamma, 
                  self.delta, 
                  self.basis_func)
                    
        
        vel_lrm = LRM(self.nu_vel, 
                      self.gamma_vel, 
                      self.delta_vel, 
                      self.basis_func_vel)
        
        self.inference = ChangePointInference(lrm, 
                                         self.p, 
                                         N, 
                                         filter_alg, 
                                         vel_lrm=vel_lrm, 
                                         num_sensors=self.num_sensors,
                                         verbose=self.verbose, seed=seed)
        
        return self.inference
    
# ------------- auxillary functions ---------------------------------
    def _read_changepoints(self):
        # returns list of change point positions
        cp_positions_long = self.inference.get_changepoint_indices() 
        cp_positions = []
        for i in range(len(cp_positions_long)-1, -1, -1):
            if len(cp_positions) > 0 and cp_positions[-1] == cp_positions_long[i][0]-1:
                cp_positions[-1] = cp_positions_long[i][0]# merge segments of length 1
            else:
                cp_positions.append(cp_positions_long[i][0])
        return cp_positions
    
    def _read_changepoints_with_apost(self):
        # returns list of [change point position, change point a posterior probability]
        cps_long = self.inference.get_changepoint_indices() 
        cps = []
        for i in range(len(cps_long)-1, -1, -1):
            if len(cps) > 0 and cps[-1] == cps_long[i][0]-1:
                cps[-1] = cps_long[i]# merge segments of length 1
            else:
                cps.append(cps_long[i])
        return cps
        
    
    def _set_takedims(self, taks_space_dim):
        if self.take_dims is None:
            self.take_dims = range(taks_space_dim)
        self.take_dims_vel = [int(i/3.) for i in self.take_dims[0::3]]
    
    def _determine_datat(self, data, t, data_vel = None):
        if data_vel is not None:
            return np.concatenate((np.array(data[t, self.take_dims], ndmin=2), 
                                   np.array(data_vel[t, self.take_dims_vel],
                                                         ndmin=2)), axis=1)
        else:
            return np.array(data[t, self.take_dims], ndmin=2)
    
    def _determine_parameters(self, data, data_vel):
        
        if self.nu is None:
            self.nu = self._calc_nu()
        
        if self.gamma is None:
            self.gamma, self.gamma_vel = self._calc_gamma(data, data_vel)
            
        if self.delta is None:
            self.delta, self.delta_vel = self._calc_delta(data_vel)
            
            
        if self.p is None:
            self.p = self._calc_d(data)
        
        
    def _calc_gamma(self, data, data_vel = None):
        gamma = np.diag(np.diag(np.array(np.cov(np.diff(data[:, self.take_dims].T)), ndmin=2)))
        if np.linalg.det(gamma) == 0:  # would cause log(0) during inference
            gamma = gamma + 1.e-5 * np.eye(np.shape(gamma)[0])
        
        if data_vel is None:
            return gamma, None        
        else:
            gamma_vel = np.diag(np.diag(np.array(np.cov(np.diff(data_vel[:, self.take_dims_vel].T)), ndmin=2)))
            return gamma, gamma_vel
    
    def _calc_delta(self, data_vel = None):
        delta = 10 * np.ones(self.q) # best for AR basis
        
        if data_vel is None:
            return delta, None
        else:
            delta_vel = np.sqrt(np.max(data_vel[:, self.take_dims_vel]) / 2.) * np.ones(2)
            return delta, delta_vel
        
    def _calc_d(self, data):
        return 1. / np.shape(data)[0]# 1/T
    
    def _calc_nu(self):
        return len(self.take_dims)+2
    

