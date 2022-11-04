# -*- coding: utf-8 -*-
"""
Collection of classes to implement the mutiple change-point inference
introduced in :cite:p:`Fearnhead2007` and its extension to the velocity-based Multiple Change-Point 
Inference (vMCI) introduced in :cite:p:`Senger2014`.
   
:Author: Lisa Gutzeit (Lisa.Gutzeit@uni-bremen.de)
:Last update: 2022-06
"""
import numpy as np
from copy import copy

from vmci_segmentation.lrm import SplitBasis, ARBasis
from .segments import Segment

class ChangePointInference:
    """ Main class of Multiple Change-Point Inference proposed in :cite:p:`Fearnhead2007`,
    including the extension to the velocity-based multiple change-point
    inference presented in :cite:p:`Senger2014`.
    
    Attributes
    -----------
    LRM : instance of :class:`.lrm.LRM`
        Linear regression model used to represent the data.
        
    p : double
        Prior-parameter of segment length.
        
    N : int
        Maximum number of particles.
        
    filter_alg : one of ['SOR', M] or ['SCR', alpha]
        Defines the filter algorithm ("SOR" or "SRC") and its parameter 
        (M or alpha).
        
    vel_lrm : instance of :class:`.lrm.LRM`, optional
        Linear regression model used to represent the velocity of the data. If
        ``None`` LRM of the data is also used for its velocity. Defaults ``None``.
        
    num_sensors : int, optional
        Number of markers/sensors in the data. Just needed, if split LRM is used. 
        Defaults ``None``
        
    seed : int, optional
        Seed for numpy.random.RandomState. Defaults ``None``. 
        
    verbose : int, optional
        Verbosity level. Defaults ``1``.    
    """
    def __init__(self, LRM, p, N, filter_alg, vel_lrm=None, num_sensors=None,
                 verbose=1, seed = None):
        self.regression_model = LRM
        self.split = False
        if vel_lrm is not None:
            # split LRM, i.e. approximate velocity independent from the position
            # using single radial basis function called rbfv.
            
            # check if LRM for velocity dimension has correct basis
            if vel_lrm.basis_name != 'rbfv':
                raise RuntimeError("Linear regression model for velocity dimension should have basis 'rbfv'!")
            if not num_sensors:
                raise RuntimeError("num_sensors unknown. Define number of markers to run vMCI!")
            self.vel_regression_model = vel_lrm
            self.split = True
            self.num_sensors = num_sensors
            
        
        self.q_max = self.regression_model.max_model_order
        if self.split:
            self.q_max *= self.vel_regression_model.max_model_order
            
        self.p = p
        self.N = N       
        self.filter_algorithm = filter_alg[0]
        
        # set parameters of the filter algorithm
        if filter_alg[0] == "SOR":
            # filter down to M particles
            self.M = filter_alg[1]
        elif filter_alg[0] == "SRC":
            # filter down with a maximum introduced error of alpha
            self.alpha = filter_alg[1]

        self.max_path = []
        self.particles = []
        
        # additional parameters for the posterior distribution of the change
        # point positions
        self.changepoint_position_prob = []
        self.fit_probabilities = [] # at time t: fit_prob[j,q] = P(j,t,q) (j = 0,...,t-1)
        self.fit_probabilities_prev = [] # at time t-1: fit_prob[j,q] = P(j,t,q) (j = 0,...,t-2)
        
        self.d = None

        self.verbose = verbose
        self.random_state = np.random.RandomState(seed)
        
        
    def evaluate_time_point(self, t, x, y):
        """ Calculate MAP estimates of change-point positions up to time point t. 
        
        The new data point x, y is added to the MAP change-point estimation.
        
        Parameters
        ----------
        t : int
            Time index.  
            
        x : float
            Time at index t.
            
        y : array, shape (1, d)
            Data point at time index t. If the LRM is split. i.e. vMCI is run, 
            the observed data must have the format 
            [3D_pos_marker1, 3D_pose_marker2, ...,vel_marker1, vel_marker2, ...],
            with shape = [1, dim*num_marker] or 
            shape = [1, dim*(num_position_marker + num_velocity_marker)].
        
        """
        
        #### some initializations
        if not self.d:
            self.d = np.shape(y)[1]
            if self.split:
                self.d = self.d - self.num_sensors
                
        self._check_input(y)
        
        # if AR basis, update h_prev
        if self.regression_model.basis_name == "ar":
            self._update_ar_basis(t, y)
        ####
    
        last_cp_position = np.zeros(t)
        
        for p in self.particles:
            # update particle
            p.update(y, x, t)
                        
            # compute fit probabilities for particle
            norm_factor = self._compute_fit_prob(t, p)
            
            # update self.changepoint_positions_prob
            last_cp_position = self._update_cp_prob(t, p, norm_factor, 
                                                    last_cp_position)
            
                    
        # normalize probability distribution of last change point to one
        if t>0:
            if np.sum(last_cp_position) > 0:
                last_cp_position = 1./np.sum(np.sort(last_cp_position)[::-1])*last_cp_position
            else:
                # if probability of a changepoint at position j> 0 is zero for all j
                # set most probable changepoint position to j=0.
                last_cp_position[0] = 1.
        self.changepoint_position_prob.append(last_cp_position)
        
        
        # filter if necessary
        if len(self.particles) > self.N:
            self._filter()

        # determine Viterbi path
        if t==0:
            self.max_path = []
            MAP = np.log(1./self.q_max)
        else:
            max_particle = self._determine_max_particle(self.particles)
            max_particle.path.append([copy(max_particle), copy(max_particle.basis)])
            self.max_path = max_particle.path
            MAP = max_particle.a_post

        # create new particles for a changepoint at time t
        self._create_particles(t, x, y, MAP)
            
        # create self.fit_probabilities matrix
        self.fit_probabilities_prev = self.fit_probabilities
        self.fit_probabilities = np.zeros((t+1, self.q_max))
    
    
        
    ################################################################################
    # output and processing of the results
    ################################################################################
    def _simulate_changepoints(self):
        """ Simulates change-points from filtering density P(C_t | y).
        See :cite:p:`Fearnhead2007` for details.
        
        Returns
        --------
        changepoints : list
            Simulated change-points.
        """
        T = len(self.changepoint_position_prob)
        changepoints = [T-1] # first changepoint at T-1
        k = 0
        
        while changepoints[k] > 0:
            # simulate next change point
            cumprob = np.concatenate((np.array([0]), np.cumsum(self.changepoint_position_prob[changepoints[k]])))
            uni = self.random_state.rand()
            next_cp = (cumprob[(cumprob<uni).nonzero()[0]+1]>=uni).nonzero()[0][0]
            changepoints.append(next_cp)
            k = k+1
        
        return changepoints[::-1]
    
    def calculate_changepoint_posterior(self, T, n_simulations):
        """ Returns estimated posterior probability of change-point positions and 
        posterior of the number of segments.
        
        Samples *n* times from the filtering density P(C_t|y).
        
        Parameters
        ----------
        T : int
            Number of sample points of the data.
        
        n_simulations : int
            Number of simulations.
         
        Returns
        --------
        changepoint_posterior : list, len=T
            Posterior probability for each time step.
        
        posterior_number_segments : list, len=num_simulations
            Number of segments for each simulation.
        """
        changepoint_posterior = np.zeros(T)
        posterior_segment_length = []
        
        for i in range(n_simulations):
            sim_cp = self._simulate_changepoints()
            changepoint_posterior[sim_cp] = changepoint_posterior[sim_cp]+1
            posterior_segment_length.append(len(sim_cp))
        
        return changepoint_posterior/np.double(n_simulations), posterior_segment_length


    def get_changepoint_positions(self):
        """ Returns a list with [changepoint_position, changepoint_probability]. """
        path_copy = copy(self.max_path)

        changepoints = []
        if len(path_copy) > 0:
            particle = path_copy.pop()[0]
            changepoints.append([particle.end_seg_sample, 1])
            changepoints.append([particle.position_sample, np.exp(particle.a_post)])

        while len(path_copy) != 0:
            particle = path_copy.pop()[0]
            # find the particle whose segment end s equal to the last changepoint
            if particle.end_seg_sample == changepoints[-1][0]:
                changepoints.append([particle.position_sample, np.exp(particle.a_post)])

        return changepoints
    
    def get_changepoint_indices(self):
        """ Returns a list with [changepoint_index, changepoint_probability]. """
        path_copy = copy(self.max_path)

        changepoints = []
        if len(path_copy) > 0:
            particle = path_copy.pop()[0]
            changepoints.append([particle.end_seg, 1])
            changepoints.append([particle.position, np.exp(particle.a_post)])

        while len(path_copy) != 0:
            particle = path_copy.pop()[0]
            # find the particle whose segment end s equal to the last changepoint
            if particle.end_seg == changepoints[-1][0]:
                changepoints.append([particle.position, np.exp(particle.a_post)])

        return changepoints
    
    def get_segments(self, data, time, data_vel = None, take_dims = [], 
                     demo_name = []):
        """ Returns a list of segments.
         
        Parameters
        -----------
        data : array, shape (T, d)
            Observed data.
        
        time : array, shape (T,)
            Time
            
        data_vel : array, shape (T, num_sensors), optional
            Velocity of observed data. Defaults ``None``.
    
        take_dims : list of int, optional
            Dimensions which were considered for inference. Are saved as an
            attribute of the segments. Defaults ``None``.
            
        demo_name : string
            Name of the demonstration/experiment. Is saved as an attribute
            of the segments. Defaults ``None``.
        
        Returns
        -------
        segments : list of instances of :class:`.segments.Segment`
            Result of inference as a list of segments.
        """
        path = copy(self.max_path)
        segments = []
        [particle,basis] = path.pop()
        
        vel = []
        alg = 'MCI'
        if data_vel is not None:
            vel = data_vel[particle.position:particle.end_seg+1, :]
            alg = 'vMCI'
        
        seg = Segment(data[particle.position:particle.end_seg+1, :], # +1 because this is the last segment, i.e. last point has to be included
                      time[particle.position:particle.end_seg+1],
                      particle.position, particle.end_seg+1,
                      basis,
                      particle.beta,
                      alg, # segmentation alg.
                      '', # clustering alg.
                      data_vel = vel,
                      take_dims = take_dims)
        seg.demonstration_name = demo_name
        segments.append(seg)

        while len(path) != 0:
            [particle,basis] = path.pop()
            
            # find the particle whose segment end is equal to the last changepoint
            if particle.end_seg_sample == segments[-1].t[0]:
                if data_vel is not None:
                    vel = data_vel[particle.position:particle.end_seg, :]
                    
                
                seg = Segment(data[particle.position:particle.end_seg, :],
                                  time[particle.position:particle.end_seg],
                                  particle.position, particle.end_seg,
                                  basis,
                                  particle.beta,
                                  alg, # segmentation alg.
                                  '', # clustering alg.
                                  data_vel = vel,
                                  take_dims = take_dims)
                seg.demonstration_name = demo_name
                segments.append(seg)
        
        segments.reverse()
        return segments
        
    def write_changepoints(self):
        """
        Returns string with detected change-point positions and indices.
        """
        s = ''
        # write results
        s += "\ndetected changepoint positions (indices):\n"
        cp = self.get_changepoint_positions()
        cp_ind = self.get_changepoint_indices()
        for i in range(len(cp)-1, 0, -1):
            s += "%.2f (%i), " %(cp[i][0], cp_ind[i][0])
        s += "%.2f (%i)\n\n" %(cp[0][0], cp_ind[0][0])
        
        return s
    
    ################################################################################
    # auxillary functions (for inference)
    ################################################################################
    
    def _g(self, l):
        """ Probability mass function of the segment length l. """
        # assumed to be binomial
        return self.p*(1-self.p)**(l-1)

    def _G(self, l):
        """ Density function of the segment length l. """
        # assumed to be geometric
        return 1-(1-self.p)**l

    def _model_prior(self, q):
        """ Uniform model prior. """
        return 1./self.regression_model.max_model_order 

    def _determine_max_particle(self, particles):
        """ Returns particle with maximum posterior probability a_post. """
        max_particle = particles[0]
        for p in particles:
            if p.a_post > max_particle.a_post:
                max_particle = p

        return max_particle

    def _determine_alpha(self, weights):
        """ Calculates alpha for SOR algorithm, see :cite:p:`Fearnhead1998`
        for details. """
        M = self.M
        k_old = - 1
        k = 0

        # sort weights (descending)
        weights = np.sort(weights)[::-1]

        while k!=k_old:
            k_old = k
            c = (M-k_old)/np.sum(weights[k_old::])
            cw = c*weights[k_old::]
            k = k_old + np.sum(cw>1)
        return 1./c

    def _filter(self):
        """ Filters particles with SOR or SRC algorithm. See :cite:p:`Fearnhead2007`
        for details.
        
        Particles with a small a posteriori probability are sorted out. The 
        others are resampled. During this step a posteriori probabilities
        of each particle are adapted (normalized) as well as the MAP probabilities.
        
        SOR: stratified optimal resampling
        SRC: stratified rejection control
        """
        particles = self.particles
        resampled_particles = []
        rest_particles = []
         
        # get all weights of the particles (needed for normalization)
        weights = np.zeros(len(particles))
        for i in range(len(particles)):
            p = particles[i]
            weights[i] = p.a_post
            if weights[i] != weights[i]:
                weights[i] = -np.Inf # give zero probability if it is NaN
        
        # Normalization:        
        # If the normalization factor is zero (or close to), the a posteriori 
        # probabilities are so small that is comes to stability problems. To 
        # solve this problem, the normalization is done in two steps:
        
        # 1. step: pre-normalization (in log to prevent inaccuracies due
        # to stability problems):
        norm_factor1 = np.max(weights)
        weights = weights - norm_factor1
        
        # 2. step: regular normalization:
        normalization_factor = np.sum(np.exp(weights)) # should not be zero any more
        weights = np.exp(weights)/normalization_factor
        
        # combine factors to one normalization factor (in log)
        normalization_factor = np.log(normalization_factor) + norm_factor1

                        
        if self.filter_algorithm == "SOR":
            # SOR1: calculate alpha
            alpha = self._determine_alpha(weights)
        elif self.filter_algorithm == "SRC":
            alpha = self.alpha
    
        # SOR2/SRC1
        for i in range(len(particles)):
            p = particles[i]
            # change original weight to normalized one
            p.a_post = weights[i]
            # normalize MAP value
            p.MAP = p.MAP - normalization_factor
            
            # check if particle is kept
            if (p.a_post >= alpha):
                p.a_post = np.log(p.a_post)
                resampled_particles.append(p)
            else:
                rest_particles.append(p)
            
        # SOR3/SRC2 (stratified resampling algorithm of Carpenter et al.)
        u = alpha * self.random_state.rand(1)
    
        for i in range(len(rest_particles)):
            p = rest_particles[i]
            if (p.a_post >= alpha):
                p.a_post = np.log(p.a_post)
                resampled_particles.append(p)
            else:
                u = u - p.a_post
                if u <= 0:
                    p.a_post = np.log(alpha)
                    resampled_particles.append(p)
                    u = u + alpha     
                     
        self.particles = resampled_particles
    
    def _check_input(self, y_obs):
        # check if input is correct        
        if type(y_obs) == list:
            raise RuntimeError("Observed data should be a 2D array (nD).")
        
        if y_obs.ndim != 2 or np.shape(y_obs)[0] != 1:
            raise RuntimeError("Observed data should be a 2D array, shape= [1, task_space_dim]")

        # check size of gamma if d>1
        if self.d > 1 and (self.d != np.shape(self.regression_model.gamma)[0] or \
            self.d != np.shape(self.regression_model.gamma)[1]):
            raise RuntimeError("Size of gamma not adjusted to size of observed data. Should be of size (task_space_dim, task_space_dim)")
        
        # check data type of y_observed
        if y_obs.dtype < 'float32':
            raise RuntimeError("Type of observed data has to be at least float32.")
    
    def _update_ar_basis(self, t, y_obs):
        if self.split:
            y = y_obs[:, 0:-self.num_sensors]
        else:
            y = y_obs
            
        h_prev = self.regression_model.h_prev
        order = self.regression_model.max_model_order
        for it in range(len(list(np.array(y).flatten()))-1,-1,-1):
            h_prev.insert(0, list(np.array(y).flatten())[it])
        if t==0 and order == 1:
            # set last part of h_prev(which is used for the first basis)
            # to y_0 
            h_prev[self.regression_model.d:(order+1)*self.regression_model.d] = h_prev[0:self.regression_model.d]
        h_prev = h_prev[0:(order+1)*self.regression_model.d]
        self.regression_model.h_prev = h_prev
        
    def _compute_fit_prob(self, t, p):
        p.calc_fit_prob()
        j = p.position
        
        P_tjq = np.log(1-self._G(t-j-1)) + p.fit_prob \
                + np.log((1.)/self.q_max) + p.MAP 
        p.a_post = (P_tjq + np.log(self._g(t-j))) \
                    - np.log((1-self._G(t-j-1)))
        
        if np.isnan(p.a_post):
            # can occur if probability of segment length is (close to) zero
            p.a_post = -np.inf
            
        # update self.fit_probabilities
        # normalization factor if fit probabilities are very big (long segments)
        # resulting probability of last change-point position stays unnormalized
        factor=1
        if np.isinf(np.exp(p.fit_prob)):
            factor=200
            if p.fit_prob > 800:
                factor = factor + (p.fit_prob-800)
            self.fit_probabilities[j, :] = self.fit_probabilities[j,:]/factor
            self.fit_probabilities[j, p.q-1] = np.exp(p.fit_prob-factor)
        else:
            self.fit_probabilities[j, p.q-1] = np.exp(p.fit_prob)
            
        return factor
    
    def _update_cp_prob(self, t, p, norm_factor, last_cp_position):
        j = p.position
        if t==1:
            last_cp_position[j] = 1
        else:                
            if np.isinf(np.sum(self.fit_probabilities[j,:])):
                norm_factor += 100
                self.fit_probabilities[j,:] /= norm_factor
            if j==t-1:
                weight = np.sum(self.fit_probabilities[j,:])
                last_cp_position[j] = weight*norm_factor * np.sum(self.p*self.changepoint_position_prob[t-1][0:t-1])
            else:
                # if probabilities are too small (avoid division by zero)
                if np.sum(self.fit_probabilities[j,:]) == 0:
                    weight = 0
                elif np.sum(self.fit_probabilities_prev[j,:]/norm_factor) == 0:
                    weight = np.sum(self.fit_probabilities[j,:])
                else:
                    weight = np.sum(self.fit_probabilities[j,:])/np.sum(self.fit_probabilities_prev[j,:]/norm_factor)
                last_cp_position[j] = weight * (1 - self.p)*self.changepoint_position_prob[t-1][j]

        self.fit_probabilities *= norm_factor # re-do normalization
        
        return last_cp_position
    
    def _create_particles(self, t, x, y_obs, MAP):
        for q in range(1, self.q_max+1):
            # Note if the LRM is split,i.e. vMCI is used, the MAP values of the individual
            # particles (for each LRM) belong to the overall LRM!
            if self.split:
                q2 = int((q-1) % (self.vel_regression_model.max_model_order) + 1)
                q1 = int((q-q2)/self.vel_regression_model.max_model_order + 1)
                
                particle1 = Particle(self.regression_model, t, q1, 
                                    self.max_path, MAP, x, y_obs,
                                    relevant_dims=range(self.d))
                
                particle2 = Particle(self.vel_regression_model, t, q2,
                                    self.max_path, MAP, x, y_obs,
                                    relevant_dims= range(self.d, self.d+self.num_sensors))
                
                new_particle = MergedParticle(self.vel_regression_model, t, q, 
                                              self.max_path, MAP, x, y_obs,
                                              particle1, particle2)
            else:
                new_particle = Particle(self.regression_model, t, q, 
                                        self.max_path, MAP, x, y_obs)
            self.particles.append(new_particle)

class Particle:
    """ Particles for multiple change-point inference. 
    
    A particle represents a segment approximated by a certain linear regression model
    with a fixed model order. Using :meth:`.update`, the segment can be extended
    by a new incoming data point and the a posteriori and maximum a posteriori 
    probabilities are updated (during this the model parameters :math:`\\beta`
    are also updated).
    
    Attributes
    -----------
    LRM : instance of :class:`.lrm.LRM`
        Linear regression model used to represent this particle/segment.
    
    position : int
        Starting index ``j`` of the segment represented by the particle.
    
    q : int
        Defines used basis model of the segment, e.g. number of basis functions 
        in case of AR-basis or number of centers in case of velocity Basis (RBFv).
    
    path : list of instances of class:`.inference.Particle`
        Viterbi path to this particle.
    
    MAP : float
        Maximum a posteriori probability of a change-point occurring at ``j``
        :math:`MAP \\mathrel{\\widehat{=}} P_j^{MAP}` 
    
    x_j : float
        Time at starting position/index ``j``.
    
    y_j : array, shape (1, d)
        Data at starting position/index ``j``.
    
    relevant_dims: list of int, optional
        Defines the task space dimensions this particle refers to. If ``None`` 
        all dimensions are taken. Defaults ``None``.
        
    beta : array, shape (q, d) 
        Weights to model the data using the linear regression model with the
        defined basis functions.
    
    fit_prob : double
        Fit probability P(j,t,q) of the segment defined by the particle.
    
    """
    
    def __init__(self, LRM, position, q, path, MAP, x_j, y_j, relevant_dims = None,
                 incr=False):
        self.LRM = LRM  
        self.relevant_dims = relevant_dims   
        
        # If vMCI is used and this particle represents the velocity segment,
        # just the last task space dimension (which should be the velocity) is treated
        if relevant_dims:
            y_j = np.array(y_j[:, relevant_dims], ndmin=2)
        
        # task space dimension
        self.d = np.shape(y_j)[1]

        self.position = position
        self.q = q
        self.path = path
        self.MAP = MAP
        
        # a posteriori probability of a change point at time j using model q
        # given the data until current time point j, i.e.  :math:`a\_post=P_t(j,q)g(t-j)/(1-G(t-j-1))`
        self.a_post = 0
        
        # model parameters
        self.beta = []
        
        # end index of segment
        self.end_seg = position
        
        # length of segment
        self.l = 0
        
        # start and end values (x_j,x_t) of the segment defined by the particle
        self.position_sample = x_j
        self.end_seg_sample = x_j
        
        # initialize fit probability to zero
        self.fit_prob = -np.inf 
        
        ##################################################################
        # INIT basis
        ##################################################################
        if not incr:
            #create instance for the basis function of the segment defined by the particle
            self.basis = LRM.create_basis_for_segment(q)
            
            # segment data: # x: hole sample point vector from time j up to time t, same for y
            self._x = [x_j]
            self._y = y_j
            
        ##################################################################
        # INIT sufficient statistics and model hyper-parameters
        ##################################################################
        M = self.basis.num_basis_func
        
        # sufficient statistics needed to compute P(j,t,q) (updated every time a
        # new data point comes in)
        self._A = np.zeros((M, M))
        self._b = np.zeros((M, 1))
        self._yy = []
        
        # precompute some values
        self._D_inv = self.LRM.D_inv[0:M, 0:M]
        self._log_det_D = np.log(np.linalg.det(self.LRM.D[0:M, 0:M]))
        
        # if the basis is AR, D is larger:
        if self.LRM.basis_name == "ar":
            self._D_inv = self.LRM.D_inv[0:(self.LRM.d*M), 0:(self.LRM.d*M)]
            self._log_det_D = np.log(np.linalg.det(self.LRM.D[0:(self.LRM.d*M), 0:(self.LRM.d*M)]))
               

    def update(self, y_t, x_t, t, incremental=False):
        """ Updates sufficient statistics with new data point.
        
        Calculates the sufficient statistics A,b for a new data point y_t,x_t.
        (See :cite:p:`Konidaris2012` for a more detailed explanation of how the calculation of
        the sufficient statistics is accelerated.)
                
        Parameters
        -----------
        y_t : array, shape (1, d)
            Data point at time index t.
            
        x_t : float
            Time at index t.
        
        t : int
            Time index.   
        
        incremental : boolean
            Defines if used basis function is updated incrementally. 
            Defaults ``False``.
        """
        self.end_seg = t
        self.end_seg_sample = x_t
        self.l = self.l + 1
        
        if self.relevant_dims:
            y_t = np.array(y_t[:, self.relevant_dims], ndmin=2)
        
        
        if not incremental:
            # update sufficient statistics for previous t (this has to be done because
            # update function is called before calculating a_post)
            self.basis.update(self._x[-1], self._y[-1], t-1-self.position)
            if type(self._yy) != np.ndarray:
                self._yy = np.outer(self._y[-1], self._y[-1])
            else:
                self._yy = self._yy + np.outer(self._y[-1], self._y[-1])
            
            H = self.basis.evaluate(self._x, t-1-self.position)
            self._A = np.dot(H.transpose(), H)
            self._b = np.dot(H.transpose(), self._y)
            
            # update particle parameters
            self._x.append(x_t)
            self._y = np.concatenate((self._y,y_t))
        
            # check if overflow encountered
            if np.sum(np.isinf(self._yy)):
                raise RuntimeError("Overflow encountered in calculation of yy^T. Values in y too large!")


    def calc_fit_prob(self):
        """ Calculates fit probability P(j,t,q) of the segment defined by the particle.
        Additionally the model parameters beta of the particle are updated.           
        """
        M = np.linalg.inv(self._A+self._D_inv) # numerical instabilities if entries in D are too large
        yPy = self._yy - np.dot(self._b.transpose(), np.dot(M,self._b))
        
        if np.sum(np.linalg.eigvals(yPy)) < 0:
            # Possible solution: more bits for data
            raise RuntimeError("yPy not positive definite! Probably caused by numerical instabilities.")
           
        self._weights_posterior(M)
        evidence = self._model_evidence(M, yPy)
        
        # If first weight (alpha_1 in :cite:p:`Senger2014`) of RBFv basis is negative,
        # the data representation is no longer bell-shaped but turned around. 
        # Thus, the probability for this approximation is set to zero
        if self.LRM.basis_name == 'rbfv' and np.sum(self.beta[0,:]< 0) > 0:
            self.fit_prob = -np.inf
        else:
            self.fit_prob = evidence
    
    def _model_evidence(self, M, yPy):
        if np.linalg.det(yPy + self.LRM.gamma) < 0:
            raise RuntimeError("Logarithm of negative number. Maybe gamma too small.")
        
        return (self.LRM.nu/2.)*np.log(np.linalg.det(self.LRM.gamma)) \
               - ((self.l+self.LRM.nu)/2.)*np.log(np.linalg.det(yPy + self.LRM.gamma)) \
               + (self.d/2.)*(np.log(np.linalg.det(M))-self._log_det_D) \
               - ((self.l*self.d)/2.)*np.log(np.pi) + self._sum_gamma1() \
               - self._sum_gamma2()
    
    def _sum_gamma1(self):
        sum1 = 0
        for i in range(1, self.d+1):
            try:
                sum1 += np.math.lgamma((self.l + self.LRM.nu + 1 - i)/2.)
            # catch exception that occurs with long segments
            except OverflowError:
                print("Cannot calculate gamma(" + str((self.l+self.LRM.nu)/2.-i) + "), or " \
                      + str(int((self.l+self.LRM.nu-i)/2.)-1) + \
                      "! -> segment of this particle contains too many elements! Try again with down-sampled data.")
                raise
        return sum1
    
    def _sum_gamma2(self):
        return np.sum([np.math.lgamma((self.LRM.nu + 1 - i)/2.) for i in range(1, self.d+1)])
        
            
    def _weights_posterior(self, M):
        # estimation of the model parameters beta with the posterior p(beta|y)
        # beta = (A +D^(-1))^(-1)*b = M*b (without the use of variance priors)
        self.beta = np.dot(M, self._b)
        

    
class MergedParticle(Particle):
    """ Particle to represent a segment with different basis functions
    for different segments. (Needed in vMCI to model velocity with 
    independent from the position.)
    
    Attributes
    -----------
    LRM : instance of :class:`.lrm.LRM`
        Linear regression model used to represent this particle/segment.
    
    position : int
        Starting index ``j`` of the segment represented by the particle.
    
    q : int
        Defines used basis model of the segment, e.g. number of basis functions 
        in case of AR-basis or number of centers in case of velocity Basis (RBFv).
        Sum of model orders of individual particles.
    
    path : list of instances of class:`.inference.Particle`
        Viterbi path to this particle.
    
    MAP : float
        Maximum a posteriori probability of a change-point occurring at ``j``
        :math:`MAP \\mathrel{\\widehat{=}} P_j^{MAP}` 
    
    x_j : float
        Time at starting position/index ``j``.
    
    y_j : array, shape (1, d)
        Data at starting position/index ``j``.
    
    particle1 : instance of class:`.inference.Particle`
        Particle to represent first part of the data.
        
    particle2 : instance of class:`.inference.Particle`
        Particle to represent second part of the data.
        
    beta : array, shape (q, d) 
        Weights to model the data using the linear regression model with the
        defined basis functions.
    
    fit_prob : double
        Fit probability P(j,t,q) of the segment defined by the particle.
    
    Methods
    -------
    update(y_t, x_t, t)
        Updates sufficient statistics with new data point.
        
    calc_fit_prob()
        Calculates fit probability P(j,t,q) of the segment defined by the 
        particle.
    
    """
    
    def __init__(self, LRM, position, q, path, MAP, x_j, y_j, particle1, particle2):
        Particle.__init__(self, LRM, position, q, path, MAP, x_j, y_j)
        self.particle1 = particle1
        self.particle2 = particle2
        self.basis = SplitBasis(copy(particle1.basis), copy(particle2.basis))
        
    def update(self, y_t, x_t, t):
        self.end_seg = t
        self.end_seg_sample = x_t
        self.l = self.l + 1
        
        self.particle1.update(y_t, x_t, t)
        self.particle2.update(y_t, x_t, t)
        self.basis = SplitBasis(copy(self.particle1.basis), copy(self.particle2.basis))
        
    def calc_fit_prob(self):
        self.particle1.calc_fit_prob()
        self.particle2.calc_fit_prob()
        self.fit_prob = self.particle1.fit_prob + self.particle2.fit_prob
        
        beta = np.zeros((self.particle1.basis.num_basis_func + self.particle2.basis.num_basis_func, self.d))
        if isinstance(self.particle1.basis, ARBasis):
            beta = np.zeros((self.particle1.d*self.particle1.basis.num_basis_func + self.particle2.basis.num_basis_func, self.d))
        beta[0:np.shape(self.particle1.beta)[0], 0:np.shape(self.particle1.beta)[1]] = self.particle1.beta
        beta[np.shape(self.particle1.beta)[0]::, np.shape(self.particle1.beta)[1]::] = self.particle2.beta
        self.beta = beta

