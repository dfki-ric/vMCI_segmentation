# -*- coding: utf-8 -*-
""" Different classes representing the linear regression models needed in the 
multiple change-point inference algorithm.

The details of the data representation are explained in :class:`MCI.linear_regression_model.LinearRegressionModel`.


**Classes** \n
Main classes:
 :class:`MCI.linear_regression_model.LinearRegressionModel` \n
 :class:`MCI.linear_regression_model.Basis`

Classes for different bases:
 :class:`MCI.linear_regression_model.ARBasis` \n
 :class:`MCI.linear_regression_model.RBFBasis` \n
 :class:`MCI.linear_regression_model.RBFBasisVel` \n
 :class:`MCI.linear_regression_model.SplitBasis` \n

:Author: Lisa Gutzeit (Lisa.Gutzeit@uni-bremen.de)
:Last update: 2022-06
"""
import numpy as np

class Basis:
    """ Main class for all bases of a linear regression model.
    
    Attributes
    ----------
    
    num_basis_func : int
        Number of basis functions used to represent the data.
    """
    def __init__(self, num_basis_func):
        self.num_basis_func = num_basis_func
    
    def update(self, x_t, y_t, t):
        """
        Updates basis with new incoming data at time ``t``. To be implemented
        in subclasses.
        
        Parameters
        ----------
        x_t : float
            Time at index t.
            
        y_t : array, shape (d, )
            Data at time *t* of dimension *d*.
            
        t : int
            Time index.
        """
        raise NotImplementedError()
    
    def evaluate(self, x, t):
        """
        Evaluates basis function at time ``x``. To be implemented
        in subclasses.
        
        Parameters
        ----------
        x : array, shape (T, )
            Time component of segment of length *T*.
            
        t : int
            Time index of end point of the segment.
            
        Returns
        -------
        H : array, shape (T, num_basis_funct) or shape (T, d*num_basis_funct)
            Matrix of basis functions for sample points ``x``.
            Shape differs for different basis functions.
        """
        raise NotImplementedError()
    
class ARBasis(Basis):
    r""" Autoregressive basis for linear regression model (LRM).
    
    With a LRM with AR basis, the current data point :math:`y_t` is approximated 
    with a weighted sum of the previous time points :math:`(y_{t-1}, ... y_{t-r})`.
    :math:`r` defines the order of the model/basis.
    
    The matrix of basis function (*design matrix*) is of shape 
    (segment_length) :math:`\times` (r :math:`\cdot` dim)
    and is defined as:
    
    .. math::
    
        H_{s:t} = \begin{pmatrix}
                    y_s & y_{s-1} & \cdots & y_{s-r} \\
                    \vdots & \vdots & \ddots & \vdots \\
                    y_t & y_{t-1} & \cdots & y_{t-r}
                    \end{pmatrix} 
                  = \begin{pmatrix}
                    y_{s,1} & \cdots & y_{s,d} & \cdots & y_{s-r,1} & \cdots &  y_{s-r,d} \\
                    \vdots & & & \ddots & & & \vdots \\
                    y_{t,1} & \cdots & y_{t,d} & \cdots & y_{t-r,1} & \cdots &  y_{t-r,d}
                    \end{pmatrix} 
    
    .. note:: With this basis the matrix with model weights :math:`\beta` gets
              size *r*d x d* (instead of *r x d*). 
              It is assumed that the data has shape *(segment_length, dim)*
              
    Attributes
    ----------
    r : int
        Model order, i.e. number of previous data points to be used to model 
        the current time point.
    d : int
        Dimension of the data.
    h_prev : array, shape(r,), optional
        Initialization of the design matrix H. *h_prev = [y_{t}, ..., y_{t-r}]*,
        where *t* is the start index of the sequence. If ``None``, H is initialized
        with zeros. Defaults ``None``.
    
    """
    def __init__(self, r, d, h_prev = None):
        Basis.__init__(self, r)
        self._r = r
        self._d = d
        self._design_matrix = np.zeros((1, r*d)) # t=0: H[0, :]
        # the design matrix is growing with each call of ARBasis.update
        
        # init first row with previous values y_{t-1}, ..., y_{t-r}
        if h_prev is not None:
            self._design_matrix[0, :] = h_prev[d::]
        
    def _set_design_matrix(self, X):
        """ Set design matrix to X (shape = [dim, segment_len])."""
        self._design_matrix = X
        
    def update(self, x_t, y_t, t):
        """ Updates design matrix for time point t+1, i.e. calculates H[t+1, :]. 
        
        Parameters
        ----------
        x_t : float
            Time at index t.
            
        y_t : array, shape (d, )
            Data at time *t* of dimension *d*.
            
        t : int
            Time index.
        """
        # Shape of design matrix is (num_seen_time_points+1)x(r*task_space_dim)
        if t+1 != np.shape(self._design_matrix)[0]:
            raise RuntimeError("Wrong AR design matrix. Missed previous call of ARBasis.update()")
        
        new_row = list(np.array(y_t, ndmin=1))
        new_row.extend(self._design_matrix[-1, 0:-1*self._d])
        new_row = np.array(new_row, ndmin=2) # H[t+1, :]
        
        self._design_matrix = np.concatenate((self._design_matrix, new_row), axis=0)
    
    def evaluate(self, x, t):
        """
        Evaluates autoregressive basis at time ``x``. 
        
        Parameters
        ----------
        x : array, shape (T, )
            Time component of segment of length *T*.
            
        t : int
            Time index of end point of the segment.
            
        Returns
        -------
        H : array, shape  (T, d*num_basis_funct)
            Matrix of basis functions for sample points ``x``.
            Shape differs for different basis functions.
        """
        return np.array(self._design_matrix[0:t+1, 0:self._d*(self.num_basis_func)], ndmin=2)

    
class RBFBasisVel(Basis):
    """
    Velocity basis as introduced in :cite:p:`Senger2014`.
    
    Basis consists of a single radial basis function (RBF) with width ``r`` and
    center position ``center`` which is one of several possible center positions
    uniformly distributed over the segment and a constant part which is set to
    one.
    
    Parameters
    ----------
    r : double
        Width of the RBF.
    
    num_centers : int
        Number of possible center positions. Center positions are
        uniformly distributed over the segment.
    
    center : int
        Index of the center used for the basis out of the possible center
        positions.
    """
    def __init__(self, r, num_centers, center): # num_centers gives number of possible center positions
        Basis.__init__(self, 2)
        self.calc_r = False
        if r==0:
            # r has to be calculated
            self.calc_r = True
        self.r = r
         
        self.num_centers = num_centers
        self.center = center
        
        # parameters to define RBF centers
        self._segment_start = None 
        self._segment_end = None
    
    def update(self, x_t, y_t, t):
        """ Updates the end point of the segment used to create RBF-centers.
        
        End of the segment is set to x_t.
        
        Parameters
        ----------
        x_t : float
            Time at index t.
            
        y_t : array, shape (d, )
            Data at time *t* of dimension *d*.
            
        t : int
            Time index.
        """
        if self._segment_start is None:
            self._segment_start = x_t
                        
        self._segment_end = x_t
    
    def evaluate(self, x, t):
        """
        Evaluates basis function at time ``x``. 
        
        Parameters
        ----------
        x : array, shape (T, )
            Time component of segment of length *T*.
            
        t : int
            Time index of end point of the segment.
            
        Returns
        -------
        H : array, shape (T, 2) 
            Matrix of basis functions for sample points ``x``.
        """
        H = np.zeros((np.size(x), 2))
        
        center_dist = (self._segment_end - self._segment_start)/(1.*self.num_centers + 1)
        c = self.center
        # choose middle of the sequence as center for RBF1
        center = self._segment_start + c*center_dist
        center_middle = self._segment_start + (self._segment_end - self._segment_start)/2.
        
        if self.calc_r:
            # chose width automatically
            r = (self._segment_end - center_middle)/2.
            if r==0: 
            # only appears if segment length is 1 -> r is set to 1 to prevent
            # devision by zero
                r = 1
                
            self.r = r
        else:
            r = self.r

        H[:, 0] = np.exp(-((center-x)**2)/(r**2)) # RBF1 part
        H[:, 1] = np.ones(np.size(x)) # constant part
 
        return H

class SplitBasis(Basis):
    """ Class to represent a basis of a split linear regression model.
    
    In a split LRM the dimensions of the input data is split into two
    parts which are modeled with different basis functions. 
    
    Attributes
    ----------
    basis1 : Instance of subclass of :class:`vMCI_segmentation.lrm.Basis`
        Basis applied to the first part of the data (e.g. position).
        
    basis2 : Instance of subclass of :class:`vMCI_segmentation.lrm.Basis`
        Basis applied to the second part of the data (velocity).
    
    
    Methods
    -------
    update(x_t, y_t, t)
        Updates each basis with new incoming data at time ``t``. If data is split
        into position and velocity, ``y_t`` should only contain position data.
        
    evaluate(x, t)
        Evaluates each basis function at time ``x`` and concatenates results.
    """
    def __init__(self, basis1, basis2):
        self.basis1 = basis1
        self.basis2 = basis2
        Basis.__init__(self, basis1.num_basis_func+basis2.num_basis_func)
                
    def evaluate(self, x_t, t):
        H1 = self.basis1.evaluate(x_t, t)
        H2 = self.basis2.evaluate(x_t, t)
        
        H = np.concatenate((H1, H2), axis=1)
        return H
    
    def update(self, x_t, y_t, t):
        self.basis1.update(x_t, y_t, t)
        self.basis2.update(x_t, y_t, t)
    
    
class LRM:
    r""" Linear Regression Model for Multiple Change Point Inference.
    
    The data Y of shape n x d is represented with a linear regression model of 
    order q with basis functions :math:`\phi = (\phi_1, ..., \phi_q)`:
    
    .. math::
        &Y = \sum_{k=1}^q \beta_k \phi_k + \varepsilon \\
        & = H \cdot \beta + \varepsilon
        
    with:
        
    .. math::
    
        &\beta \sim \mathcal{MN}(m , D, \Sigma) \quad \text{model weights, shape: }q \times d \\            
        &\varepsilon \sim \mathcal{MN}(0, I_n, \Sigma) \quad \text{noise, shape: } n \times d \\
        &\Sigma \sim \mathcal{IW}(\nu, S) \quad \text{variance along columns, shape: }d \times d
        
    and:
        
    .. math::
    
        &\nu \quad \text{ Inverse-Wishart hyper-prior, scalar} \\
        &S \quad \text{ Inverse-Wishart hyper-prior, shape: } d \times d \\
        &m \quad \text{ weights mean, shape: } q \times d \\
        &D \quad \text{ weights variance along rows, shape: } q \times d
            
            
    
    Possible basis functions:
        'rbfv' - special radial basis for the absolute velocity, see :cite:p:`Senger2014` 
                (:class:`vmci_segmentation.lrm.RBFBasisVel`)

        'ar' - autoregressive basis (:class:`vmci_segmentation.lrm.ARBasis`)
    
    Attributes
    ----------
    m : int or array, shape (num_basis_funct, d)
        Mean of weights. If set to ``0``, zero mean is assumed.
        
    nu : double
        Prior, Inverse-Wishart noise prior parameter.
        
    gamma : array, shape (d, d), optional
        Prior, Inverse-Wishart noise prior parameter. 
        
    delta : array, shape (num_basis_func, ) optional
        Prior, variance of model parameters along columns. 
        
    basis_func : One of ['ar', q] or ['rbfv', c]
        Defines basis function and its order.
    
    basis : Instance of subclass of :class:`vmci_segmentation.lrm.Basis`
        Basis of LRM instance
        
    """
    
    def __init__(self, nu, S, delta, basis_function):
        self.nu = nu
        self.gamma = np.array(S, ndmin=1)
        self.delta = delta
        self.basis_name = basis_function[0]

        # get number of basis functions
        if np.size(basis_function) > 1:
            self.num_basis_func = basis_function[1]
            if basis_function[0] == "rbfv":
                self.num_centers = basis_function[1]
        else:
            self.num_basis_func = 3
        
        # fixed number of basis function for RBF-velocity
        if basis_function[0] == "rbfv":
            self.num_basis_func = 2
            self.r = 0
        
        if np.size(self.delta) != self.num_basis_func:
            print(np.size(self.delta))
            print(self.num_basis_func)
            if basis_function[0] == "ar":
                raise ValueError("Hyperprior D is not adjusted to the number of basis functions! If AR basis is chosen the size of delta  has to be (q, ).")
            else:
                raise ValueError("Hyperprior D is not adjusted to the number of basis functions!")
        
        # init number of different possible models
        self.max_model_order = self.num_basis_func
        if basis_function[0] == "rbfv":
            self.max_model_order = self.num_centers
             
        # pre-compute some values
        self.D = np.diag(np.multiply(self.delta[0:self.num_basis_func], self.delta[0:self.num_basis_func])) 
        if self.basis_name == "ar":
            self.d = np.shape(self.gamma)[0]
            self.D = np.kron(self.D, np.eye(self.d))
            self.h_prev = [0]*self.d*self.max_model_order
            
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._nu_log_gamma = (self.nu/2.) * np.log(self.gamma)
        
        if np.linalg.det(self.D) != 0: 
            self.D_inv = np.linalg.inv(self.D)
        else:
            raise ValueError("Hyperprior D is not invertible (singular matrix). Choose different delta. (D = diag(delta^2))")
        self._log_gamma_nu = np.log(np.math.gamma(self.nu/2.))

        
    def create_basis_for_segment(self, q):
        """ Creates instance of basis function of the type that is defined
        for this linear regression model.
        
        Parameters
        ----------
        q : int
            Order of basis.
        """
        if self.basis_name == 'ar':
            basis = ARBasis(q, self.d, self.h_prev)
        elif self.basis_name == 'rbfv':
            basis = RBFBasisVel(self.r, self.num_centers, q)
        else:
            raise RuntimeError("Basis -"+self.basis_name+"- not known")
        return basis        