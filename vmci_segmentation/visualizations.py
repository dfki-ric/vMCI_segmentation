'''
:Author: Lisa Gutzeit (lisa.gutzeit@uni-bremen.de)
:Last update: 2022-06
'''
import pylab as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
        
def plot_changepoints_time(fig, title, segments,
                          changepoint_posterior = None,
                          num_simulations = 0,
                          figsize = (12, 10)):
    """
    Plots results of the movement segmentation in 1D. Two figures are created:
    
    Figure 1 (fig):
    Plot of the observed data and the MAP change-point positions. If the 
    posterior probabilities of change-point positions are given using the
    parameter `changepoin_posterior`, they are plotted in a subplot in this 
    figure.
    
    Figure 2 (fig+1):
    Shows the histogram of the number of change points. Just plotted if
    if `changepoint_posterior` is given.
          
    Parameters
    ----------
    fig : int
        Defines number of first figure.
    
    title : string
        Title of figure.
        
    segments : list of instances of :class:`.segments.Segment`
        Segmented demonstrations as list of length *num_segments*, e.g., 
        elements of the list returned by :meth:`.VMCI.segment`. 
    
    changepoint_posterior : array, shape (T,), optional
        Posterior of change-point positions. Defaults ``None``.
    
    num_simulations : int, optional
        Number of simulations used to determine the change-point posterior.
        Defaults ``None``.
    
    figsize : tuple, optional
        Size of first figure. Defaults ``(12, 10)``.
    """       
    if changepoint_posterior is not None:
        pl.figure(fig, figsize, dpi=80)
        pl.clf()
        pl.subplot(211)
    else:
        pl.figure(fig, figsize=(12, 5), dpi=80)
        pl.clf()
        
    
    pl.title(title)
    pl.xlim((segments[0].t[0], segments[-1].t[-1]))
    pl.xlabel('time')
    
    x = []
    
    # plot dummy data to generate legend
    pl.plot(0,0,'k',linewidth=2,label='observed data')
    pl.plot(0,0, '--k',linewidth=1, label='estimated data')
    pl.plot(0,0, 'r',linewidth=1, label='segment border')
    
    for s in segments:
        x.extend(s.t)
        
        pl.gca().set_prop_cycle(None)
        Y = s.data[:, s.take_dims]
        if len(s.data_vel) > 0:
            Y = np.concatenate((s.data[:, s.take_dims], s.data_vel), axis=1)
        pl.plot(s.t, Y,  linewidth=2)
        
        # plot vertical line an change point position
        pl.axvline(s.t[0], color='r')
        
        # plot estimated model
        pl.gca().set_prop_cycle(None)
        pl.plot(s.t, s.data_est, '--', linewidth=1)
    pl.legend()       
    
    if changepoint_posterior is not None:
        # plot simulated change points
        pl.subplot(212)
        pl.title("Posterior of change point positions (%i simulations)" %(num_simulations))
        pl.plot(x, changepoint_posterior, 'k')
        pl.xlim((x[0],x[-1]))
        pl.xlabel('time')
    
        
        
def plot_changepoints_space(fig, title, segments, figsize = (12,6), 
                            view = (None, None)):
    """    
    Plots results of the movement segmentation for the first sensor in the 
    data (which is assumed to have x, y, z data in the first, second and third 
    column respectively) in 3D.
    
    Parameters
    ----------
    fig : int
        Defines number of first figure.
    
    title : string
        Title of figure.
        
    segments : list of instances of :class:`.segments.Segment`
        Segmented demonstrations as list of length *num_segments*, e.g., 
        elements of the list returned by :meth:`.VMCI.segment`.
    
    figsize : tuple, optional
        Size of first figure. Defaults ``(12, 6)``.
        
    view : tuple, optional
        View point of 3D plot. Defaults ``(None, None)``.
    """
    
    fig = pl.figure(fig, figsize= figsize, dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(view[0], view[1])
    pl.title(title)
    
    i = 0
    for s in segments:
        Y = s.data[:, s.take_dims]
        if i==0:
            i = 1
            ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], 'b', linewidth=2, label = 'sensor position')
            ax.scatter(Y[0, 0], Y[0, 1], Y[0, 2],  c = 'r', s=15, label = 'segment border')
        else:
            ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], 'b', linewidth=2)
            ax.scatter(Y[0, 0], Y[0, 1], Y[0, 2], c = 'r', s=15)
        
    ax.legend(numpoints = 1)
    
        