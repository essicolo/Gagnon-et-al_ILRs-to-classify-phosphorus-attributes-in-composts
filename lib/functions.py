import numpy as np
import pandas as pd
from skbio.stats.composition import clr_inv
from scipy.stats import f as statsf
import matplotlib.pyplot as plt

def sbp_basis(sbp):
    """
    Builds an orthogonal basis from a sequential binary partition
    
    
    Parameter
    ---------
    sbp: np.array or pd.DataFrame, int
        A contrast matrix, also known as a sequential binary partition, where
        every row represents a partition between two groups of features. A part
        labelled `+1` would correspond to that feature being in the numerator of
        the given row partition, a part labelled `-1` would correspond to
        features being in the denominator of that given row partition, and `0`
        would correspond to features excluded in the row partition.
    
    Returns
    -------
    np.array
        An orthonormal basis in the Aitchison simplex

    """
    n_pos = (sbp == 1).sum(axis=1)
    n_neg = (sbp == -1).sum(axis=1)
    psi = np.zeros(sbp.shape)
    for i in range(0, sbp.shape[0]):
        psi[i, :] = sbp[i, :] * np.sqrt((n_neg[i] / n_pos[i])**sbp[i, :] /
                                        np.sum(np.abs(sbp[i, :])))
    return clr_inv(psi)
    
def ilrDefinition(sbp, side="-+", sep_elem = ",", sep_bal = " | ", sep_left = "[", sep_right = "]"):

    """
    Creates a vector of pretty names for ilr balances, e.g. [Mg,Ca | N,P,K]
    
    Parameters
    ----------
    sbp: pd.DataFrame, int
        A sequantial binary partition
    side: string
        '-+' for the numerator on left of '+-' for the numerator on right
    sep_elem: string
        A string separating elements of the simplex
    sep_bal: string
        A string separating denominator and numerator
    sep_left: string
        A string to start
    sep_right: strinf
        A string to end
        
    Returns
    -------
    np.array, string
        A vector of strings
    
    """
    
    if sbp.shape[0] != (sbp.shape[1] -1):
        raise ValueError("SBP not valid")
    
    ilrDef = []
    
    for n in range(sbp.shape[0]):
        pos = sbp.loc[n, sbp.iloc[n,:] == 1].index.values
        neg = sbp.loc[n, sbp.iloc[n,:] == -1].index.values
    
        if (side=="-+"):
            pos = pos[::-1]
            neg = neg[::-1]

        pos_group = str()
        neg_group = str()
    
        for i in range(len(pos)):
            if i == (len(pos)-1):
                pos_group = pos_group + pos[i]
            else:
                pos_group = pos_group + pos[i] + sep_elem

        for i in range(len(neg)):
            if i == (len(neg)-1):
                neg_group = neg_group + neg[i]
            else:
                neg_group = neg_group + neg[i] + sep_elem
        
        if side=="+-":
            ilrDef.append(sep_left + pos_group + sep_bal + neg_group + sep_right)
        else:
            ilrDef.append(sep_left + neg_group + sep_bal + pos_group + sep_right)
        
    return ilrDef
    
    
def ellipse(X, level=0.95, method='deviation', npoints=100):
    """
    Returns coordinates of a 2D error of deviation ellipse.
    
    Parameter
    ---------
    X: np.array or pd.DataFrame
        A 2D matrix or table
    level: float
        The confidence level
    method: string
        Either 'deviation' or 'error'
    npoints:
        number of points drawing the ellipse
    
    Returns
    -------
    np.array
        Coordinates of the ellipse

    """
    cov_mat = np.cov(X.T)
    dfd = X.shape[0]-1
    dfn = 2
    center = np.mean(X, axis=0)
    if method == 'deviation':
        radius = np.sqrt(2 * statsf.ppf(q=level, dfn=dfn, dfd=dfd))
    elif method == 'error':
        radius = np.sqrt(2 * statsf.ppf(q=level, dfn=dfn, dfd=dfd)) / np.sqrt(X.shape[0])
    angles = (np.arange(0,npoints+1)) * 2 * np.pi/npoints
    circle = np.vstack((np.cos(angles), np.sin(angles))).T
    ellipse = center + (radius * np.dot(circle, np.linalg.cholesky(cov_mat).T).T).T
    return ellipse    

def biplot(objects, eigenvectors, eigenvalues=None, 
           labels=None, scaling=1, xpc=0, ypc=1,
           group=None, plot_ellipses=False):
    """
    Returns a biplot
    
    Parameter
    ---------
    objects: np.array or pd.DataFrame
        The scores
    eigenvectors: np.array or pd.DataFrame, float
        The loadings
    eigenvalues: np.array or pd Series, float
        The eigenvalues needed for scaling 2
    labels: np.array or pd Series, string
        Labels shown on loadings
    scaling: int
        Either 1 (correlation biplot) or 2 (distance biplot)
    xpc: int
        The axis index defining the x-axis component
    ypc: int
        The axis index defining the y-axis component
    group: np.array or pd.DataFrame, int
        A vector defining the grouping
    plot_ellipses: bool
        If True, both deviation and error ellipses are plotted at 0.95 level
        
    
    Returns
    -------
    Matplotlib object
    """
    # select scaling
    if scaling == 1 or scaling == 'distance':
        scores = objects[:, [xpc, ypc]]
        loadings = eigenvectors[[xpc, ypc], :]
    elif scaling == 2 or scaling == 'correlation':
        scores = objects.dot(np.diag(eigenvalues**(-0.5)))[:, [xpc, ypc]]
        loadings = eigenvectors.dot(np.diag(eigenvalues**0.5))
    
    # draw the cross
    plt.axvline(0, ls='solid', c='k', linewidth=0.2)
    plt.axhline(0, ls='solid', c='k', linewidth=0.2)
    
    # draw the ellipses
    if group is not None and plot_ellipses:
        groups = np.unique(group)
        for i in range(len(groups)):
            mean = np.mean(scores[group==groups[i], :], axis=0)
            plt.text(mean[0], mean[1], groups[i],
                     ha='center', va='center', color='k', size=10)
            ell_dev = ellipse(X=scores[group==groups[i], :], level=0.95, method='deviation')
            ell_err = ellipse(X=scores[group==groups[i], :], level=0.95, method='error')
            plt.fill(ell_err[:,0], ell_err[:,1], alpha=0.6, color='grey')
            plt.fill(ell_dev[:,0], ell_dev[:,1], alpha=0.2, color='grey')
    
    # plot scores
    if group is None:
        plt.scatter(scores[:,xpc], scores[:,ypc])
    else:
        markers = ['o', '^', 's', 'x', 'p', 'P', '*']
        for i in range(len(np.unique(group))):
            cond = group == np.unique(group)[i]
            plt.plot(scores[cond, 0], scores[cond, 1], marker=markers[i], linewidth=0,
                    label=np.unique(group)[i])
    
    # plot loadings
    for i in range(loadings.shape[1]):
        plt.arrow(0, 0, loadings[xpc, i], loadings[ypc, i], 
                  color = 'black', head_width=np.ptp(objects)/100)
    
    # plot loading labels
    if labels is not None:
        for i in range(loadings.shape[1]):
            plt.text(loadings[xpc, i]*1.2, loadings[ypc, i]*1.2, labels[i], 
                     color = 'black', ha = 'center', va = 'center')


