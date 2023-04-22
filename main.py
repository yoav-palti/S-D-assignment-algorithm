# S-D assignment argorithm
__auther__ = "Yoav Palti"
__email__ = "yoav.palti99@gmail.com"

import numpy as np
import scipy.optimize as opt
import sparse


def assignsd(
        cost_mat,
        desired_gap=0.01,
        max_iterations=100
):
    """
    returns a table of assignments, assignments, of detections to tracks by finding a suboptimal
    solution to the S-D assignment problem using Lagrangian relaxation. The cost of each potential
    assignment is contained in the cost matrix, costmatrix. The algorithm terminates when the gap
    reaches below desired_gap or if the number of iterations reaches max_iterations

    Parameters
    ----------
    cost_mat: numpy.ndarray
         n-dimensional cost matrix where costmatrix(i,j,k ...)
         defines the cost of the n-tuple (i,j,k, ...)  in assignment

    desired_gap: float
        the desired duality gap where the duality gap is JS - J2*
    max_iterations:


    Returns
    -------
    assignments:
    cost:

    """
    # init_vars

if __name__ == '__main__':
    pass
