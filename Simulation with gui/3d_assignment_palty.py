# S-D assignment argorithm
__auther__ = "Yoav Palti"
__email__ = "yoav.palti99@gmail.com"

import numpy as np
import scipy.optimize as opt
#import sparse

def get_dual_gap(feasable_cost, dual_cost):
    return (feasable_cost - dual_cost)/np.abs(dual_cost)

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
    # will be recursive latter...
    lagrange_multipliers = np.zeros(cost_mat.shape[-1])
    min_feasable_cost = np.inf
    max_dual_cost = -np.inf
    dual_gap = np.inf
    while dual_gap > desired_gap:
        # find the relaxed solution (dual)
        relaxed_cost_mat = cost_mat - lagrange_multipliers
        reduced_cost_idxes = np.argmin(relaxed_cost_mat, axis=-1)
        # reduced_cost is just np.amin(relaxed_cost_mat, axis=-1)
        reduced_cost = np.take_along_axis(relaxed_cost_mat, np.expand_dims(reduced_cost_idxes, axis=-1), axis=-1).squeeze(axis=-1)
        row_ind, col_ind = opt.linear_sum_assignment(reduced_cost)
        dual_cost = np.sum(reduced_cost[row_ind, col_ind]) + np.sum(lagrange_multipliers)


        # find the feasible (primal) solution
        feasable_cost_mat = relaxed_cost_mat[row_ind, col_ind]
        feasable_row_idx, feasable_col_idx = opt.linear_sum_assignment(feasable_cost_mat)
        feasable_idx_solution = tuple((row_ind[feasable_row_idx],col_ind[feasable_row_idx],feasable_col_idx))
        feasable_cost = np.sum(cost_mat[feasable_idx_solution])

        # update the lagrange multiplier
        # remove FA solution in collision count
        fa_solution_filter = row_ind + col_ind != 0  # can not both be 0
        last_scan_idexes = reduced_cost_idxes[row_ind[fa_solution_filter], col_ind[fa_solution_filter]]
        assign_collision_count = np.bincount(last_scan_idexes, minlength=cost_mat.shape[-1])
        collision_score = np.ones_like(assign_collision_count)-assign_collision_count
        collision_score[0] = 0  # no collision count for unassigned detections (demi)
        collision_score_normalize = np.square(collision_score).sum()

        if feasable_cost < min_feasable_cost:
            min_feasable_cost = feasable_cost
        if dual_cost > max_dual_cost:
            max_dual_cost = dual_cost
        dual_gap = get_dual_gap(min_feasable_cost, max_dual_cost)
        c_a = (min_feasable_cost-max_dual_cost) / collision_score_normalize
        lagrange_multipliers = lagrange_multipliers + c_a * collision_score
        # print(f'dual_gap: {dual_gap}')
        print(f'feasable_cost: {feasable_cost}')
        print(f'dual_cost: {dual_cost}')
        # print(f'feasable_idx_solution: {feasable_idx_solution}')
        # print(f'lagrange_multipliers: {lagrange_multipliers}')



if __name__ == '__main__':
    # test
    # example from Design and Analysis Of Modern Tracking Systems S Blackman R Popoli page 415
    cost_mat = np.zeros((3, 3, 3))
    cost_mat[0, 1, 1] = -10.2
    cost_mat[0, 2, 1] = -4.7
    cost_mat[0, 2, 2] = -5.5
    cost_mat[1, 0, 1] = -6.8
    cost_mat[1, 0, 2] = -5.2
    cost_mat[1, 1, 0] = -6.8
    cost_mat[1, 2, 0] = -10.9
    cost_mat[1, 1, 1] = -18
    cost_mat[1, 1, 2] = -14.8
    cost_mat[1, 2, 1] = -17
    cost_mat[1, 2, 2] = -9.9
    cost_mat[2, 0, 1] = -13.2
    cost_mat[2, 0, 2] = -10.6
    cost_mat[2, 1, 0] = -4.5
    cost_mat[2, 2, 0] = -11.1
    cost_mat[2, 1, 2] = -14.1
    cost_mat[2, 2, 1] = -9
    cost_mat[2, 2, 2] = -16.7
    print(cost_mat)

    assignsd(cost_mat)
