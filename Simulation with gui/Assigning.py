from Sensor import Sensor
from Emmiter import Emmiter
import numpy as np
import scipy
from Tree_of_Data import TreeOfData
from Testing import Testing
import tqdm
from typing import List, Union, Tuple
#use pip install -r requirments.txt

def assign_2d(costMatrix: np.ndarray) -> Tuple[np.ndarray, float]:
    ''' This function solves a 2-D assignment problem with 1 index assigned to
    the dummy. Algorithm uses scipy.optimize.linear_sum_assignment as the assignment algorithm.

    a 4x3 cost matrix would look like this:
          ┌─────┬─────┬─────┐
      →   │  *  │  *  │  *  │
    Dummy │  *  │  *  │  *  │
          │  *  │  *  │  *  │
          │  *  │  *  │  *  │
          └─────┴─────┴─────┘
             ↑
    Dummy Assignment

    :param costMatrix: A 2-D cost matrix with the cost of assigning each row to
    each column.
    :return: A 2-D numpy array with the assignment. The assignment is a 2-D
    '''
    # The dummy-dummy assignment does not take part in the solution. We will set
    # the cost of dummy-dummy assignment to 0 and restore it later.
    temp = costMatrix[0, 0]
    costMatrix[0, 0] = 0
    # Assess the size of costMatrix.
    nRow, nCol = costMatrix.shape
    # Padding is done using the (1,2:end) and (2:end,1) as the
    # cost of unassignment for rows and columns respectively.
    relaxedMatrix = pad_cost_matrix(costMatrix, 0)
    # Assign using the specified algorithm.
    assignments = np.array(scipy.optimize.linear_sum_assignment(relaxedMatrix))
    assignments = assignments.T
    # priceWithDummy is assigned to 0 as price has no meaning for non-auction
    # algorithms.
    cost = compute_cost(relaxedMatrix, assignments)
    assignments = handle_dummy_assignments(assignments, nRow, nCol)

    return assignments, cost

def assign_sd(tree: TreeOfData, number_of_iterations:int ,min_gap_ratio:float):
    # initializing the variables
    num_constraints = tree.tree_size - 2
    best_dual_cost = -np.inf
    best_feasible_cost = np.inf * np.ones(num_constraints)
    feasible_cost = np.zeros(num_constraints)
    # in the tree documentation, we do the first and and second state automatically.
    for iter_num in tqdm.trange(number_of_iterations):
        real_children_paths = [[i + 1] for i in range(len(tree.sensor_times[0]))]
        # the matrix of the first and second sensor in the tree is received
        # by using the first from the tree node to the first sensor signals nodes.
        # this paths are described by real_children_paths.
        two_d_cost = tree.convert_2D_matrix(real_children_paths)

        # calculating the relaxed solution
        constrained_assignment, cost = assign_2d(two_d_cost)
        tree_depth = 2
        lagrangian_multiplier_sum = tree.get_lagrangian_multipliers_sum(tree_depth)

        total_dummy_costs = two_d_cost[0, :]
        print('the two d cost is:\n', two_d_cost)
        dummy_negative_cost = np.sum(np.where(total_dummy_costs < 0, total_dummy_costs, 0))
        print(
            f'the cost contibutors are - cost:{cost}, lagrangian_sum:{lagrangian_multiplier_sum}, total_dummy:{dummy_negative_cost}')

        # dual cost is the cost of the relaxed solution
        dual_cost = cost + lagrangian_multiplier_sum + dummy_negative_cost
        best_dual_cost = np.max((dual_cost, best_dual_cost))

        assignments = constrained_assignment
        number_of_path_uses = []
        # The lagrange multipliers of the two dimention problem are zero
        number_of_path_uses.append(np.zeros(len(tree.lagrange_multipliers[0])))
        number_of_path_uses.append(np.zeros(len(tree.lagrange_multipliers[1])))
        number_of_path_uses.append(tree.get_number_of_uses_from_paths_list(assignments))

        for i in range(num_constraints):
            # Compute feasible costs
            tree_depth = i + 3
            lagrangian_multiplier_sum = tree.get_lagrangian_multipliers_sum(tree_depth)
            feasible_cost_matrix = tree.convert_2D_matrix(assignments)

            print('the feasible_cost_matrix is:\n', feasible_cost_matrix)

            total_dummy_costs = feasible_cost_matrix[0, :]
            dummy_negative_cost = np.sum(np.where(total_dummy_costs < 0, total_dummy_costs, 0))
            # Get feasible solution.
            enforced_assignment, cost = assign_2d(feasible_cost_matrix)

            # Compute feasible cost
            feasible_cost[i] = cost + lagrangian_multiplier_sum + dummy_negative_cost
            best_feasible_cost[i] = np.min((best_feasible_cost[i], feasible_cost[i]))
            assignments = chain_assignments(assignments, enforced_assignment)
            print(
                f'the cost contibutors are cost:{cost}, lagrangian_sum:{lagrangian_multiplier_sum}, total_dummy:{dummy_negative_cost}')
            print('the cost is:', feasible_cost[i])
            if i != num_constraints - 1:
                number_of_path_uses.append(tree.get_number_of_uses_from_paths_list(assignments))
                # number_of_path_uses[i][j] = number of usages of signal j received in sensor i.

        update_solution = best_feasible_cost[num_constraints - 1] == feasible_cost[num_constraints - 1]
        # updateing the solution if a new minimum for the feasible cost was found.
        gap_ratio = (feasible_cost[num_constraints - 1] - dual_cost) / np.abs(dual_cost)
        if update_solution or iter_num == 0:
            best_solution = assignments
            best_gap_ratio = gap_ratio
        # update Lagrangian Multiplers.
        # number_of_path_uses are used to calculate the update in the lagrangian multipliers.
        # each leaf in the tree has its own gradient.
        # i will assume that the gradients = 1 - number of usages of a certain signal.
        gradients = calculate_gradients(number_of_path_uses)

        # calculating the gaps between the costs.
        gaps = calculate_gaps(best_feasible_cost, best_dual_cost)
        print('the gaps are: ', gaps)
        # calculate the lagrangian multipliers gradient.
        # another way to calculate the lagrangian multipliers gradient is to use the
        # advanced way. we have no time to implement it, so we will not use it.

        lagrangian_multiplers_gradient = calculate_lagrangian_multiplers_change(gradients, gaps, iter_num)
        tree.update_tree_lagrangian_multipliers(lagrangian_multiplers_gradient)
        if gap_ratio < min_gap_ratio:
            print('stopped at iteration: ', iter_num)
            return best_solution, tree

    print('stopped at gap: ', best_gap_ratio)
    return best_solution, tree

def assign_sensor_data(sensor_location:Union[List[Tuple[float]], np.ndarray], sensor_times: Union[List[List[float]],np.ndarray],
                       number_of_iterations:int = 200, min_gap_ratio:float = 0.01, time_of_cycle:float = None) -> np.ndarray:
    """returns the assignment for the sd - assignment problem, using lagrangian relaxation.
        it this case, the assignment algorithm receives the sensor locations
        and the signals received time, and returns the assignment of the signals to
        different emitters.
        this algorithm should work the same way as the matlab assignsd algorithm.

        :param sensor_location: the location of the sensors.
        :param sensor_times: a time array of the signals received by the sensors.
        :param number_of_iterations: the number of algorithm iterations
        :param min_gap_ratio: the minimal gap ratio between the best feasible cost and the best dual cost.
        :param time_of_cycle: the period time of the sensors.
        :return: the assignment of the signals to the emitters.
    """
    if time_of_cycle is None:
        import time
        start_time = time.time()
        tree = TreeOfData(sensor_location, sensor_times)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The tree was created in: {execution_time} seconds")
        print(f"The number of nodes in the tree is: {tree.number_of_paths}")
    else:
        tree = TreeOfData(sensor_location, sensor_times, time_of_cycle)
    return assign_sd(tree, number_of_iterations, min_gap_ratio)


def calculate_gradients(gradient_list: List[np.ndarray]) -> List[np.ndarray]:
    '''calculates the g vector list, using the gradient list.
    :param gradient_list - the gradient list.
    :return: the g vector list.
    '''
    g = []
    for gradient_array in gradient_list:
        g.append(1 - np.array(gradient_array))
    return g

def calculate_gaps(feasible_cost_array: np.ndarray, dual_cost: float) -> np.ndarray:
    '''calculates the gaps between the costs.
    :param feasible_cost_array: the feasible cost array.
    :param dual_cost: the dual cost (sometimes called the relaxed cost).
    :return: the gaps between the costs.
    '''

    num_constraints = len(feasible_cost_array)
    gaps = np.zeros(num_constraints + 2)
    for i in range(num_constraints):
        if i == 0:
            gaps[i + 2] = feasible_cost_array[i] - dual_cost
        else:
            gaps[i + 2] = feasible_cost_array[i] - feasible_cost_array[i - 1]
        if np.isneginf(gaps[i+2]):
            Exception('the gap is negative infinity')
    return gaps
def calculate_lagrangian_multiplers_change(gradients: List[np.ndarray], gap: np.ndarray) -> List[np.ndarray]:
    '''returns the lagrangian multipliers gradient, using
    the method described in the book "Design and Analysis Of Modern
    Tracking Systems" by Samuel Blackman and Robert Popoli.
    :param gradients: the gradient vector list.
    :param gap: the gap vector.
    :return: the lagrangian multipliers change.
    '''
    lagrangian_multipliers_gradient = []
    for i in range(len(gradients)):
        squared_sum_of_g = np.sum(np.power(gradients[i], 2))
        if squared_sum_of_g == 0:
            lagrangian_multipliers_gradient.append(np.zeros(len(gradients[i])))
        else:
            lagrangian_multipliers_gradient.append(gradients[i] * gap[i] / squared_sum_of_g)
    return lagrangian_multipliers_gradient


def chain_assignments(previous_assignment: np.ndarray, current_assignment: np.ndarray) -> np.ndarray:
    """returns the chained assignments of the previous assignment and the current assignment.
    Args:
        previous_assignment - the assignment of the previous assignment.
        current_assignment - the assignment of the current assignment.

    Returns:
        the chained assignments of the previous assignment and the current assignment.
    """
    previous_assignment_length = previous_assignment.shape[1]
    non_dummy_current_assignment = current_assignment[current_assignment[:, 0] != 0]
    dummy_current_assignment = current_assignment[current_assignment[:, 0] == 0]
    chained_assignments_non_dummy = np.concatenate((previous_assignment, non_dummy_current_assignment[:, 1:]), axis=1, dtype=int)
    chained_assignments_dummy = np.concatenate((np.zeros((len(dummy_current_assignment),previous_assignment_length - 1), dtype=int), dummy_current_assignment[:]), axis=1, dtype=int)

    chained_assignments = np.concatenate((chained_assignments_non_dummy, chained_assignments_dummy))

    return chained_assignments


def pad_cost_matrix(costMatrix: np.ndarray, temp: float) -> np.ndarray:
    '''This function calculates the padded cost for allowing multiple
    dimensions on index 1 on each dimension.

    transforms the matrix in the following way:
    a 4x3 cost matrix would look like this:
          ┌─────┬─────┬─────┐
      →   │  *  │  *  │  *  │
    Dummy │  *  │  *  │  *  │
          │  *  │  *  │  *  │
          │  *  │  *  │  *  │
          └─────┴─────┴─────┘
             ↑
    Dummy Assignment
    and would be transformed to this:
          ┌─────┬─────┬──────┬──────┬──────┐
          │  *  │  *  │  *   │ inf  │ inf  │
          │  *  │  *  │ inf  │  *   │ inf  │
          │  *  │  *  │ inf  │ inf  │  *   │
      →   │  *  │ inf │<temp>│<temp>│<temp>│
      →   │ inf │  *  │<temp>│<temp>│<temp>│
    Dummy └─────┴─────┴──────┴──────┴──────┘
    Assignments           ↑      ↑      ↑
                        Dummy Assignments
    The dummy assignments are used to make the dummy assignments can be
    assigned to each assignment. Therefore, each dummy assignment row and
    column is seperated into multiple rows and columns, and the dummy dummy
    assignments are emitted because they are not needed.
    :param costMatrix: The cost matrix.
    :param temp: The value of the dummy - dummy assignment.
    :return: The padded cost matrix.
    '''

    innerCost = costMatrix[1:, 1:]
    dummyFirst = costMatrix[0, 1:]
    dummySecond = costMatrix[1:, 0]
    nFirst = len(dummyFirst)
    nSecond = len(dummySecond)
    dummyCostFirst = np.full((nFirst, nFirst), np.inf)
    dummyCostSecond = np.full((nSecond, nSecond), np.inf)
    indFirst = np.arange(nFirst)
    dummyCostFirst[indFirst, indFirst] = dummyFirst
    indSecond = np.arange(nSecond)
    dummyCostSecond[indSecond, indSecond] = dummySecond
    dummyDummyAssignment = temp * np.ones((nFirst, nSecond))
    paddedCost = np.concatenate((np.concatenate((innerCost, dummyCostSecond), axis=1),
                                 np.concatenate((dummyCostFirst, dummyDummyAssignment), axis=1)),
                                axis=0)
    return paddedCost


# assign-2d helping function
def handle_dummy_assignments(assignments: np.ndarray, nRow: int, nCol: int) -> np.ndarray:
    '''Restoring the original indices (the indices used before the padding) from the
    padded assignments input.
    :param assignments: the padded assignments input
    :param nRow: the number of rows in the original cost matrix
    :param nCol: the number of columns in the original cost matrix
    :return: the assignments in the original indices
    '''

    assignments = assignments + 1
    # Modify the priceWithDummy to update the index.
    # Remove the padded rows and columns from the solution and restore the
    # actual solution.

    # clarification - valid = not dummy.
    solution_rows = assignments[:, 0]  # the assignment solution rows
    validRows = solution_rows < nRow  # valid rows are the ones that are not dummy
    valid_assignment_columns = assignments[validRows, 1]  # the assigned second column valid rows
    valid_assignment_rows = assignments[validRows, 0]  # the assigned first column valid rows
    valid_assignment_columns[
        valid_assignment_columns >= nCol] = 0  # changing rows that coupled with the dummy assignment to 0
    dummy_solution_columns = assignments[~validRows, 1]  # the assigned second column that has dummy rows
    valid_to_dummy_assignments = dummy_solution_columns < nCol
    dummy_solution_columns = dummy_solution_columns[valid_to_dummy_assignments]

    assignments = np.concatenate([np.vstack([valid_assignment_rows, valid_assignment_columns]).T,
                                  np.vstack(
                                      [np.zeros(len(dummy_solution_columns), dtype=int), dummy_solution_columns]).T],
                                 axis=0)
    assignments = assignments[np.argsort(assignments[:, 0]), :]
    # remove the dummy - dummy assignments (one dummy - dummy assignment is always present,
    # and will be added later)
    dummyAssignment = np.logical_and(assignments[:, 0] == 0, assignments[:, 1] == 0)
    numDummies = np.sum(dummyAssignment)
    if numDummies >= 1:
        assignments = assignments[numDummies:, :]

    return assignments


# assign-2d helping function
def compute_cost(costMatrix: np.ndarray, assignments: np.ndarray) -> float:
    '''This function computes the cost of assignment given the cost matrix and
    the assignment. The assignment is a 2-D array with the first column
    representing the row and the second column representing the column.
    :param costMatrix: the cost matrix
    :param assignments: the assignments, where the first column is the assigned
    row and the second column is the assigned column
    :return: the cost of the assignment
    '''
    row = assignments[:, 0]
    col = assignments[:, 1]
    cost = costMatrix[row, col].sum()
    return cost

if __name__ == '__main__':
    test1 = Testing(lambda x: 0)
    sensor_locs, sensor_ts = test1.create_assignment_scenario(8, 5)
    print(assign_sensor_data(sensor_locs, sensor_ts, 200))