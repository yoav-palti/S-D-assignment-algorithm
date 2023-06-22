import numpy as np
from tqdm import tqdm
import Constants as C
import matplotlib.pyplot as plt
import scipy.optimize as sci
import itertools
import time
from numba import jit

MINDIST = 0.3
SIGMA = C.SIGMA
N_SIGMA = 30
M_SIGMA = 1
THRESHOLD_C_TIME_DIFFERENCE = N_SIGMA * SIGMA
YMAX = 500000
YMIN = -500000


@jit
def distance(points):
    return np.sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2)


def four_sensor_intersection_helper(emitter_location, sensors_loc, cdt41):
    """
    param: cdt41 is c times time 4 minus time 1
    """
    c_time_difference = (
            distance((emitter_location, sensors_loc[1])) - distance((emitter_location, sensors_loc[0])) - cdt41)
    if abs(c_time_difference) < THRESHOLD_C_TIME_DIFFERENCE:
        return True
    return False


def check_new_sensor_intersection_and_update_emitter_location(emitter_loc, cdt_new_to_first, points):
    d = distance(points)
    if (cdt_new_to_first) ** 2 > (d + M_SIGMA * SIGMA) ** 2:
        return None, False
    if is_one_point(emitter_loc):
        if four_sensor_intersection_helper(emitter_loc, points, cdt_new_to_first):
            return emitter_loc, True
    else:
        two_points = False
        the_real_point = 2
        for index in range(len(emitter_loc)):
            emitter = emitter_loc[index]
            if four_sensor_intersection_helper(emitter, points, cdt_new_to_first):
                the_real_point = index
                if two_points:
                    return emitter_loc, True
                two_points = True
        if the_real_point < 2:
            return emitter_loc[the_real_point], True
    return None, False


# probably won't finish
def check_new_sensor_and_update_emitter_location(depth, emitter_loc, current_sensor_locations, current_snake_toas):
    pass
    # d = distance(points)
    # if cdt_new_to_first ** 2 > d ** 2:
    #     return None, False
    # if is_one_point(emitter_loc):
    #     if four_sensor_intersection_helper(emitter_loc, points, cdt_new_to_first):
    #         return emitter_loc, True
    # else:
    #     two_points = False
    #     the_real_point = 2
    #     for index in range(len(emitter_loc)):
    #         emitter = emitter_loc[index]
    #         if four_sensor_intersection_helper(emitter, points, cdt_new_to_first):
    #             the_real_point = index
    #             if two_points:
    #                 return emitter_loc, True
    #             two_points = True
    #     if the_real_point < 2:
    #         return emitter_loc[the_real_point], True
    # return None, False


def is_one_point(point):
    return type(point[0]) != tuple


def hyperbola_intersect_scipy(x_0, y_0, x_1, y_1, a_0, b_0, a_1, b_1, theta_0, theta_1, y_max, y_min,
                              threshold_distance=MINDIST):
    """
    DO NOT TOUCH!

    This function gets two one sided hyperbolas and returns two of their intersection points
    using Newton-Raphson secant method.
    the hyperbola is parameterised in the following way:
    x = x_i + a_i * sqrt( 1 + ((y - y_i)/b_i)^2)
    and i is an indicator from {0,1}
    The hyperbolas aren't guaranteed to be x axis aligned.
    The parameter theta_i is the rotation from the x_axis of the hyperbola

    since for most hyperbolas there will be two intersection
    we compute with starting points at the edge of the search area [y_min, y_max]
    """

    theta = theta_1 - theta_0
    x_0_rot = np.sin(theta_0) * y_0 + np.cos(theta_0) * x_0
    y_0_rot = - np.sin(theta_0) * x_0 + np.cos(theta_0) * y_0
    x_1_rot = np.sin(theta_1) * y_1 + np.cos(theta_1) * x_1
    y_1_rot = - np.sin(theta_1) * x_1 + np.cos(theta_1) * y_1

    y_intersects = np.zeros(2)
    x_intersects = np.zeros(2)
    if b_0 == 0 or b_1 == 0:
        return line_hyperbola_intersect()
        if b_0 == 0:
            pass
        if b_1 == 0:
            pass

    # newton-raphson from minus strarting point
    args = (x_0_rot, y_0_rot, x_1_rot, y_1_rot, a_0, b_0, a_1, b_1, theta)
    starting_points = (y_max, y_min)

    intersected = newton_raphson_intersection(starting_points, x_intersects, y_intersects, args, threshold_distance)
    if not intersected:
        return None, 0

    return rotated_back(x_intersects, y_intersects, theta_0, threshold_distance)


# TODO: Fix RuntimeError
def newton_raphson_intersection(starting_points, x_intersects, y_intersects, args, threshold_distance):
    """
    DO NOT TOUCH!
    """

    """ args = (x_0_rot, y_0_rot, x_1_rot, y_1_rot, a_0, b_0, a_1, b_1, theta) """
    x_0_rot = args[0]
    y_0_rot = args[1]
    a_0 = args[4]
    b_0 = args[5]
    number_of_fails = 0

    for i in range(len(starting_points)):
        starting_point = starting_points[i]
        try:
            y_intersects[i] = sci.newton(f, starting_point, args=args, tol=0.01 * threshold_distance)
            x_intersects[i] = x_0_rot + a_0 * np.sqrt(1 + ((y_intersects[i] - y_0_rot) / b_0) ** 2)
            if number_of_fails == 1:
                y_intersects[0] = y_intersects[1]
                x_intersects[0] = x_intersects[1]
        except RuntimeError:
            number_of_fails += 1
            if number_of_fails == 2:
                return False
            elif i == 1:
                y_intersects[1] = y_intersects[0]
                x_intersects[1] = x_intersects[0]
    return True


def rotated_back(x_intersects, y_intersects, theta_0, threshold_distance):
    if distance(((x_intersects[0], y_intersects[0]), (x_intersects[1], y_intersects[1]))) < threshold_distance:
        x_return = x_intersects[0] * np.cos(theta_0) - y_intersects[0] * np.sin(theta_0)
        y_return = y_intersects[0] * np.cos(theta_0) + x_intersects[0] * np.sin(theta_0)
        return (x_return, y_return), 1

    x_return_1 = x_intersects[0] * np.cos(theta_0) - y_intersects[0] * np.sin(theta_0)
    y_return_1 = y_intersects[0] * np.cos(theta_0) + x_intersects[0] * np.sin(theta_0)
    x_return_2 = x_intersects[1] * np.cos(theta_0) - y_intersects[1] * np.sin(theta_0)
    y_return_2 = y_intersects[1] * np.cos(theta_0) + x_intersects[1] * np.sin(theta_0)

    return ((x_return_1, y_return_1), (x_return_2, y_return_2)), 2


def three_sensor_intersection_scipy(sensors_locations, times_of_arrival):  # , debug=False, emitter_debug=(0, 0)):
    """
    this function gets the locations and TOAs of three sensors and calculates their shared emitter's location
    """
    # defining and calculating important constants
    all_hyperbolas_are_defined, args = get_hyperbolas_parameters(sensors_locations, times_of_arrival)
    if all_hyperbolas_are_defined:
        intersections_array, two_hyperbolas_did_not_intersect = get_all_intersections(args)
        if two_hyperbolas_did_not_intersect:
            return None, False
        # finding all the valid intersections
        all_points = all_three_hyperbolas_intersections(intersections_array, THRESHOLD_C_TIME_DIFFERENCE)
        if len(all_points) > 1:
            return tuple(all_points), True
        elif len(all_points) == 1:
            return all_points[0], True
    return None, False


def get_hyperbolas_parameters(sensors_locations, times_of_arrival):
    """
    hyperbola equation is: x - x_center =a*sqrt(1+((y-y_center)/b)^2)
    the hyperbola is rotated in angle theta in respect to x axis

    @param sensors_locations:
    @param times_of_arrival:
    @return: all_hyperbolas_defined, args

            all of args entries are arrays of len 3 describing the parameters of each hyperbola
            args = (x_center, y_center, a, b, theta)
    """
    cdt = np.array([times_of_arrival[0] - times_of_arrival[1], times_of_arrival[1] - times_of_arrival[2],
                    times_of_arrival[2] - times_of_arrival[0]]) * C.C
    d = np.array(
        [distance((sensors_locations[0], sensors_locations[1])), distance((sensors_locations[1], sensors_locations[2])),
         distance((sensors_locations[2], sensors_locations[0]))])
    if np.all(cdt ** 2 < d ** 2):
        # defining and calculating important constants
        a = np.abs(cdt / 2)
        b = 1 / 2 * np.sqrt(d ** 2 - cdt ** 2)
        x_center = np.array([(sensors_locations[i][0] + sensors_locations[(i + 1) % 3][0]) / 2 for i in range(3)])
        y_center = np.array([(sensors_locations[i][1] + sensors_locations[(i + 1) % 3][1]) / 2 for i in range(3)])
        x_vec = np.array([(sensors_locations[i][0] - sensors_locations[(i + 1) % 3][0]) for i in range(3)])
        y_vec = np.array([(sensors_locations[i][1] - sensors_locations[(i + 1) % 3][1]) for i in range(3)])
        flip = (cdt > 0).astype(int)
        theta = np.arctan2(y_vec, x_vec) + np.pi * flip
        args = (x_center, y_center, a, b, theta)
        return True, args
    return False, None


def get_all_intersections(args):
    """ args = (x_center, y_center, a, b, theta) """
    x_center, y_center, a, b, theta = args
    intersections_array = []
    for hyperbola in range(3):
        next_hyperbola = (hyperbola + 1) % 3
        points, num_intersects = hyperbola_intersect_scipy(
            x_0=x_center[hyperbola], y_0=y_center[hyperbola], x_1=x_center[next_hyperbola],
            y_1=y_center[next_hyperbola], a_0=a[hyperbola], b_0=b[hyperbola], a_1=a[next_hyperbola],
            b_1=b[next_hyperbola], theta_0=theta[hyperbola], theta_1=theta[next_hyperbola], y_min=YMIN, y_max=YMAX)
        if num_intersects == 0:
            return intersections_array, True
        elif num_intersects == 1:
            intersections_array.append([points])
        else:
            intersections_array.append([points[0], points[1]])
    return intersections_array, False


def all_three_hyperbolas_intersections(points, threshold):
    all_points = []
    for point_0, point_1, point_2 in itertools.product(points[0], points[1], points[2]):
        mid_point = ((point_0[0] + point_1[0] + point_2[0]) / 3, (point_0[1] + point_1[1] + point_2[1]) / 3)
        sum_distances = distance((mid_point, point_0)) + distance((mid_point, point_1)) + distance((mid_point, point_1))

        if sum_distances / 3 < threshold:
            all_points.append(mid_point)
    return all_points


def line_hyperbola_intersect():
    return False, None


@jit
def f(y, x_0, y_0, x_1, y_1, a_0, b_0, a_1, b_1, theta):
    """
        DO NOT TOUCH!

        This function calculates the value of the second hyperbola function
        as a function of y substituting x from the closed form of the first hyperbola .
        The function returns 0 for points on the second hyperbola.
        theta is the relative angle between the two hyperbolas

    """
    if (b_0 == 0 or b_1 == 0):
        raise Exception("DEGENERATE HYPERBOLA")
    x = x_0 + a_0 * np.sqrt(1 + ((y - y_0) / b_0) ** 2)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return (a_1 * np.sqrt(1 + ((y * cos_theta - x * sin_theta - y_1) / b_1) ** 2)) - (
            x * cos_theta + y * sin_theta) + x_1


def snake_score(sensors_loc, times, emitter_location, sigma=C.SIGMA):
    sensors_location = np.array(sensors_loc).T
    arrival_times = np.array(times)
    # TODO create this function with gradient descent and Grid.
    if emitter_location is None:
        return 0, None
    elif is_one_point(emitter_location):
        # return -(len(sensors_loc) * (len(sensors_loc) - 1))
        location = np.array(emitter_location)
        args = (sensors_location, arrival_times, sigma)
        bounds = ((emitter_location[0] - 10 * sigma, emitter_location[0] + 10 * sigma),
                  (emitter_location[1] - 10 * sigma, emitter_location[1] + 10 * sigma))
        emitter_place = sci.minimize(grid_score, location, args, bounds=bounds, tol=0.001)
        # emitter_place = sci.minimize(grid_score, location, args)
        return emitter_place.fun, emitter_place.x
        # return emitter_place.fun

    else:
        location_1 = np.array(emitter_location[0])
        location_2 = np.array(emitter_location[1])
        args = (sensors_location, arrival_times, sigma)
        bounds_1 = ((emitter_location[0][0] - 10 * sigma, emitter_location[0][0] + 10 * sigma),
                    (emitter_location[0][1] - 10 * sigma, emitter_location[0][1] + 10 * sigma))
        bounds_2 = ((emitter_location[1][0] - 10 * sigma, emitter_location[1][0] + 10 * sigma),
                    (emitter_location[1][1] - 10 * sigma, emitter_location[1][1] + 10 * sigma))
        fit_1 = sci.minimize(grid_score, location_1, args, bounds=bounds_1, tol=0.001)
        fit_2 = sci.minimize(grid_score, location_2, args, bounds=bounds_2, tol=0.001)
        if fit_1.fun < fit_2.fun:
            return fit_1.fun, fit_1.x
        return fit_2.fun, fit_2.x


@jit
def grid_score(emitter_loc, sensors_location, arrival_times, sigma=C.SIGMA):
    emitter_x = emitter_loc[0]
    emitter_y = emitter_loc[1]
    sum_score = 0
    distances = np.sqrt(np.power(sensors_location[0] - emitter_x, 2) + np.power(sensors_location[1] - emitter_y, 2))
    for first_sensor in range(len(sensors_location[0])):
        for second_sensor in range(first_sensor + 1, len(sensors_location[0])):
            sum_score -= np.exp(-np.power((distances[first_sensor] - distances[second_sensor] - C.C * (
                    arrival_times[first_sensor] - arrival_times[second_sensor])) / sigma, 2) / 2)
    return sum_score  # + len(sensors_location[0])


def condense_matrix(matrix, num_paths, num_signals):
    condensed_matrix = np.zeros((num_paths + 1, num_signals + 1))
    for row, column in itertools.product(range(num_paths + 1), range(num_signals + 1)):
        if row == 0 and column == 0:
            condensed_matrix[row][column] = matrix[num_paths][num_signals]
        elif row == 0:
            condensed_matrix[row][column] = matrix[num_paths][column - 1]
        elif column == 0:
            condensed_matrix[row][column] = matrix[row - 1][num_signals]
        else:
            condensed_matrix[row][column] = matrix[row - 1][column - 1]
    return condensed_matrix


def get_real_serial_number(prevs):
    """
    this function belongs to the Greedy Assigning Algorithm.
    After removing a path from the tree the real serial of the TOA needs to be computed.
    @param prevs:
    @return:
    """
    a = 1
    a += 1
    while True:
        index, value = min(enumerate(prevs), key=lambda x: x[1])
        if index == len(prevs) - 1:
            return prevs[index]
        else:
            if prevs[index] > 0:
                for i in range(index + 1, len(prevs)):
                    if prevs[i] > 0:
                        prevs[i] += 1
            prevs[index] = 1000


def find_max_in_grid(sensors_location, arrival_times, ranged, sigma=C.SIGMA):
    N = 501
    grid = np.zeros((N, N))
    for row in range(N):
        for column in range(N):
            x = ranged - 2 * ranged / (N - 1) * column
            y = ranged - 2 * ranged / (N - 1) * row
            x = ranged / (N - 1) * column
            y = ranged / (N - 1) * row
            a = np.array((x, y))
            b = grid_score(a, sensors_location, arrival_times, sigma)
            grid[row, column] = b
    minimum = grid.argmin()
    mini = grid.min()
    row = minimum // N
    column = minimum % N
    x = ranged - 2 * ranged / (N - 1) * column
    y = ranged - 2 * ranged / (N - 1) * row
    x = ranged / (N - 1) * column
    y = ranged / (N - 1) * row
    ex = (1 - 2 * np.linspace(0, 1, N)) * ranged
    why = (1 - 2 * np.linspace(0, 1, N)) * ranged
    # plt.contourf(ex, why, grid, 20, cmap='RdGy')
    # plt.colorbar()
    return (x, y, mini), grid


def Test_DrawHyperbolas(sensors_loc, times, emitter, dimensions=20000):
    """

    @param sensors_loc: sensors locations
    @param times: TOAs
    @param emitter: emitter location
    @param dimensions: size of map
    @return: None
    """
    cdt = np.array([times[0] - times[1], times[1] - times[2], times[2] - times[0]]) * C.C
    d = np.array([distance((sensors_loc[0], sensors_loc[1])), distance((sensors_loc[1], sensors_loc[2])),
                  distance((sensors_loc[2], sensors_loc[0]))])
    alpha = np.all(cdt < d)
    if (alpha):
        a = np.abs(cdt / 2)
        b = 1 / 2 * np.sqrt(d ** 2 - cdt ** 2)
        x_center = np.array([(sensors_loc[i][0] + sensors_loc[(i + 1) % 3][0]) / 2 for i in range(3)])
        y_center = np.array([(sensors_loc[i][1] + sensors_loc[(i + 1) % 3][1]) / 2 for i in range(3)])
        x_vec = np.array([(sensors_loc[i][0] - sensors_loc[(i + 1) % 3][0]) for i in range(3)])
        y_vec = np.array([(sensors_loc[i][1] - sensors_loc[(i + 1) % 3][1]) for i in range(3)])
        flip = (cdt > 0).astype(int)
        theta = np.arctan2(y_vec, x_vec) + np.pi * flip
        # plot the hyperbolas.
        if True:
            # plt.scatter([sensors_loc[0][0], sensors_loc[1][0], sensors_loc[2][0]],
            #             [sensors_loc[0][1], sensors_loc[1][1], sensors_loc[2][1]])
            plt.scatter([sensors_loc[0][0]],
                        [sensors_loc[0][1]], color="black")
            plt.scatter([sensors_loc[1][0]],
                        [sensors_loc[1][1]], color="purple")
            plt.scatter([sensors_loc[2][0]],
                        [sensors_loc[2][1]], color="orange")
            plt.scatter([emitter[0]], [emitter[1]], color="blue")
            y_1 = np.linspace(-dimensions, dimensions, 201)
            y_2 = np.linspace(-dimensions, dimensions, 201)
            y_3 = np.linspace(-dimensions / 8, dimensions / 8, 201)
            x_1 = abs(a[0]) * np.sqrt(1 + ((y_1) / b[0]) ** 2)
            x_2 = abs(a[1]) * np.sqrt(1 + ((y_2) / b[1]) ** 2)
            x_3 = abs(a[2]) * np.sqrt(1 + ((y_3) / b[2]) ** 2)
            x1 = np.cos(theta[0]) * x_1 - np.sin(theta[0]) * y_1 + x_center[0]
            y1 = np.cos(theta[0]) * y_1 + np.sin(theta[0]) * x_1 + y_center[0]
            x2 = np.cos(theta[1]) * x_2 - np.sin(theta[1]) * y_2 + x_center[1]
            y2 = np.cos(theta[1]) * y_2 + np.sin(theta[1]) * x_2 + y_center[1]
            x3 = np.cos(theta[2]) * x_3 - np.sin(theta[2]) * y_3 + x_center[2]
            y3 = np.cos(theta[2]) * y_3 + np.sin(theta[2]) * x_3 + y_center[2]

            plt.plot(x1, y1, "red")
            plt.plot(x2, y2, "yellow")
            plt.plot(x3, y3, "green")
            plt.axis('equal')
            plt.show()


if __name__ == '__main__':
    pass
