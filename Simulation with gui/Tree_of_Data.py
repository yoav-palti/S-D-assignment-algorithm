import numpy as np
from tqdm import tqdm
import Constants as C
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm
from MyNode import MyNode
import ScoreCalculation as scc
import scipy.optimize as sci
import itertools
import time


class TreeOfData:
    """
    this class is to store the scores of each viable snake. it is stored in this tree structure
    each layer of the tree represents a different sensor.
    """

    # DONE
    def __init__(self, sensor_locations, sensor_times, time_of_cycle=C.TIME_OF_CYCLE, empty_tree=False):
        """
        param: root_Node - the pointer for the current branch
            param: sensor_locations - location of all sensors in the algorithm
            param: sensor_times - 2d array of all the times that a specific sensor received.

            field: sensor_locations - location of all sensors in the algorithm
            field: sensor_times - 2d array of all the times that a specific sensor received.
            field: sensor_num_times - the length of each array in sensor_times
            field: tree_size: number of sensors
            field: weights - the weights for the relaxed solution
            field: tree - the main data structure

            field: current_snake_locations - a field used for building the tree
            field: current_snake_times - a field used for building the tree
        """
        self.sensor_locations = sensor_locations
        self.sensor_times = sensor_times
        self.sensor_num_times = []
        self.lagrange_multipliers = []
        self.time_of_cycle = time_of_cycle
        for i in range(len(sensor_times)):
            self.sensor_num_times.append(len(sensor_times[i]))
            self.lagrange_multipliers.append([])
            for j in range(len(sensor_times[i])):
                self.lagrange_multipliers[i].append(0)
        self.tree_size = len(sensor_locations)
        self.current_snake_locations = [(0., 0.) for _ in range(self.tree_size)]
        self.current_snake_times = [0 for _ in range(self.tree_size)]
        self.number_of_paths = 0
        self.tree = MyNode(-1, 0, None)
        if not empty_tree:
            self.tree_reset()

    def tree_reset(self):
        self.tree = MyNode(-1, 0, None)
        self.special_builder(self.tree, None, 0, 0)

    def convert_2D_matrix(self, paths, default_score=C.DEFAULT_SCORE):
        """
            converts a set of paths to a 2d score matrix with size number_of_inputted_paths + num_signals_in_sensot_depth
            padded with zeros
        """
        depth = len(paths[0])
        number_of_inputted_paths = len(paths)
        num_signals_in_sensor_depth = len(self.sensor_times[depth])
        matrix_size = number_of_inputted_paths + num_signals_in_sensor_depth
        if depth < self.tree_size:
            matrix = np.zeros((matrix_size, matrix_size))
            for row, column in itertools.product(range(matrix_size), repeat=2):
                matrix[row][column] = self.get_2d_matrix_entry(depth, row, column, paths, default_score=default_score)
            condensed_matrix = scc.condense_matrix(matrix, number_of_inputted_paths, num_signals_in_sensor_depth)
            return condensed_matrix
        raise Exception("REACHED END OF TREE")

    def get_2d_matrix_entry(self, depth, row, column, paths, default_score):
        number_of_inputted_paths = len(paths)
        num_signals_in_sensor_depth = len(self.sensor_times[depth])
        node_0 = self.get_node_from_path(list(np.zeros(depth)))
        if row < number_of_inputted_paths:
            node = self.get_node_from_path(paths[row])
        else:
            node = node_0
        if column < num_signals_in_sensor_depth:
            index = node.get_child_index_from_serial_number(column + 1)
        else:
            index = 0
        if index is None:
            return default_score
        else:  # row < number_of_inputted_paths or column < num_signals_in_sensor_depth:
            if column < num_signals_in_sensor_depth:
                return node.children[index].relaxed_value + self.lagrange_multipliers[depth][column]
            return node.children[index].relaxed_value
        # else:
        #     return 0

    def get_node_from_path(self, path) -> MyNode:
        node = self.tree
        for i in range(len(path)):
            index = node.get_child_index_from_serial_number(path[i])
            if index is None:
                raise Exception("PATH DOES NOT EXIST")
            node = node.children[index]
        return node

    def get_number_of_uses_from_paths_list(self, paths):
        depth = len(paths[0])
        number_of_uses = np.zeros(len(self.sensor_times[depth]), dtype=int)
        for path in paths:
            node = self.get_node_from_path(path)
            if node.favorite_child is not None and node.favorite_child != 0:
                number_of_uses[node.favorite_child - 1] += 1
        return number_of_uses

    # def get_number_of_uses_from_paths_lists(self, paths):
    #     number_of_uses = []
    #     for i in range(len(self.sensor_times)):
    #         number_of_uses.append([])
    #         for j in range(len(self.sensor_times[i])):
    #             number_of_uses[i].append(0)
    #
    #     for path in paths:
    #         node = self.get_node_from_path(path)
    #         depth = len(path)
    #         while node.favorite_child is not None:
    #             if node.favorite_child != 0:
    #                 number_of_uses[depth][node.favorite_child - 1] += 1
    #             index = node.get_child_index_from_serial_number(node.favorite_child)
    #             node = node.children[index]
    #             depth += 1
    #     return number_of_uses

    def get_lagrangian_multipliers_sum(self, starting_depth=0):
        summed_lagrangian_multipliers = 0
        for sensor_index in range(starting_depth, self.tree_size):
            lagrangian_vector = self.lagrange_multipliers[sensor_index]
            summed_lagrangian_multipliers += sum(lagrangian_vector)
        return summed_lagrangian_multipliers

    def update_tree_lagrangian_multipliers(self, gradients):

        for sensor_index in range(len(self.lagrange_multipliers)):
            sensor_lagrange_multipliers = self.lagrange_multipliers[sensor_index]
            sensor_gradient = gradients[sensor_index]
            for signal_index in range(len(sensor_lagrange_multipliers)):
                sensor_lagrange_multipliers[signal_index] += sensor_gradient[signal_index]
        self.__update_tree_helper(self.tree)

    def __update_tree_helper(self, root_Node: MyNode):
        if root_Node.depth + 1 < self.tree_size:
            for i in range(0, len(root_Node.children)):
                child = root_Node.children[i]
                self.__update_tree_helper(child)
        weight = 0
        if root_Node.serial_number > 0:
            weight = self.lagrange_multipliers[root_Node.depth][root_Node.serial_number - 1]
        root_Node.children_relaxed_min_value(self.tree_size, root_Node.depth, weight)

    # TODO: create_cases and do not rename
    def special_builder(self, root_Node: MyNode, emitter_location, current_sensor, size_of_snake):
        """
        This is the main function of the tree. in here we create the cost of all the *possible* NAKNIKIM
            param: root_Node - the pointer for the current branch
            param: emitter_location - if the current branch has more than 3 sensors then we can use this param
            to eliminate times that their hyperbola with the first time don't pass through this point. None otherwise
            param: current_sensor - recursion depth.
            param: size of snake - the current size of the snake.
        """
        if current_sensor < self.tree_size:
            root_Node.add_child(0, None)
            self.special_builder(root_Node.children[0], emitter_location, current_sensor + 1, size_of_snake)
            number_of_children = 1
            if size_of_snake == 0:
                for serial_number in range(len(self.sensor_times[current_sensor])):
                    time_of_arrival = self.sensor_times[current_sensor][serial_number]
                    self.current_snake_times[size_of_snake] = time_of_arrival
                    self.current_snake_locations[size_of_snake] = self.sensor_locations[current_sensor]

                    self.add_child(serial_number, time_of_arrival, root_Node, None, current_sensor,
                                   size_of_snake, number_of_children)
                    number_of_children += 1
            else:
                self.current_snake_locations[size_of_snake] = self.sensor_locations[current_sensor]
                # f = [case1, case2...][size_of_snake]
                for serial_number in range(len(self.sensor_times[current_sensor])):

                    ambiguity_solved = self.check_ambiguity_and_set_toa(current_sensor, size_of_snake, serial_number)
                    # new_time_of_arrival = self.sensor_times[current_sensor][serial_number]
                    new_time_of_arrival = self.current_snake_times[current_sensor]

                    if size_of_snake == 1 and ambiguity_solved:
                        self.add_child(serial_number, new_time_of_arrival, root_Node, None, current_sensor,
                                       size_of_snake, number_of_children)
                        number_of_children += 1

                    elif size_of_snake == 2 and ambiguity_solved:
                        emitter_next, adding_child = scc.three_sensor_intersection_scipy(
                            self.current_snake_locations[0:3], self.current_snake_times[0:3])
                        if adding_child:
                            self.add_child(serial_number, new_time_of_arrival, root_Node, emitter_next, current_sensor,
                                           size_of_snake, number_of_children)
                            number_of_children += 1

                    elif size_of_snake > 2 and ambiguity_solved:
                        cdt_new_to_first = C.C * (new_time_of_arrival - self.current_snake_times[0])
                        locations_tuple = (self.current_snake_locations[0], self.sensor_locations[current_sensor])
                        emitter_next, adding_child = scc.check_new_sensor_intersection_and_update_emitter_location(
                            emitter_loc=emitter_location, cdt_new_to_first=cdt_new_to_first, points=locations_tuple)
                        # emitter_next, adding_child = scc.check_new_sensor_and_update_emitter_location(
                        #     current_sensor, emitter_location, self.current_snake_locations, self.current_snake_times)
                        if adding_child:
                            self.add_child(serial_number, new_time_of_arrival, root_Node, emitter_next, current_sensor,
                                           size_of_snake, number_of_children)
                            number_of_children += 1
        else:
            self.number_of_paths += 1
        root_Node.children_min_value(self.tree_size, current_sensor, self.current_snake_locations[0:size_of_snake],
                                     self.current_snake_times[0:size_of_snake], emitter_location)
        root_Node.relaxed_value = root_Node.value

    def add_child(self, serial_number, time_of_arrival, root_Node, emitter, current_sensor, size_of_snake,
                  number_of_children):
        root_Node.add_child(serial_number + 1, time_of_arrival)
        self.special_builder(root_Node.children[number_of_children], emitter, current_sensor + 1,
                             size_of_snake + 1)

    def check_ambiguity_and_set_toa(self, current_sensor, size_of_snake, serial_number):
        self.current_snake_times[size_of_snake] = self.sensor_times[current_sensor][serial_number]
        sign = np.sign(self.current_snake_times[size_of_snake] - self.current_snake_times[0])
        if sign == 0.0:
            sign = 1
        warning_flag = False
        for ambiguity in [0, 1]:
            factor = ambiguity * sign
            time_of_arrival = self.sensor_times[current_sensor][serial_number] - factor * self.time_of_cycle
            cdt_new_to_first = C.C * (time_of_arrival - self.current_snake_times[0])
            distance = scc.distance(
                (self.current_snake_locations[0], self.sensor_locations[current_sensor]))
            if abs(cdt_new_to_first) < distance + scc.M_SIGMA * scc.SIGMA:
                # if ambiguity ==1:
                #     print("WTF")
                if warning_flag:
                    raise RuntimeWarning("TWO TIMES WORK FOR THE TIME SHIFT")
                self.current_snake_times[size_of_snake] = time_of_arrival
                warning_flag = True
        return warning_flag

    def get_max_child_path_and_times(self):
        node = self.tree
        path = []
        times = []
        while node.favorite_child is not None:
            path.append(node.favorite_child)
            favorite_child_index = node.get_child_index_from_serial_number(node.favorite_child)
            times.append(node.children[favorite_child_index].time_of_arrival)
            node = node.children[favorite_child_index]
        location = node.emitter_location
        return path, times, location

    def greedy_assigning(self, max_number_of_emitters):
        path = []
        pathogen = []
        times = []
        dummy_sensor_times = []
        emitter_locations = []
        self.greedy_assigning_remove_max(path, times, emitter_locations, pathogen, dummy_sensor_times, 0)
        dummy_tree = TreeOfData(self.sensor_locations, dummy_sensor_times, time_of_cycle=self.time_of_cycle)
        for iteration in range(1, max_number_of_emitters):
            dummy_tree.greedy_assigning_remove_max(path, times, emitter_locations, pathogen, None, iteration)
            dummy_tree.tree_reset()
        return pathogen, emitter_locations, times

    def greedy_assigning_remove_max(self, path, times, emitter_locations, pathogen, dummy_sensor_times, iteration):
        new_path, new_times, emitter_location = self.get_max_child_path_and_times()
        path.append(new_path[0:len(new_path)])
        times.append(new_times[0:len(new_times)])
        pathogen.append(new_path[0:len(new_path)])
        emitter_locations.append(emitter_location)
        for sensor_num in range(len(path[iteration])):
            if path[iteration][sensor_num] > 0:
                chosen_time = path[iteration][sensor_num]
                sliced_times_array = np.concatenate(
                    (self.sensor_times[sensor_num][0:chosen_time - 1],
                     self.sensor_times[sensor_num][chosen_time:len(self.sensor_times[sensor_num])]))
                if iteration == 0:
                    dummy_sensor_times.append(sliced_times_array)
                else:
                    self.sensor_times[sensor_num] = sliced_times_array
            elif iteration == 0:
                dummy_sensor_times.append(self.sensor_times[sensor_num][0:len(self.sensor_times[sensor_num])])
            if iteration > 0:
                path_across = [0 for _ in range(iteration + 1)]
                for delta in range(iteration + 1):
                    path_across[delta] = path[delta][sensor_num]
                pathogen[iteration][sensor_num] = scc.get_real_serial_number(path_across)
                # [path[u][sensor_num] for u in range(iteration + 1)][0:iteration + 1])

    def greedy_assigning_get_locations_and_heatmaps(self, max_number_of_emitters):
        pathogen, emitter_locations, times = self.greedy_assigning(max_number_of_emitters)
        locations_array = np.array(self.sensor_locations).T
        for k in range(len(pathogen)):
            print(f"the snake is \n{pathogen[k]}")
            # print(times[k])
            if emitter_locations[k] is None:
                print(None, end=" ")
                print(0, end="\n\n")
            elif scc.is_one_point(emitter_locations[k]):
                ttime = []
                llocations = []
                for i in range(len(times[k])):
                    if times[k][i] is not None:
                        ttime.append(times[k][i])
                        llocations.append(self.sensor_locations[i])
                locations_array = np.array(llocations).T
                time_array = np.array(ttime)
                # return -(len(sensors_loc) * (len(sensors_loc) - 1))

                data = scc.find_max_in_grid(locations_array, time_array, 100000)
                location = np.array([data[0], data[1]])
                args = (locations_array, time_array, C.SIGMA)
                bounds = ((location[0] - 500, location[0] + 500),
                          (location[1] - 500, location[1] + 500))
                # emitter_place = sci.minimize(scc.grid_score, location, args, bounds=bounds)
                emitter_place = sci.minimize(scc.grid_score, location, args)
                if len(ttime) > 3:
                    print(
                        f"the location of the string is{emitter_place.x}\nwith score: {emitter_place.fun} and precision {-2 * emitter_place.fun / ((len(ttime)) * (len(ttime) - 1))}\n")
                else:
                    print(
                        f"the location of the string is{emitter_place.x}\nwith score: {emitter_place.fun} and 3 sensor precision\n")
                # if len(ttime) > 3:
                #     print(
                #         f"the location of the string is{location}\nwith score: {data[2]} and precision {-2 * data[2] / ((len(ttime)) * (len(ttime) - 1))}\n")
                # else:
                #     print(
                #         f"the location of the string is{location}\nwith score: {data[2]} and 3 sensor precision\n")

                emitter_locations[k] = emitter_place.x

        return pathogen, emitter_locations, times

    def get_heatmap_for_snake(self, path):
        ttime = []
        llocations = []
        for i in range(len(path)):
            if path[i] > 0:
                ttime.append(self.sensor_times[i][path[i] - 1])
                llocations.append(self.sensor_locations[i])
        locations_array = np.array(llocations).T
        time_array = np.array(ttime)
        # return -(len(sensors_loc) * (len(sensors_loc) - 1))
        if len(ttime) > 3:
            data, grid = scc.find_max_in_grid(locations_array, time_array, 100000)
            location = np.array([data[0], data[1]])
            args = (locations_array, time_array, C.SIGMA)
            bounds = ((location[0] - 1000, location[0] + 1000),
                      (location[1] - 1000, location[1] + 1000))
        # emitter_place = sci.minimize(scc.grid_score, location, args, bounds=bounds)
            emitter_place = sci.minimize(scc.grid_score, location, args)
            if len(ttime) > 3:
                print(
                    f"the location of the string is{emitter_place.x}\nwith score: {emitter_place.fun} and precision {-2 * emitter_place.fun / ((len(ttime)) * (len(ttime) - 1))}\n")
            else:
                print(
                    f"the location of the string is{emitter_place.x}\nwith score: {emitter_place.fun} and 3 sensor precision\n")
            return grid, location
        return None, (None, None)
    def test_Tree(self):
        while True:
            path_str = input(f"Enter the path of length {self.tree_size}. if you want to exit type N: ")
            if path_str == "N":
                break
            path = [int(part) for part in path_str.split()]

            try:
                node = self.get_node_from_path(path)
                node_2 = self.tree
                times = []
                counter = 0
                while node_2.favorite_child is not None:
                    child_index = node_2.get_child_index_from_serial_number(path[counter])
                    times.append(node_2.children[child_index].time_of_arrival)
                    node_2 = node_2.children[child_index]
                    location = node_2.emitter_location
                print(f"path is {path}")
                print(f"path score is {node.value}")
                print(f"approximate location is {node.emitter_location}")
                create_heat_map = input(f"Do you want to see a heat map? Y/N: ")
                while create_heat_map not in ["Y", "N"]:
                    create_heat_map = input(f"Do you want to see a heat map? Y/N: ")
                if create_heat_map == "Y":
                    ttime = []
                    llocations = []
                    for i in range(len(times)):
                        if times[i] is not None:
                            ttime.append(times[i])
                            llocations.append(self.sensor_locations[i])
                    locations_array = np.array(llocations).T
                    time_array = np.array(ttime)
                    data = scc.find_max_in_grid(locations_array, time_array, 100000)
                    print(f"approximate location is {(data[0], data[1])}")
                    print(f"path score is {data[2]}")

                create_hyperbolas_map = input(f"Do you want to see a Hyperbola Intersection? Y/N: ")
                while create_hyperbolas_map not in ["Y", "N"]:
                    create_hyperbolas_map = input(f"Do you want to see a Hyperbola Intersection? Y/N: ")
                if create_hyperbolas_map == "Y":
                    graphed_sensors = input(f"what inputs do you want?: ")
                    graphed = [int(part) for part in graphed_sensors.split()]
                    ttime = []
                    llocations = []
                    for i in range(len(times)):
                        if times[i] is not None and i in graphed:
                            ttime.append(times[i])
                            llocations.append(self.sensor_locations[i])
                    locations_array = np.array(llocations)
                    time_array = np.array(ttime)
                    scc.Test_DrawHyperbolas(locations_array, time_array, node.emitter_location, 100000)
            except:
                print("PATH DOES NOT EXIST")


if __name__ == '__main__':
    if True:
        # pointss = [(10000, 5000), (-10000, 5000), (50000, -5000), (-10000, -5000), (20000, 20000), (-25000, 20000)]
        # sensor_locs = (
        #     (-10000, 0), (30000, 0), (50000, 5000), (-50000, 5000), (10000, 10000), (20000, -30000), (-3000, 7000),
        #     (-10000, 10000), (-20000, -30000), (-3000, -8000), (40000, 16000), (13000, 39000), (-14000, -15000),
        #     (17000, 21478))

        pointss = [(50000, 65000), (60000, 55000), (59000, 80000), (70000, 70000), (93000, 55000), (91000, 61000)]
        sensor_locss = [(-10000, 0), (-30000, 0), (-50000, 5000), (-50000, -55000), (10000, 10000), (20000, -30000),
                        (-3000, 7000), (-10000, 10000), (-20000, -30000), (-3000, -8000), (40000, 16000),
                        (13000, 39000), (-14000, -15000), (17000, 21478)]
    sensor_ts = []
    weights = []
    for j in range(1):
        points = pointss  # [0:1]
        sensor_locs = sensor_locss[0:10]
        emitter_random_start = np.random.random(len(points)) * C.TIME_OF_CYCLE
        print(emitter_random_start)
        emitter_random_start = np.zeros(len(points))
        for i in range(len(sensor_locs)):
            sensor_loc = sensor_locs[i]
            a = np.random.normal(0, C.SIGMA / 3, 1)[0]
            # a=0
            print(a)
            b = []
            probability = 1
            boolean_vector = np.random.choice([False, True], size=len(points), p=[1 - probability, probability])
            for h in range(len(points)):
                if boolean_vector[h]:
                    b.append((scc.distance((points[h], sensor_loc)) + a) / C.C - emitter_random_start[h])
                    if b[-1] < 0:
                        print(f"emitter {h} at sensor {i} ALIASED")
                        b[-1] += C.TIME_OF_CYCLE
            sensor_ts.append(b)
        start_time = time.time()
        tree = TreeOfData(sensor_locs, sensor_ts)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        print("Success")
        print(f"test for {len(sensor_locs)} sensors and {len(points)} emitters")
        print(f"The Number of Paths in this tree is {tree.number_of_paths}")

        # tree.update_tree_lagrangian_multipliers()
        # tree.greedy_assigning_get_locations_and_heatmaps(8)
