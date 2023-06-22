from Grid import Grid
from Sensor import Sensor
import random
from Emmiter import Emmiter
from Signal import Signal
import time
import ScoreCalculation as scc
from Tree_of_Data import TreeOfData
import itertools
import numpy as np
import Constants as C


class Testing():
    def __init__(self, score_function):
        self.score_function = score_function

    def random_emitters(self, emitter_borders, signal):
        """
        Generate a list of random emitters within the specified borders.

        Args:
            emitter_borders (List[Tuple[float, float, float, float]]): A list of tuples, each representing the borders of an emitter's position as (x_min, x_max, y_min, y_max).
            signal (Signal): The signal transmitted by the emitters.

        Returns:
            List[Emitter]: A list of randomly generated emitters, each with a position within the specified borders, the given signal, and a random time delay.
        """

        emitter_number = len(emitter_borders)
        emitter_list = []

        for i in range(emitter_number):
            border_x = emitter_borders[i][0:2]
            border_y = emitter_borders[i][2:4]

            # Generate a random position within the specified borders using a triangular distribution.
            x = random.triangular(border_x[0], border_x[1])
            y = random.triangular(border_y[0], border_y[1])

            # Create a new emitter with the random position, given signal, and a random time delay.
            emitter = Emmiter((x, y), signal, 1, dt=0.01 * random.random())
            emitter_list.append(emitter)

        return emitter_list

    def random_sensors(self, sensor_borders, sensor_time_error, triangle_x=1 / 2, triangle_y=1 / 2):
        """
        Generate random sensors within the specified borders.

        Args:
        - sensor_borders: a list of tuples representing the borders of the sensors, where each tuple contains 4 elements in the order of (x_min, x_max, y_min, y_max)
        - sensor_time_error: the time error of the generated sensors
        - triangle_x: optional parameter for specifying the peak location of the triangular distribution along the x-axis. Default value is 1/2.
        - triangle_y: optional parameter for specifying the peak location of the triangular distribution along the y-axis. Default value is 1/2.

        Returns:
        - A list of Sensor objects generated within the specified borders.
        """

        sensor_number = len(sensor_borders)
        sensor_list = []

        for i in range(sensor_number):
            # Extract borders for each sensor
            border_x = sensor_borders[i][0:2]
            border_y = sensor_borders[i][2:4]

            # Generate random x,y coordinates within the specified borders using triangular distribution
            x = random.triangular(border_x[0], border_x[1], triangle_x)
            y = random.triangular(border_y[0], border_y[1], triangle_y)

            # Create a new sensor object with the generated coordinates and time error
            sensor = Sensor(location=(x, y), time_err=sensor_time_error)
            sensor_list.append(sensor)

        return sensor_list

    def create_scenario_red(self, emitter_num, sensor_num, plot_save_str = None):
        q = -6
        sensor_time_error = 10 ** q
        SCAN_SIZE = 2 * 10 ** 2
        scan_borders = (-50000, 50000, -50000, 50000)

        emitter_signal = 1000000 * Signal.random_signal().one_cycle_signal

        emitter_borders = emitter_num * [(-50000, 50000, -50000, 50000)]
        emitter_list = self.random_emitters(emitter_borders, emitter_signal)
        sensor_borders = int(np.ceil(sensor_num / 2)) * [(25000, 50000, -50000, 50000), (-50000, 50000, 25000, 50000)]
        sensor_borders = sensor_borders[0:sensor_num]
        sensor_list = self.random_sensors(sensor_borders, sensor_time_error)


        test_grid = Grid(SCAN_SIZE, scan_borders, sensor_list, emitter_list, self.score_function, emitter_list[0].get_signal(),
                         sensor_time_error)
        test_grid.emit_signals_random(1 / 2)

        test_grid.calculate_sensor_time_diff()
        test_grid.calculate_score()
        if plot_save_str is not None:
            test_grid.heat_map(title='Red Scenario of ' + str(emitter_num) + ' emitters ' +
                                     str(sensor_num) + ' sensors.', savename=plot_save_str)
        else:
            test_grid.heat_map(title='Red Scenario of ' + str(emitter_num) + ' emitters ' +
                                     str(sensor_num) + ' sensors.')

        return

    def create_scenario_blue(self, emitter_num, sensor_num, plot_save_str = None):
        q = -6
        sensor_time_error = 10 ** q
        SCAN_SIZE = 2 * 10 ** 2
        scan_borders = (-100000, 100000, -50000, 50000)

        emitter_signal = 1000000 * Signal.random_signal().one_cycle_signal

        emitter_borders = emitter_num * [(-100000, 0, -50000, 50000)]
        emitter_list = self.random_emitters(emitter_borders, emitter_signal)
        sensor_borders = int(np.ceil(sensor_num / 2)) * [(20000, 50000, -50000, 50000), (20000, 100000, -50000, 50000)]
        sensor_list = self.random_sensors(sensor_borders[0:sensor_num], sensor_time_error)

        test_grid = Grid(SCAN_SIZE, scan_borders, sensor_list, emitter_list, self.score_function, emitter_list[0].get_signal(),
                         sensor_time_error)
        test_grid.emit_signals_random(1 / 2)

        test_grid.calculate_sensor_time_diff()
        test_grid.calculate_score()
        if plot_save_str is not None:
            test_grid.heat_map(title='Blue Scenario of ' + str(emitter_num) + ' emitters ' +
                                     str(sensor_num) + ' sensors.', savename=plot_save_str)
        else:
            test_grid.heat_map(title='Blue Scenario of ' + str(emitter_num) + ' emitters ' +
                                     str(sensor_num) + ' sensors.')

    def create_scenario_green(self, emitter_num, sensor_num, number_receivers_of_isolated, plot_save_str = None):
        q = -6
        sensor_time_error = 10 ** q
        SCAN_SIZE = 2 * 10 ** 2
        scan_borders = (-100000, 100000, -50000, 50000)

        emitter_signal = 1000000 * Signal.random_signal().one_cycle_signal

        emitter_borders = (emitter_num - 1) * [(-100000, 0, -50000, 50000)]
        emitter_list = self.random_emitters(emitter_borders, emitter_signal)

        # creating the isloated emitter of the green scenario
        isolated_emitter_borders = [(-100000, -80000, -50000, -30000)]
        isolated_emitter_list = self.random_emitters(isolated_emitter_borders, emitter_signal)
        isolated_emitter = isolated_emitter_list[0]

        sensor_borders = int(np.ceil(sensor_num / 2)) * [(20000, 50000, -50000, 50000), (20000, 100000, -50000, 50000)]
        sensor_list = self.random_sensors(sensor_borders[0:sensor_num], sensor_time_error)

        test_grid = Grid(SCAN_SIZE, scan_borders, sensor_list, emitter_list, self.score_function, emitter_list[0].get_signal(),
                         sensor_time_error)
        test_grid.emit_signals_random(1 / 2)

        if number_receivers_of_isolated >= len(sensor_list):
            print('Warning. The isloated emitter is not really isolated')
        test_grid.add_emitter_and_emit(isolated_emitter, 'first', number_receivers_of_isolated)

        test_grid.calculate_sensor_time_diff()
        test_grid.calculate_score()
        if plot_save_str is not None:
            test_grid.heat_map(title='Green Scenario of ' + str(emitter_num) + ' emitters ' +
                                     str(sensor_num) + ' sensors, and \n' + str(number_receivers_of_isolated)
                                     + ' of them receive the isolated sensor.', savename=plot_save_str)
        else:
            test_grid.heat_map(title='Green Scenario of ' + str(emitter_num) + ' emitters ' +
                                     str(sensor_num) + ' sensors, and \n ' + str(number_receivers_of_isolated)
                                     + ' of them receive the isolated sensor.')


    def create_scenario_yellow(self, emitter_num, sensor_num, number_receivers_of_isolated, plot_save_str = None):
        q = -6
        sensor_time_error = 10 ** q
        SCAN_SIZE = 2 * 10 ** 2
        scan_borders = (-100000, 100000, -50000, 50000)

        emitter_signal = 1000000 * Signal.random_signal().one_cycle_signal

        emitter_borders = (emitter_num-1) * [(-100000, 0, -20000, 50000)]
        emitter_list = self.random_emitters(emitter_borders, emitter_signal)
        sensor_borders = int(np.ceil(sensor_num / 2)) * [(20000, 50000, -50000, 50000), (20000, 100000, -50000, 50000)]

        sensor_list = self.random_sensors(sensor_borders[0:sensor_num], sensor_time_error)

        # creating the isloated emitter of the yellow scenario
        isolated_emitter_borders = [(-100000, -80000, -50000, -30000)]
        isolated_emitter_list = self.random_emitters(isolated_emitter_borders, emitter_signal)
        isolated_emitter = isolated_emitter_list[0]

        test_grid = Grid(SCAN_SIZE, scan_borders, sensor_list, emitter_list, self.score_function, emitter_list[0].get_signal(),
                         sensor_time_error)
        test_grid.emit_signals_random(1 / 2)

        if number_receivers_of_isolated >= len(sensor_list):
            print('Warning. The isloated emitter is not really isolated')
        test_grid.add_emitter_and_emit(isolated_emitter, 'first', number_receivers_of_isolated)

        test_grid.calculate_sensor_time_diff()
        test_grid.calculate_score()
        if plot_save_str is not None:
            test_grid.heat_map(title='Yellow Scenario of ' + str(emitter_num) + ' emitters ' +
                                     str(sensor_num) + ' sensors, and \n' + str(number_receivers_of_isolated)
                                     + ' of them receive the isolated sensor.', savename=plot_save_str)
        else:
            test_grid.heat_map(title='Yellow Scenario of ' + str(emitter_num) + ' emitters ' +
                                     str(sensor_num) + ' sensors, and \n ' + str(number_receivers_of_isolated)
                                     + ' of them receive the isolated sensor.')

    def create_tree(self):
        pointss = [(50000, 65000), (60000, 55000), (59000, 80000), (70000, 70000), (93000, 55000), (91000, 61000)]
        sensor_locss = (
            (-10000, 0), (-30000, 0), (-50000, 5000), (-50000, -55000), (10000, 10000), (20000, -30000), (-3000, 7000),
            (-10000, 10000), (-20000, -30000), (-3000, -8000), (40000, 16000), (13000, 39000), (-14000, -15000),
            (17000, 21478))
        sensor_ts = []
        points = pointss[0:4]  # [0:1]
        sensor_locs = sensor_locss[0:7]
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
        print(f"The tree was created in: {execution_time} seconds")
        return tree

    def create_assignment_scenario(self, number_of_sensors, number_of_emitters, probability = 1, pointss = None, sensor_locss = None):
        if pointss is None:
            pointss = [(50000, 65000), (60000, 55000), (59000, 80000), (70000, 70000), (93000, 55000), (91000, 61000)]
        if sensor_locss is None:
            sensor_locss = (
                (-10000, 0), (-30000, 0), (-50000, 5000), (-50000, -55000), (10000, 10000), (20000, -30000), (-3000, 7000),
                (-10000, 10000), (-20000, -30000), (-3000, -8000), (40000, 16000), (13000, 39000), (-14000, -15000),
                (17000, 21478))
        sensor_ts = []
        points = pointss[0:min(number_of_emitters,6)]  # [0:1]
        sensor_locs = sensor_locss[0:min(number_of_sensors,14)]
        # emitter_random_start = np.random.random(len(points)) * C.TIME_OF_CYCLE
        # print(emitter_random_start)
        emitter_random_start = np.zeros(len(points))
        for i in range(len(sensor_locs)):
            sensor_loc = sensor_locs[i]
            a = np.random.normal(0, C.SIGMA/20, 1)[0]
            # a=0
            # print(a)
            b = []
            boolean_vector = np.random.choice([False, True], size=len(points), p=[1 - probability, probability])
            for h in range(len(points)):
                if boolean_vector[h]:
                    b.append((scc.distance((points[h], sensor_loc)) + a) / C.C - emitter_random_start[h])
                    if b[-1] < 0:
                        print(f"emitter {h} at sensor {i} ALIASED")
                        b[-1] += C.TIME_OF_CYCLE
            sensor_ts.append(np.array(b))
        return sensor_locs, sensor_ts

    def create_specific_tree_3d_scenario(self):
        from MyNode import MyNode
        pointss = [(50000, 65000), (60000, 55000), (59000, 80000), (70000, 70000), (93000, 55000), (91000, 61000)]
        sensor_locss = (
            (-10000, 0), (-30000, 0), (-50000, 5000), (-50000, -55000), (10000, 10000), (20000, -30000), (-3000, 7000),
            (-10000, 10000), (-20000, -30000), (-3000, -8000), (40000, 16000), (13000, 39000), (-14000, -15000),
            (17000, 21478))
        sensor_ts = []
        points = pointss[0:3]
        sensor_locs = sensor_locss[0:3]
        emitter_random_start = np.random.random(len(points)) * C.TIME_OF_CYCLE
        print(emitter_random_start)
        emitter_random_start = np.zeros(len(points))
        for i in range(len(sensor_locs)):
            sensor_loc = sensor_locs[i]
            # a = np.random.normal(0, C.SIGMA / 3, 1)[0]
            a=0
            # print(a)
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
        tree = TreeOfData(sensor_locs, sensor_ts, empty_tree=True)
        end_time = time.time()
        tree.tree = MyNode(-1, 0, None)
        cost_mat = self.create_cost_mat()
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    if cost_mat[i, j, k] != 0:
                        tree.tree.add_child((i, j, k), None)
            tree.tree.add_child(i, None)

        execution_time = end_time - start_time
        print(f"The tree was created in: {execution_time} seconds")
        return tree

    def create_tree_for_book_example(self):
        tree = TreeOfData([0,0,0],[[0,0],[0,0],[0,0]], empty_tree=True)
        the_tree = tree.tree
        temp_0 = tree.tree

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
        for i in range(3):
            the_tree.add_child(i, 0)
            temp_0 = the_tree.children[i]
            for j in range(3):
                temp_0.add_child(j, 0)
                temp_1 = temp_0.children[-1]
                for k in range(3):
                    if cost_mat[i,j,k]!=0:
                        temp_1.add_child(k, 0)
                        temp_1.children[-1].value = cost_mat[i,j,k]
                        temp_1.children[-1].relaxed_value = cost_mat[i,j,k]
                    elif (i!=0 and j!=0) or (i!=0 and k!=0) or (j!=0 and k!=0):
                        pass
                    else:
                        temp_1.add_child(k, 0)
                        temp_1.children[-1].value = 0
                        temp_1.children[-1].relaxed_value = 0
                temp_1.children_min_value(3, 1,[],[],None)
                temp_1.relaxed_value = temp_1.value

            temp_0.children_min_value(3, 0, [], [], None)
            temp_0.relaxed_value = temp_0.value
        the_tree.children_min_value(3, -1, [], [], None)
        the_tree.relaxed_value = the_tree.value
        print("Hi")
        return tree
