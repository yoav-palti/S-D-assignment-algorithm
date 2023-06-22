import math
import numpy as np
from tqdm import tqdm
import Constants as C
import matplotlib.pyplot as plt
class Grid():
    def __init__(self, scan_size, scan_len, sensor_list, emmiter_list, point_score_function, signal, sensor_time_error):
        self.scan_size = scan_size
        self.scan_len = scan_len
        self.sensor_list = sensor_list
        self.emmiter_list = emmiter_list
        self.point_score_function = point_score_function
        #gets the input (sensor_time_difference, sensor_time_difference_error)
        self.emmitor_signal = signal
        self.sensor_time_error = sensor_time_error

        self.borders = scan_borders # the borders (x_lower_border, x_upper_border, y_lower_border, y_upper_border)
        square_rib_size = np.sqrt((scan_borders[1] - scan_borders[0])*(scan_borders[3] - scan_borders[2]))
        self.step = square_rib_size/scan_size
        self.x_steps_size = math.ceil((scan_borders[1] - scan_borders[0])/self.step)
        self.y_steps_size = math.ceil((scan_borders[3] - scan_borders[2])/self.step)

        #self.step = self.scan_len / self.scan_size
        self.score_grid = np.zeros((self.y_steps_size, self.x_steps_size))
        self.sensor_time_difference_list = []

        # A dictionary of dictionaries, with emmiters as the keys, that maps them to the time differences
        # from the sensors.
        # self.sensor_time_differences = self.sensor_time_difference_init()
        # self.sensor_emmiter_matrix_time_differences = self.sensor_time_difference_matrix_init()

    def add_emitter_and_emit(self, emitter, emitter_action = None, additional_variable = None):
        self.emmiter_list.append(emitter)
        if emitter_action == 'random':
            emitter.emit_signal_random_block(self.sensor_list, additional_variable)
        if emitter_action == 'all':
            emitter.emit_signal(self.sensor_list)
        if emitter_action == 'first':
            #emitting the signal to the first <additional_variable> sensors in the sensor list.
            truncated_sensor_list = self.sensor_list[0:min(additional_variable, len(self.sensor_list))]
            emitter.emit_signal(truncated_sensor_list)

    def emit_signals(self):
        for emmiter in self.emmiter_list:
            emmiter.emit_signal(self.sensor_list)

    def emit_signals_random(self, probability):
        for emmiter in self.emmiter_list:
            emmiter.emit_signal_random_block(self.sensor_list, probability)

    """
    def sensor_time_difference_init(self):
        time_difference_dict = {}
        for sensor in self.sensor_list:
            sensor_time_difference_dict = {}
            for emmiter in self.emmiter_list:
                sensor_time_difference_dict[emmiter] = sensor.correlation_to_signal(sensor.signal, emmiter.signal)
            time_difference_dict[sensor] = sensor_time_difference_dict
        return time_difference_dict

    def sensor_time_difference_matrix_init(self):
        time_diff_mat = np.zeros((len(self.sensor_list), len(self.emmiter_list)))
        for idx_sen, sensor in enumerate(self.sensor_list):
            for idx_emm, emmiter in enumerate(self.emmiter_list):
                time_diff_mat[idx_sen][idx_emm] = self.sensor_time_differences[sensor][emmiter]
        return time_diff_mat
    """
    def calculate_sensor_time_diff(self):
        for sen_idx, sensor in enumerate(self.sensor_list):
            self.sensor_time_difference_list.append(sensor.correlation_to_signal(sensor.signal, self.emmitor_signal))

    def calculate_score(self):
        for i in tqdm(range(self.x_steps_size)):
            x = self.borders[0] + i * self.step
            for j in range(self.y_steps_size):
                y = self.borders[2] + j * self.step
                self.score_grid[j][i] = self.get_point_score((x, y))

    def get_point_score(self, point): #((x, y), sensor_list, sensor_time_difference)
        # This function takes a point on the grid (a tuple of x and y coordinates), a list of sensors sen_list, and a list of times sen_times at which the signal was received at each sensor.
        summed = 0
        for l in range(len(self.sensor_list)):
            for j in range(len(self.sensor_time_difference_list[l])): # sensor_time_difference
                for k in range(l, len(self.sensor_list)):
                    if k != l and len(self.sensor_time_difference_list[k]) != 0:
                        # For each pair of sensors, the function calculates the simulated time difference of arrival at the point and the real time difference of arrival, and adds the score for the difference to a running total.
                        dt_sim = (self.sensor_list[l].distance_to_point(point) - self.sensor_list[k].distance_to_point(point)) / C.C
                        dt = min(abs((self.sensor_time_difference_list[l][j] - self.sensor_time_difference_list[k][i]) - dt_sim) for i in range(len(self.sensor_time_difference_list[k])))
                        # dt_real = sen_times[l][j] - sen_times[k]
                        summed += self.point_score_function(dt, self.sensor_time_error)
            # The function returns the inverse of the total score.
        return summed

    def plot_emitters(self):
        for emitter in self.emmiter_list:
            plt.plot(*emitter.get_location(),'o',color = 'blue', alpha=0.4)

    def plot_sensors(self):
        for sensor in self.sensor_list:
            plt.plot(*sensor.get_location(),'o',color = 'purple')

    #will be changed to 'show scenario' later
    def heat_map(self, title, savename = None):
        x = np.linspace(self.borders[0], self.borders[1], self.x_steps_size + 1)[0:-1]
        y = np.linspace(self.borders[2], self.borders[3], self.y_steps_size + 1)[0:-1]
        print(len(x))
        print(len(y))
        print(len(self.score_grid[0]))
        print(len(self.score_grid))
        plt.contourf(x, y, self.score_grid, 20, cmap='RdGy')
        plt.colorbar()
        self.plot_emitters()
        self.plot_sensors()
        plt.title(title)
        if savename is not None:
            plt.savefig(savename + '.png')
        plt.show()


    def circle_equality_check(self, associated_timings, maximal_time_error) -> bool:
        """Checks if the timing associated with one target (p from the paper) fit with each other.
        ** TO FILL LATER
                Args:
                    associated_timings (list): list of the signal time measured by each sensor.
                    the length of the list is the number of sensors.
                    if the i'th sensor has no measure of the signal emitted by this emitter, the list will contain
                    None in the i'th place.

                    maximal_time_error: the maximal sensor time error

                Returns:
                    bool: true if the associated timings fit with each other, and false otherwise.
        """

    def probability_of_measurment_from_emmiter(self, associated_timing, emitter, time_error, time_error_std):
        """returns the probability to detect the specified associated timing from a known emitter.
            Args:
                associated_timings (list): list of the signal time measured by each sensor.
                the length of the list is the number of sensors.
                if the i'th sensor has no measure of the signal emitted by this emitter, the list will contain
                None in the i'th place.

                emitter: an emitter object specifying the signal and location of the emitter.

                time_error: the average of the gaussian time error.

                time_error_std: the standard deviation of the gaussian time error

            Returns:
                double - returns the probability of this state.
        """

    def likelihood_of_measurements_from_emitter(self, associated_timings, emitter, time_error, time_error_std):
        """returns the probability to detect the specified associated timings from a known emitter.
            Args:
                associated_timings (list): list of the signal time measured by each sensor.
                the length of the list is the number of sensors.
                if the i'th sensor has no measure of the signal emitted by this emitter, the list will contain
                None in the i'th place.

                emitter: an emitter object specifying the signal and location of the emitter.

                time_error: the average of the gaussian time error.

                time_error_std: the standard deviation of the gaussian time error

            Returns:
                double - returns the likelyhood of this state.
        """

    def grid_from_associated_timings(self, associated_timings, time_error, time_error_std):
        return

    def location_from_time_diff(self, two_dimentional_grid):
        """returns the location of the emitter that emitted the associated timings. the location should be calculated
        using maximal likelihood estimation, as specified in the paper. this is the equivalent of calculating the maximum
        of the "grid" function. A proposed way is using gradient decent.
        Args:
            two_dimentional_grid: two dimentional grid is a function of (x,y). two_dimentional_grid(x,y,)
            gives the score.

        Returns:
            emitter - returns an emitter with the location and signal calculated here
        """

