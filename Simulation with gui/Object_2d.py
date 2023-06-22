import numpy as np
import scipy.constants
import numpy.typing as npt

class Object_2d():


    def __init__(self, location):
        self.location = location

    def x_location(self):
        return self.location[0]

    def y_location(self):
        return self.location[1]

    def get_location(self):
        return self.location

    def distance(self, obj):
        return np.sqrt(np.square(self.location[0] - obj.x_location())+np.square(self.location[1] - obj.y_location()))

    def distance_to_point(self, point):
        return np.sqrt(np.square(self.location[0] - point[0])+np.square(self.location[1] - point[1]))

    def distance_list(self, obj_list):
        distance_lst = [self.distance(obj_2d) for obj_2d in obj_list]
        return np.array(distance_lst)