from Sensor import Sensor
import math
from scipy import signal
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import numpy as np

c=3*10**8

class Hyperbole():

    @staticmethod
    def intersect_hyperbolas(hyperbolas, xrange=np.linspace(-5, 5, 10000)):

        eqns = []
        x = xrange
        for hyperbola in hyperbolas:
            a, b, h, k, theta = hyperbola
            # Transform the coordinates using the rotation matrix
            y = b * np.sqrt(np.square(x) / a ** 2 + 1)
            x_rotated = (x - h) * np.cos(theta) - (y - k) * np.sin(theta)
            y_rotated = (x - h) * np.sin(theta) + (y - k) * np.cos(theta)
            eqn = (x_rotated, y_rotated)
            eqns.append(eqn)

        intersections = []
        for i in range(len(eqns)):
            for j in range(0, i):
                first_hyperbola = eqns[i]
                second_hyperbola = eqns[j]

                plt.plot(first_hyperbola[0], first_hyperbola[1], '-')
                plt.plot(second_hyperbola[0], second_hyperbola[1], '-')

                first_line = LineString(np.column_stack(first_hyperbola))
                second_line = LineString(np.column_stack(second_hyperbola))
                intersection = first_line.intersection(second_line)
                intersections.append(intersection)
                if intersection.geom_type == 'MultiPoint':
                    plt.plot(*LineString(intersection).xy, 'o')
                elif intersection.geom_type == 'Point':
                    plt.plot(*intersection.xy, 'o')
        plt.show()
        # Plot the hyperbolas and the intersection
        return intersections

    @staticmethod
    def create_hyperbole(loc1, loc2, dt):
        pass

    @staticmethod
    def get_intersect(hyp1, hyp2):
        pass

    def get_coordinates(self, theta):
        r=self.eccentricity*self.distance/(1-self.eccentricity*math.cos(math.radians(theta)))

    def __init__(self, sensor1, sensor2):
        """
        were defining an hyperbola by its midpoint and locus
        :param sensor1:
        :param sensor2:
        """
        self.dt=Sensor.correlation(sensor1,sensor2)
        self.center=((sensor1.get_location[0]+sensor2.get_location[0])/2, (sensor1.get_location[1]+sensor2.get_location[1])/2)
        self.locus = sensor1.get_location()
        if self.dt<0:
            self.locus=sensor2.get_location()
        self.angle=math.atan((self.center[1]-self.locus[1])/(self.center[0]-self.locus[0]))
        if (self.center[0]-self.locus[0])<0:
            self.angle+=math.pi
        self.alpha=c*self.dt
        self.distance=sensor1.distance(sensor2.get_location())
        self.gamma=math.sqrt(self.distance**2-self.alpha**2)
        self.eccentricity=abs(self.distance/self.alpha)




