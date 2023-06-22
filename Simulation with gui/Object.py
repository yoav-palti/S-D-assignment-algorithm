import math
import typing



class Something(object):

    """
    this is the object calss
    in this class there are two types of oblects
    transmitter and receiver
    both have a location


    """
    def __init__(self,location=[0,0]):
        self.location=location
        print("created new Object")

    def calc_distance(self, second):
        return math.sqrt((self.location[0]-second.location[0])**2+(self.location[1]-second.location[1])**2)

    def get_x(self):
        return self.location[0]

    def get_y(self):
        return self.location[1]

class Emmitter(Something):

    def __init__(self, location):
        super().__init__(self, location)
        print("it's a transmitter")


        #print(self.location)

loc=[1,1]
trans = Emmitter(loc)
print(trans.location)

