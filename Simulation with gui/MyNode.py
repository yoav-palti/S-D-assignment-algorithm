import ScoreCalculation as scc
import Constants as C


class MyNode:

    def __init__(self, depth, serial_number, time_of_arrival, value=C.DEFAULT_SCORE):
        """
            field: depth is the sensor that this node represents
        """
        self.depth = depth
        self.serial_number = serial_number
        self.value = value
        self.relaxed_value = value
        self.children = []
        self.time_of_arrival = time_of_arrival
        self.favorite_child = None
        self.children_dictionary = {}
        self.emitter_location = None

    def add_child(self, serial_number, time_of_arrival):
        self.children_dictionary[serial_number] = len(self.children)
        self.children.append(MyNode(self.depth + 1, serial_number, time_of_arrival))

    def get_child_serial_number_from_index(self, index):
        if index < len(self.children):
            return self.children[index].serial_number
        return None

    def get_child_index_from_serial_number(self, serial_number):
        return self.children_dictionary.get(serial_number, None)

    def children_min_value(self, tree_size, current, locations, times, emitter_location):
        if current < tree_size:
            self.favorite_child, self.value = min(
                enumerate([self.children[i].value for i in range(len(self.children))]), key=lambda x: x[1])
            self.favorite_child = self.get_child_serial_number_from_index(self.favorite_child)
        else:
            val, location = scc.snake_score(locations, times, emitter_location)
            self.value = val
            self.emitter_location = location

    def children_relaxed_min_value(self, tree_size, current, weight):
        """
        the value increases with the weight
        """
        if current + 1 < tree_size:
            self.favorite_child, self.relaxed_value = min(
                enumerate([self.children[i].relaxed_value for i in range(len(self.children))]), key=lambda x: x[1])
            self.favorite_child = self.get_child_serial_number_from_index(self.favorite_child)
            self.relaxed_value = self.relaxed_value - weight
        else:
            self.relaxed_value = self.value - weight
