import numpy as np


def point_grades(point_list, grade_from_distances):

    grade_from_points = [grade_from_distances(point.distance_list(point_list)) for point in point_list]
    return grade_from_points

def grader(distances):
    dist_grades = np.maximum(distances, 200) #assuming we work in meters
    dist_grades = 200/np.square(dist_grades)
    return dist_grades

def point_threshold(point_list,grade_from_distances, threshhold):
    grade_from_points = point_grades(point_list, grade_from_distances)
    keep_point = grade_from_points > threshhold
    real_points = [point for idx, point in enumerate(point_list) if keep_point[idx]]
    return real_points