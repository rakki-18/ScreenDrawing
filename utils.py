import cv2
import math
def find_sum(list_points,k, axis):
    list_axis = [sub[axis] for sub in list_points]
    
    
    return sum(list_axis[-k:])

def add_coordinates(list_points, hand_landmarks):
    
    point = []
    point.append(int(hand_landmarks.landmark[8].x*800))
    point.append(int(hand_landmarks.landmark[8].y*800))
    point.append(int(hand_landmarks.landmark[8].z*512))
    
    list_points.append(point)
    
    return list_points

def find_average(list_points, k):
    avg_x = find_sum(list_points, k, axis = 0)/k
    avg_y = find_sum(list_points, k, axis = 1)/k
    avg_z = find_sum(list_points, k, axis = 2)/k
    return avg_x,avg_y,avg_z

def add_to_average_list(list_avg, avg_x, avg_y,avg_z):
    
    list_avg.append([int(avg_x),int(avg_y),int(avg_z)])
    
    return list_avg


def draw_line(canvas,list_avg,color,thickness):
    
    # get the current coordinates as endpoint
    end_point = (list_avg[-1][0], list_avg[-1][1])
    # get the previous coordinates as the start point
    start_point = (list_avg[-2][0], list_avg[-2][1])
    # draw a line joining these points
    cv2.line(canvas,start_point, end_point,color, thickness)
    return canvas


def draw_circle(cursor, list_avg, radius, color,thickness):
    # get the current coordinates as endpoint
    end_point = (list_avg[-1][0], list_avg[-1][1])
    cv2.circle(cursor,end_point,radius,color,thickness)
    
    return cursor
def find_distance(index, thumb):
    distance = (index.x - thumb.x)**2+ (index.y - thumb.y)**2
    distance = math.sqrt(distance)
    distance = distance*800
    return distance
