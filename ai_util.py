#!/usr/bin/env python

import glob
import os
import sys
from collections import deque
import math
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


"""
Returns correlation (with car heading), magnitude, sign
"""
def calculate_steering_error(destination,destination2,car_position,car_heading, lookahead_window = 10, max_forward_lookahead=10):
    target_vector = np.array([
        destination.x - car_position.x,
        destination.y - car_position.y
    ])
    magnitude = np.linalg.norm(target_vector)
    if magnitude == 0:
        return target_vector,0,-1

    if magnitude > lookahead_window:
       # Clamp the target_vector to a certain length
        target_vector = target_vector*(lookahead_window/magnitude)
    elif destination2 is not None:
        # Look ahead so as not to stop still at waypoints
        secondary_vector = np.array([
            destination2.x - destination.x,
            destination2.y - destination.y
        ])
        secondary_vector = secondary_vector/np.linalg.norm(secondary_vector)

        #Ease into the new heading, don't just turn offroad, see how well it correlates with the current heading
        forward_lookahead = min(lookahead_window - min(magnitude,lookahead_window),max_forward_lookahead)
        forward_correlation = np.dot(target_vector/magnitude,secondary_vector)
        target_vector = target_vector + forward_correlation*forward_lookahead*secondary_vector#*abs(forward_correlation)
        #target_vector = (magnitude*target_vector + forward_lookahead*secondary_vector)/(magnitude+forward_lookahead)
 


    # Compare to cars heading
    car_vector = np.array([car_heading.x,car_heading.y])
    corr = np.dot(target_vector,car_vector)/(
            math.hypot(car_vector[0],car_vector[1])*
            math.hypot(target_vector[0],target_vector[1])
        )
    sign = 1 if np.cross(target_vector,car_vector) < 0 else -1
    return (target_vector,math.acos(corr),sign)

def loc_string(l):
    return f"({l.x:.1f},{l.y:.1f})"

def fancy_lidar_string(num_data,offset=0):

    data = []
    for n in num_data:
        if n +offset > 6:
            data.append("")
        elif n+offset > 3:
            data.append("-")
        elif n+offset > 2.7:
            data.append("**")
        elif n+offset > 2.5:
            data.append("***")
        elif n +offset > 2.3:
            data.append( "OOO")
        elif n +offset> 2:
            data.append( "OOOOOO")
        else:
            data.append( "XXXXXXX")

    buf=f"\t\t{data[0]}\t\t\n\n"
    buf+=f"\t{data[7]}\t\t{data[1]}\t\n\n\n"
    buf+=f"{data[6]}\t\t*\t\t{data[2]}\n\n\n"
    buf+=f"\t{data[5]}\t\t{data[3]}\t\n\n"
    buf+=f"\t\t{data[4]}\t\t\n"
    return buf

def lidar_string(data):
    buf=f"\t\t{data[0]:.1f}\t\t\n\n"
    buf+=f"\t{data[7]:.1f}\t\t{data[1]:.1f}\t\n\n\n"
    buf+=f"{data[6]:.1f}\t\t*\t\t{data[2]:.1f}\n\n\n"
    buf+=f"\t{data[5]:.1f}\t\t{data[3]:.1f}\t\n\n"
    buf+=f"\t\t{data[4]:.1f}\t\t\n"
    return buf
