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
        target_vector = target_vector + forward_correlation*forward_lookahead*secondary_vector
        #target_vector = (magnitude*target_vector + forward_lookahead*secondary_vector)/(magnitude+forward_lookahead)
 


    # Compare to cars heading
    car_vector = car_heading.get_forward_vector()
    car_vector = np.array([car_vector.x,car_vector.y])

    corr = np.dot(target_vector,car_vector)/(
            math.hypot(car_vector[0],car_vector[1])*
            math.hypot(target_vector[0],target_vector[1])
        )
    sign = 1 if np.cross(target_vector,car_vector) < 0 else -1
    return (target_vector,math.acos(corr),sign)

def spawn_waypoint_marker(world, actor_list, blueprint, transform, recursion=0):
    cone = world.try_spawn_actor(blueprint, transform)
    if cone is not None:
        actor_list.append(cone)
        print('spawned %r at %s' % (cone.type_id, transform.location))
    else:
        if recursion > 20:
            print('WARNING: vehicle not spawned, NONE returned')
        else:
            return spawn_waypoint_marker(world, actor_list, cone ,transform, recursion+1)
    return cone