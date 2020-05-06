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
def calculate_steering_error(destination,car_position,car_heading, speed_balance_point =10):
    target_vector = np.array([
        destination.x - car_position.x,
        destination.y - car_position.y
    ])

    magnitude = np.linalg.norm(target_vector)
    if magnitude > speed_balance_point:
        # Clamp the target_vector to a certain length
        target_vector = target_vector*(speed_balance_point/magnitude)

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