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
import ai_knowledge as data
import ai_pid as pid
import ai_util as util
from ai_knowledge import Status

# Executor is responsible for moving the vehicle around
# In this implementation it only needs to match the steering and speed so that we arrive at provided waypoints
# BONUS TODO: implement different speed limits so that planner would also provide speed target speed in addition to direction
class Executor(object):
  def __init__(self, knowledge, vehicle):
    self.vehicle = vehicle
    self.knowledge = knowledge
    self.target_pos = knowledge.get_location()

    self.throttle_control = pid.Controller(0.24,-0.4,0.000001, name="Throttle",debug=False)
    self.steering_control = pid.Controller(0.5,-0.6,0.000001, i_damp=0.6, name="Steering",debug=False)
    
  #Update the executor at some intervals to steer the car in desired direction
  def update(self, time_elapsed):
    status = self.knowledge.get_status()
    #TODO: this needs to be able to handle
    if status == Status.DRIVING:
      dest = self.knowledge.get_current_destination()
      next = self.knowledge.get_next_destination()
      self.update_control(dest, [11,11,next], time_elapsed)

  # TODO: steer in the direction of destination and throttle or brake depending on how close we are to destination
  # TODO: Take into account that exiting the crash site could also be done in reverse, so there might need to be additional data passed between planner and executor, or there needs to be some way to tell this that it is ok to drive in reverse during HEALING and CRASHED states. An example is additional_vars, that could be a list with parameters that can tell us which things we can do (for example going in reverse)
  def update_control(self, destination, additional_vars, delta_time):
    speed_target = additional_vars[0]
    predictive_lookahead = additional_vars[1]
    forward_waypoint = additional_vars[2]

    car_position = self.knowledge.get_location()
    car_heading = self.knowledge.get_heading()
    car_velocity = self.knowledge.get_velocity()
    car_velocity = np.array([car_velocity.x,car_velocity.y])

    ### Calculate errors
    step_target_vector, magnitude, sign = util.calculate_steering_error(destination,forward_waypoint,car_position,car_heading,lookahead_window=speed_target, max_forward_lookahead=predictive_lookahead)

    #Steering error is the angle between the vector to the destination and the current car heading
    #Dist error is the distance to the target

    forward_delta = 1
    velocity_error = step_target_vector - car_velocity*forward_delta

    #velocity_correlation is how well the error (signal) aligns with the current velocity
    velocity_correlation = np.dot(step_target_vector/np.linalg.norm(step_target_vector),velocity_error/np.linalg.norm(velocity_error))

    throttle_signal = self.throttle_control.step(np.linalg.norm(velocity_error),delta_time)*velocity_correlation*abs(velocity_correlation)#+forward_throttle_correlation)
    steering_signal = self.steering_control.step(magnitude*sign,delta_time)

    brake_signal = 0

    ### Calculate throttle and brake
    if throttle_signal < 0:
      brake_signal = -throttle_signal
      throttle_signal = 0
    elif throttle_signal < 0.3:
      brake_signal = 0.3 -throttle_signal

    # Set control values
    control = carla.VehicleControl()
    control.throttle = throttle_signal
    control.brake = brake_signal
    control.hand_brake = False
    control.steer = steering_signal
    self.vehicle.apply_control(control)

# Planner is responsible for creating a plan for moving around
# In our case it creates a list of waypoints to follow so that vehicle arrives at destination
# Alternatively this can also provide a list of waypoints to try avoid crashing or 'uncrash' itself
class Planner(object):
  def __init__(self, knowledge):
    self.knowledge = knowledge
    self.path = deque([])


  # Create a map of waypoints to follow to the destination and save it
  def make_plan(self, source, destination):
    self.path = self.build_path(source,destination)
    self.update(0)
  
  # Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    self.update_plan()
    self.knowledge.update_destination(self.get_current_destination(),self.get_next_destination())
  
  #Update internal state to make sure that there are waypoints to follow and that we have not arrived yet
  def update_plan(self):
    if self.knowledge.arrived_at(self.path[0]):
      self.path.popleft()
    
    if len(self.path) == 0:
      self.knowledge.update_status(Status.ARRIVED)
    else:
      self.knowledge.update_status(Status.DRIVING)

  #get current destination 
  def get_current_destination(self):
    status = self.knowledge.get_status()
    #if we are driving, then the current destination is next waypoint
    if status == Status.DRIVING:
      #TODO: Take into account traffic lights and other cars
      traffic_light_path =  self.knowledge.get_legal_path()
      if traffic_light_path is not None:
        return traffic_light_path

      return self.path[0]
    if status == Status.ARRIVED:
      return self.knowledge.get_location()
    if status == Status.HEALING:
      #TODO: Implement crash handling. Probably needs to be done by following waypoint list to exit the crash site.
      #Afterwards needs to remake the path.
      return self.knowledge.get_location()
    if status == Status.CRASHED:
      #TODO: implement function for crash handling, should provide map of wayoints to move towards to for exiting crash state.
      #You should use separate waypoint list for that, to not mess with the original path. 
      return self.knowledge.get_location()
    #otherwise destination is same as current position
    return self.knowledge.get_location()

  def get_next_destination(self):
    if len(self.path) > 1:
      return self.path[1]
    return None

  #TODO: Implementation
  def build_path(self, source, destination):
    self.path = deque([])
    #TODO: create path of waypoints from source to
    map = self.knowledge.get_map()
    #1. turn into waypoints
    wp_start = map.get_waypoint(source.location)
    wp_end = map.get_waypoint(carla.Transform(destination).location)
    hop = 10
    forward_target = wp_end.transform.location
    while True:
      forward = wp_start.next(hop)
      #backward = wp_end.previous()
      min = None
      min_dist = 9999999
      for wp in forward:
        wp_dist = wp.transform.location.distance(forward_target)
        if wp_dist < min_dist:
          min_dist = wp_dist
          min = wp

      self.path.append(min.transform.location)
      wp_start = min
      print(f"Added waypoint {min} to route, {min_dist} meters away")
      if min_dist < hop:
        break

    self.path.append(destination)
    return self.path


