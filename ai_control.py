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

    self.throttle_control = pid.Controller(0.44,-0.4,0.000001, name="Throttle",debug=False)
    self.steering_control = pid.Controller(0.6,-0.7,0.000003, i_damp=0.5, name="Steering",debug=False)
    
  #Update the executor at some intervals to steer the car in desired direction
  def update(self, time_elapsed):
    status = self.knowledge.get_status()
    #TODO: this needs to be able to handle
    if status == Status.DRIVING:
      dest = self.knowledge.get_current_destination()
      next = self.knowledge.get_next_destination()
      target_speed = self.knowledge.get_target_speed()
      self.update_control(dest, [target_speed,next], time_elapsed)
    if status == Status.ARRIVED:
      self.update_control(None,[0,0],0)

    if status == Status.HEALING:
      print("Avoiding obstacle")
      dest = self.knowledge.get_current_destination()
      next = self.knowledge.get_next_destination()
      target_speed = self.knowledge.get_target_speed()
      self.update_control(dest, [target_speed,next], time_elapsed)

    #if status == Status.HEALING:
    #  self.

  def update_control(self, destination, additional_vars, delta_time):
    target_speed = additional_vars[0]
    forward_waypoint = additional_vars[1]

    # Set control values
    control = carla.VehicleControl()
    if target_speed != 0:
      control.throttle,control.brake,control.steer,control.reverse = self.calculate_control(destination,forward_waypoint,target_speed,delta_time)
      control.hand_brake = False
    else:
      control.brake = 1
      control.throttle = 0
      control.steer = 0
    self.vehicle.apply_control(control)


  def calculate_control(self,destination,forward_waypoint,target_speed,delta_time):
    car_position = self.knowledge.get_location()
    car_heading = self.knowledge.get_heading().get_forward_vector()
    car_velocity = self.knowledge.get_velocity()
    car_velocity = np.array([car_velocity.x,car_velocity.y])

    ### Calculate errors
    step_target_vector, magnitude, sign = util.calculate_steering_error(destination,forward_waypoint,car_position,car_heading,lookahead_window=target_speed, max_forward_lookahead=target_speed)

    #Steering error is the angle between the vector to the destination and the current car heading
    #Dist error is the distance to the target

    forward_delta = 1
    velocity_error = step_target_vector - car_velocity*forward_delta

    #velocity_correlation is how well the error (signal) aligns with the current velocity
    v_err_mag =  np.linalg.norm(velocity_error)
    step_mag = np.linalg.norm(step_target_vector)
    if v_err_mag < 0.01 or step_mag < 0.01:
      return (0,0,0,False)
    else:
      velocity_correlation = np.dot(step_target_vector/step_mag,velocity_error/v_err_mag)

    throttle_signal = self.throttle_control.step(np.linalg.norm(velocity_error),delta_time)*velocity_correlation*abs(velocity_correlation)#+forward_throttle_correlation)
    steering_signal = self.steering_control.step(magnitude*sign,delta_time)

    brake_signal = 0
    reverse = False
    ### Calculate throttle and brake
    if throttle_signal < 0:
      ## Check if we're going forward
      car_vec = np.array([car_heading.x,car_heading.y])
      v_mag = self.knowledge.get_velocity_magnitude()
      if v_mag == 0:
        forwardness = 0
      else:
        forwardness = np.dot(car_vec,car_velocity/v_mag)
      if forwardness > 0:
        brake_signal = -throttle_signal
        throttle_signal = 0
      else:
        print("REV")
        reverse = True
        throttle_signal = -throttle_signal
    elif throttle_signal < 0.3:
      brake_signal = 0.3 -throttle_signal

    return (throttle_signal,brake_signal,steering_signal,reverse)



# Planner is responsible for creating a plan for moving around
# In our case it creates a list of waypoints to follow so that vehicle arrives at destination
# Alternatively this can also provide a list of waypoints to try avoid crashing or 'uncrash' itself
class Planner(object):
  def __init__(self, knowledge,debug=False):
    self.knowledge = knowledge
    self.path = deque([])
    self.urgent_path = deque([])
    self.debug = debug

  # Create a map of waypoints to follow to the destination and save it
  def make_plan(self, source, destination):
    self.path = self.build_path(source,destination)
    self.update(0)
  
  # Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    self.update_plan()
    self.knowledge.update_destination(self.get_current_destination(),self.get_next_destination())
    #You should use separate waypoint list for that, to not mess with the original path. 

    
  
  #Update internal state to make sure that there are waypoints to follow and that we have not arrived yet
  def update_plan(self):
    status = self.knowledge.get_status()
    if status == Status.ARRIVED or status == Status.DRIVING:
      if len(self.path) == 0:
        self.knowledge.update_status(Status.ARRIVED)
      else:
        self.knowledge.update_status(Status.DRIVING)
        if self.knowledge.arrived_at(self.path[0]):
          self.path.popleft()



  #get current destination 
  def get_current_destination(self):

    status = self.knowledge.get_status()

    if status == Status.DRIVING:
      if len(self.path) > 0:
        return self.path[0]
    elif status == Status.HEALING:
      return self.knowledge.get_override_destination()
      #if len(self.urgent_path) > 0:
      #  return self.urgent_path[0]
    return self.knowledge.get_location()

  def get_next_destination(self):
    status = self.knowledge.get_status()
    if status == Status.DRIVING:
      if len(self.path) > 1:
        return self.path[1]
    return None

  def build_path(self, source, destination):
    map = self.knowledge.get_map()
    #1. turn into waypoints
    wp_start = map.get_waypoint(source)
    wp_end = map.get_waypoint(carla.Transform(destination).location)

    distance = source.distance(destination)
    if distance < 10:
      return deque([destination])
    # Some variables
    hop = 10
    arrival_threshold = 8
    visited = {}
    #forward_target = wp_end.transform.location
    forward_target = destination

    if self.debug:
      print("Planning route:")
    paths = [([wp],0) for wp in gen_new_waypoints(wp_start,hop)]
    while True:
      new_paths = []
      if not paths:
        return deque([])
      if self.debug:
        print("-")
      for path in paths:
        if path[0][-1].id in visited:
          continue
        
        visited[path[0][-1].id] =True
        if self.debug:
          print(f"{path[1]:.1f}, ({util.loc_string(path[0][-1].transform.location)})")
        path_end = path[0][-1].transform.location
        wp_dist = path_end.distance(forward_target)
        if wp_dist < arrival_threshold:
          if path_end.distance(carla.Transform(destination).location) > arrival_threshold:
            the_path = deque([wp.transform.location for wp in path[0]]+[destination])
          else:
            the_path = deque([wp.transform.location for wp in path[0]])
          if self.debug:
            print("----------------------------------------------------")
            for wp in the_path:
              print("\t",util.loc_string(wp))
            print("----------------------------------------------------\n")
          return the_path
        if wp_dist < hop:
          near_search_paths = [(path[0] + [wp],wp_dist) for wp in gen_new_waypoints(path[0][-1],wp_dist+1)]
          if near_search_paths:
            new_paths += near_search_paths
            continue
        new_paths += [(path[0] + [wp],wp_dist) for wp in gen_new_waypoints(path[0][-1],hop)]
      paths = new_paths#sorted(new_paths,key=lambda p:p[1])

def gen_new_waypoints(wp,hop):
  waypoints = wp.next(hop)
  if wp.lane_change == carla.LaneChange.Both or wp.lane_change ==carla.LaneChange.Right:
    rl = wp.get_right_lane()
    if rl:
      waypoints.append(rl)
  if wp.lane_change == carla.LaneChange.Both or wp.lane_change ==carla.LaneChange.Left:
    ll = wp.get_right_lane()
    if ll:
      waypoints.append(ll)

  return waypoints
