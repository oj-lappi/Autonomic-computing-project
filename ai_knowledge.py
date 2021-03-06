#!/usr/bin/env python

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import ai_util as util
import numpy as np
from enum import Enum

class Status(Enum):
  ARRIVED = 0  
  DRIVING = 1
  CRASHED = 2
  HEALING = 3
  UNDEFINED = 4

# Class that holds the knowledge of the current state and serves as interaction point for all the modules
class Knowledge(object):
  def __init__(self,map,traffic_lights):
    self.map = map
    self.status = Status.ARRIVED
    self.memory = {
                    'location':carla.Vector3D(0.0,0.0,0.0),
                    'heading':carla.Rotation(0.0,0.0,0.0),
                    'velocity':carla.Vector3D(0.0,0.0,0.0),
                    'bounding_box':carla.Vector3D(0.0,0.0,0.0),
                    'lidar_transform':carla.Transform(),
                    'target_speed':10,
                    'lidar_data':[],
                    'lidar_movement':[],
                    'obstacles':[],
                    'traffic_light':None,
                    'traffic_sign':None,
                    'at_traffic_light':False,
                    'at_junction':False,
                    'approaching_junction':False,
                    'override_destination':None,
                    'traffic_lights':traffic_lights,
                  }    
    self.destination = self.get_location()
    self.next_destination = None
    self.status_changed = lambda *_, **__: None
    self.destination_changed = lambda *_, **__: None
    self.data_changed = lambda *_, **__: None

  def set_data_changed_callback(self, callback):
    self.data_changed = callback

  def set_status_changed_callback(self, callback):
    self.status_changed = callback

  def set_destination_changed_callback(self, callback):
    self.destination_changed = callback

  def set_status(self, new_status):
    self.status = new_status

  def set_target_speed(self, new_target_speed):
    self.update_data('target_speed',new_target_speed)

  def set_traffic_light(self,traffic_light):
    self.update_data("traffic_light",traffic_light)

  def set_traffic_sign(self,traffic_sign):
    self.update_data("traffic_sign",traffic_sign)

  def set_at_traffic_light(self,at_traffic_light):
    self.update_data("at_traffic_light",at_traffic_light)

  def set_at_junction(self,at_junction):
    self.update_data("at_junction",at_junction)

  def set_approaching_junction(self,approaching_junction):
    self.update_data("approaching_junction",approaching_junction)

  def set_override_destination(self,override_destination):
    self.update_data("override_destination",override_destination)

  def set_lidar_data(self,lidar_data):
    self.update_data("lidar_data",lidar_data)

  def set_obstacles(self, obstacles):
    self.update_data("obstacles",obstacles)

  def add_obstacle(self, obstacle):
    obstacles = self.get_obstacles()
    obstacles.append(obstacle)
    self.update_data("obstacles",obstacles)

  # Retrieving data from memory
  # !Take note that it is unsafe and does not check whether the given field is in dic
  def retrieve_data(self, data_name):
    return self.memory[data_name]

  #updating status to correct value and making sure that everything is handled properly
  def update_status(self, new_status):
    if (self.status != Status.CRASHED or new_status == Status.HEALING) and self.status != new_status:
      self.set_status(new_status)
      self.status_changed(new_status)

  def get_status(self):
    return self.status

  def get_current_destination(self):
    return self.destination

  def get_next_destination(self):
    return self.next_destination

  def get_target_speed(self):
    return self.retrieve_data('target_speed')

  # Obstacles
  def get_obstacles(self):
    return self.retrieve_data('obstacles')

  def get_all_traffic_lights(self):
    return self.retrieve_data('traffic_lights')

  def get_at_traffic_light(self):
    return self.retrieve_data('at_traffic_light')

  def get_at_junction(self):
    return self.retrieve_data('at_junction')

  def get_approaching_junction(self):
    return self.retrieve_data('approaching_junction')

  def get_traffic_light(self):
    return self.retrieve_data('traffic_light')

  def get_traffic_sign(self):
    return self.retrieve_data('traffic_sign')

  def get_override_destination(self):
    return self.retrieve_data('override_destination')

  def get_bounding_box(self):
    return self.retrieve_data('bounding_box')

  def get_lidar_transform(self):
    return self.retrieve_data('lidar_transform')

  # Return current location of the vehicle
  def get_location(self):
    return self.retrieve_data('location')

  # Return current heading of the vehicle
  def get_heading(self):
    return self.retrieve_data('heading')

  # Return current velocity of the vehicle
  def get_velocity(self):
    return self.retrieve_data('velocity')

  def get_velocity_magnitude(self):
    v = self.retrieve_data('velocity')
    v_mag = np.linalg.norm(np.array([v.x,v.y]))
    if abs(v_mag) < .01:
      return 0
    else:
      return v_mag
  
  def get_lidar_data(self):
    return self.retrieve_data('lidar_data')

  # Return current acceleration of the vehicle
  def get_acceleration(self):
    return self.retrieve_data('acceleration')

  # Return current acceleration of the vehicle
  def get_angular_velocity(self):
    return self.retrieve_data('angular_velocity')

  def get_map(self):
    return self.map

  def arrived_at(self, destination):
    return self.distance(self.get_location(),destination) < 4.0

  def update_destination(self, new_destination,new_forward_destination=None):
    #if force or self.distance(self.destination,new_destination) < 5.0:
    self.destination = new_destination
    self.next_destination = new_forward_destination
    self.destination_changed(new_destination)
      #p = get_start_point(world,new_destination)
      #util.spawn_waypoint_marker(world, actor_list, cone, p.transform)
   
  # A function to receive data from monitor
  # TODO: Add callback so that analyser can know when to parse the data
  def update_data(self, data_name, pars):
    self.memory[data_name] = pars
    self.data_changed(data_name)

  def distance(self, vec1, vec2):
    l1 = carla.Location(vec1)
    l2 = carla.Location(vec2)
    return l1.distance(l2)
