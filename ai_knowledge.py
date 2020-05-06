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
from enum import Enum

class Status(Enum):
  ARRIVED = 0  
  DRIVING = 1
  CRASHED = 2
  HEALING = 3
  UNDEFINED = 4

# Class that holds the knowledge of the current state and serves as interaction point for all the modules
class Knowledge(object):
  def __init__(self):
    self.status = Status.ARRIVED
    self.memory = {
                    'location':carla.Vector3D(0.0,0.0,0.0),
                    'heading':carla.Rotation(0.0,0.0,0.0),
                    'velocity':carla.Vector3D(0.0,0.0,0.0),
                    'acceleration':carla.Vector3D(0.0,0.0,0.0),
                    'angular_velocity':carla.Vector3D(0.0,0.0,0.0),
                  }    
    self.destination = self.get_location()
    self.status_changed = lambda *_, **__: None
    self.destination_changed = lambda *_, **__: None
    self.data_changed = lambda *_, **__: None

  def set_data_changed_callback(self, callback):
    self.data_changed = callback

  def set_status_changed_callback(self, callback):
    self.status_changed = callback

  def set_destination_changed_callback(self, callback):
    self.destination_changed = callback

  def get_status(self):
    return self.status

  def set_status(self, new_status):
    self.status = new_status

  def get_current_destination(self):
    return self.destination

  # Retrieving data from memory
  # !Take note that it is unsafe and does not check whether the given field is in dic
  def retrieve_data(self, data_name):
    return self.memory[data_name]

  #updating status to correct value and making sure that everything is handled properly
  def update_status(self, new_status):
    if (self.status != Status.CRASHED or new_status == Status.HEALING) and self.status != new_status:
      self.set_status(new_status)
      self.status_changed(new_status)

  # Return current location of the vehicle
  def get_location(self):
    return self.retrieve_data('location')

  # Return current heading of the vehicle
  def get_heading(self):
    return self.retrieve_data('heading')

  # Return current velocity of the vehicle
  def get_velocity(self):
    return self.retrieve_data('velocity')

  # Return current acceleration of the vehicle
  def get_acceleration(self):
    return self.retrieve_data('acceleration')

  # Return current acceleration of the vehicle
  def get_angular_velocity(self):
    return self.retrieve_data('angular_velocity')

  def arrived_at(self, destination):
    return self.distance(self.get_location(),destination) < 5.0

  def update_destination(self, new_destination, force =False):
    if force or self.distance(self.destination,new_destination) < 5.0:
      self.destination = new_destination
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
