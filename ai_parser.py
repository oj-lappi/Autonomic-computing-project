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

import weakref
import carla
import ai_knowledge as data
import ai_util as util


# Monitor is responsible for reading the data from the sensors and telling it to the knowledge
# TODO: Implement other sensors (lidar and depth sensors mainly)
# TODO: Use carla API to read whether car is at traffic lights and their status, update it into knowledge
class Monitor(object):
  def __init__(self, knowledge,vehicle):
    self.vehicle = vehicle
    self.knowledge = knowledge
    weak_self = weakref.ref(self)
    
    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('heading', self.vehicle.get_transform().rotation)
    self.knowledge.update_data('velocity', self.vehicle.get_velocity())
    self.knowledge.update_data('acceleration', self.vehicle.get_acceleration())
    self.knowledge.update_data('angular_velocity', self.vehicle.get_angular_velocity())

    world = self.vehicle.get_world()
    #FIXED: this sensor had the wrong name
    #bp = world.get_blueprint_library().find('sensor.other.lane_detector')
    bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
    self.lane_detector = world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
    self.lane_detector.listen(lambda event: Monitor._on_invasion(weak_self, event))

  #Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    # Update the position, heading, and movement data of the vehicle into knowledge
    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('heading', self.vehicle.get_transform().rotation)
    self.knowledge.update_data('velocity', self.vehicle.get_velocity())
    self.knowledge.update_data('acceleration', self.vehicle.get_acceleration())
    self.knowledge.update_data('angular_velocity', self.vehicle.get_angular_velocity())

  @staticmethod
  def _on_invasion(weak_self, event):
    self = weak_self()
    if not self:
      return
    self.knowledge.update_data('lane_invasion',event.crossed_lane_markings)

# Analyser is responsible for parsing all the data that the knowledge has received from Monitor and turning it into something usable
# TODO: During the update step parse the data inside knowledge into information that could be used by planner to plan the route
class Analyser(object):
  def __init__(self, knowledge):
    self.knowledge = knowledge

  #Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    return
