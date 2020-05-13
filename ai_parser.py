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
import numpy as np
import math
from ai_knowledge import Status

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

    world = self.vehicle.get_world()
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range','10.0')
    #lidar_bp.set_attribute('rotation_frequency',str(40))
    lidar_bp.set_attribute('upper_fov','10.0')
    lidar_bp.set_attribute('lower_fov','-36.0')
    lidar_bp.set_attribute('channels','32')
    lidar_bp.set_attribute('points_per_second','5000')
    lidar_bp.set_attribute('sensor_tick','0.5')


    height = self.vehicle.get_transform().location.z

    self.lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(0,0,3),carla.Rotation(0,0,0)), attach_to=self.vehicle)
    self.lidar.listen(lambda event: Monitor._parse_lidar(weak_self, event))
    #self.lidar.listen(lambda point_cloud: point_cloud.save_to_disk('point_cloud/%.6d.ply' % point_cloud.frame))

  def __del__(self):
    self.lidar.destroy()

  #Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    # Update the position, heading, and movement data of the vehicle into knowledge
    self.knowledge.update_data('location', self.vehicle.get_transform().location)
    self.knowledge.update_data('heading', self.vehicle.get_transform().rotation)
    self.knowledge.update_data('velocity', self.vehicle.get_velocity())

    #TODO: doesn't work, need landmarks probably, these are way too short notice
    if self.vehicle.is_at_traffic_light():
      self.knowledge.set_at_traffic_light(True)
      self.knowledge.set_traffic_light(self.vehicle.get_traffic_light())
    else:
      self.knowledge.set_at_traffic_light(False)
      self.knowledge.set_traffic_light(None)


  @staticmethod
  def _parse_lidar(weak_self, event):
    self = weak_self()
    if not self:
      return
    self.knowledge.set_lidar_data(event)
      
    #points = np.frombuffer(event.raw_data, dtype=np.dtype('f4'))
    #points = np.reshape(points,(int(points.shape[0]/3),3))

def compute_lidar_proximity(points,lidar_angle):
  # N, NE, E, SE, S, SW, W, NW
  dist = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
  for point in points:

    angle = lidar_angle - math.atan2(point[1],point[0])
    if angle >= -math.pi/8 and angle < math.pi/8:
      index = 0
    elif angle >= math.pi/8 and angle < 3*math.pi/8:
      index = 1
    elif angle >= 3*math.pi/8 and angle < 5*math.pi/8:
      index = 2
    elif angle >= 5*math.pi/8 and angle < 7*math.pi/8:
      index = 3
    elif angle >= 7*math.pi/8 or angle < -7*math.pi/8:
      index = 4
    elif angle >= -7*math.pi/8 and angle < -5*math.pi/8:
      index = 5
    elif angle > -5*math.pi/8 and angle < -3*math.pi/8:
      index = 6
    elif angle > -3*math.pi/8 and angle < -math.pi/8:
      index = 7

    mag = np.linalg.norm(point)
    if mag < dist[index]:
      dist[index] = mag
  return dist

def compute_closest_point(points):
  min = 99999
  min_p = None
  for p in points:
    d = np.linalg.norm(p)
    if d < min:
      min_p = p
      min = d
  return min_p,min

def compute_lidar_movement(prev_data,curr_data):
  mvmnt = []
  for i in range(len(prev_data)):
    mvmnt.append(curr_data[i]-prev_data[i])

  return mvmnt


# Analyser is responsible for parsing all the data that the knowledge has received from Monitor and turning it into something usable
class Analyser(object):
  def __init__(self, knowledge):
    self.knowledge = knowledge
    self.knowledge.set_data_changed_callback(self.data_changed)

  # Function that is called at time intervals to update ai-state
  def update(self, time_elapsed):
    if self.knowledge.get_status() != Status.ARRIVED:
      self.update_obstacle_analysis()
      self.update_junction_knowledge()
      self.update_target_speed()


  def update_obstacle_analysis(self):
    ld = self.knowledge.get_lidar_data()
    loc = self.knowledge.get_location()
    closest = None
    dist = 1100000
    for p in ld:
      d=np.linalg.norm(np.array([p.x,p.y]))
      if d < dist and d > 1:
        dist = d
        closest = p

    car_v = self.knowledge.get_velocity()

    v = np.array([car_v.x,car_v.y])
    v_mag = self.knowledge.get_velocity_magnitude()
    dot = 0
    if v_mag != 0 and closest:
      v = v/v_mag
      u = np.array([closest.x,closest.y])
      u = u/dist
      dot = np.dot(v,u)


    if dist < 2.7 or (dot > .8 and dist < 3):
      self.knowledge.set_obstacles([closest])
      dest = find_safe_destination([closest],loc)
      self.knowledge.set_override_destination(dest)
      self.knowledge.set_status(Status.HEALING)
      self.knowledge.set_target_speed(3)
    else:
      self.knowledge.set_obstacles([])
      self.knowledge.set_status(Status.DRIVING)
    #where_from = ["N","NE","E","SE","S","SW","W","NW","?"]
    #print(f"Closest {where_from[closest]}, {dist} metres")

      

  def update_junction_knowledge(self):
    if self.knowledge.get_map().get_waypoint(self.knowledge.get_location()).get_junction():
      self.knowledge.set_at_junction(True)
    else:
      self.knowledge.set_at_junction(False)
    
    if self.knowledge.get_map().get_waypoint(self.knowledge.get_current_destination()).get_junction():
      self.knowledge.set_approaching_junction(True)
    else:
      self.knowledge.set_approaching_junction(False)

  def update_target_speed(self):
    if self.knowledge.get_status() == Status.ARRIVED:
      self.knowledge.set_target_speed(0)
      return
    traffic_light = self.knowledge.get_traffic_light()
    threshold = 50

    traffic_light_near = False
    vehicle_location = self.knowledge.get_location()
    vehicle_wp = self.knowledge.get_map().get_waypoint(vehicle_location)

    if traffic_light is not None:
      dist, dot = traffic_light_proximity(self.knowledge,traffic_light.get_location(),vehicle_location,threshold)
      traffic_light_near = True
    else:
      dist = threshold + 1
      for light in self.knowledge.get_all_traffic_lights():
        light_loc = light.get_location()
        wp = self.knowledge.get_map().get_waypoint(light_loc)
        l_dist,l_dot = traffic_light_proximity(self.knowledge,light_loc,vehicle_location,threshold)
        if l_dist > dist:
          continue
        if light.state != carla.TrafficLightState.Red:
          continue
        if wp.road_id != vehicle_wp.road_id or wp.lane_id != vehicle_wp.lane_id:
          continue
        dist,dot,traffic_light = l_dist,l_dot,light
      if dist < threshold:
        traffic_light_near = True

    at_junction = self.knowledge.get_at_junction()
    approaching_junction = self.knowledge.get_approaching_junction()
    if traffic_light_near:
      if approaching_junction or (self.knowledge.get_at_traffic_light() and dot > .8):
        self.knowledge.set_target_speed(0)
      else:
        self.knowledge.set_target_speed(5)
    elif approaching_junction:
      self.knowledge.set_target_speed(9)
    else:
      self.knowledge.set_target_speed(9)

  # Callback
  def data_changed(self, data_key):
    pass


def traffic_light_proximity(knowledge,light_loc,vehicle_loc,threshold):
  distance = vehicle_loc.distance(light_loc)
  light_vector = np.array([light_loc.x-vehicle_loc.x,light_loc.y - vehicle_loc.y])
  car_velocity = knowledge.get_velocity()
  car_velocity = np.array([car_velocity.x,car_velocity.y])
  v_mag = np.linalg.norm(car_velocity)
  if v_mag == 0:
    return (distance,0)
  else:
    velocity_correlation = np.dot(light_vector/np.linalg.norm(light_vector),car_velocity/v_mag)
  return (distance, velocity_correlation)

def find_safe_destination(obstacles,car_location):
  o = obstacles[0]
  v = np.array([o.x,o.y])
  v = v/np.linalg.norm(v)
  dest = carla.Vector3D(x=car_location.x - v[0], y=car_location.y -v[1])
  return dest