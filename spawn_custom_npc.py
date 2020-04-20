#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

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

import argparse
import random
import time

import custom_ai as ai

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-d', '--delay',
        metavar='D',
        default=2.0,
        type=float,
        help='delay in seconds between spawns (default: 2.0)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    args = argparser.parse_args()

    actor_list = []
    vai_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    try:

        world = client.get_world()
        blueprints = world.get_blueprint_library().filter('vehicle.*')

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]

        #This is definition of a callback function that will be called when the autopilot arrives at destination
        def route_finished(autopilot):
            print("Vehicle arrived at destination")
            #After vehicle has arrived we set a random spawn point as a new destination
            #TODO: Make an 'intelligent' list of targets where cars could go (has annotated waypoints so you could use that)
            controller.set_destination(random.choice(world.get_map().get_spawn_points()))
            #TODO, BONUS: Make fixed exit and entry points (for example parking lots), 
            #so that cars are removed from simulation when they enter those and new ones are created from random points.
            #use try_spawn_random_vehicle_at(random.choice(spawn_points)) to spawn new cars
            

        #Function to spawn new vehicles
        def try_spawn_random_vehicle_at(transform):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                actor_list.append(vehicle)
                #Instead of setting default autopilot, we create our own and append it to the list of autopilots 
                autopilot = ai.Autopilot(vehicle)
                #We also register callback to know when the vehicle has arrived at it's destination
                autopilot.set_route_finished_callback(route_finished)
                vai_list.append(autopilot)

                #vehicle.set_autopilot()
                print('spawned %r at %s' % (vehicle.type_id, transform.location))
                return True
            return False

        spawn_points = list(world.get_map().get_spawn_points())
        random.shuffle(spawn_points)	

        print('found %d spawn points.' % len(spawn_points))

        count = args.number_of_vehicles

        for spawn_point in spawn_points:
            if try_spawn_random_vehicle_at(spawn_point):
                count -= 1
            if count <= 0:
                break

        print('spawned %d vehicles, press Ctrl+C to exit.' % args.number_of_vehicles)

        #Infinite loop to update car statuses
        while True:
	    #This could be done in parallel with threading for better performance
            for controller in vai_list:
                status = controller.update()
                    
    finally:

        print('\ndestroying %d actors' % len(actor_list))
        client.apply_batch([carla.command.DestroyActor(x.id) for x in actor_list])


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
