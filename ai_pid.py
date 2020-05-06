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

class Controller():

    def __init__(self,k_p,k_d,k_i,i_damp=1,name="PID controller",debug=False):
        self.k_p = k_p
        self.k_d = k_d
        self.k_i = k_i
        self.name = name
        self.debug =debug

        self.i_damp = i_damp

        self.last_known_error = 0
        self.last_known_signal = 0

        #TODO: For I term, decide
        #Simpler, but less control
        self.sum = 0

        #Worth it? it's what German Ros used (carla/agents/navigation/controller)
        #self.error_buffer = deque(maxlen=50)

    def step(self,error,time_delta):
        if time_delta == 0:
            return self.last_known_signal
        
        error_diff = error - self.last_known_error
        
        #Area under the trapeze from the last error to this error
        self.sum = self.sum*self.i_damp + time_delta*(self.last_known_error + error)/2
        
        #We could use a decay for the integral if we want, or the sliding window by Ros

        self.last_known_error = error

        p = self.k_p*error
        d = self.k_d*(error_diff/time_delta)
        i = self.k_i*(self.sum)

        self.last_known_signal =  p + i + d

        if self.debug:
            print(f"{self.name}  {error} => P:{p}\tI:{i}\tD:{d} => {self.last_known_signal}")

        if self.last_known_signal < -1:
            self.last_known_signal = -1
        elif self.last_known_signal > 1:
            self.last_known_signal = 1

        return self.last_known_signal





