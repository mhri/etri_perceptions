'''
Exponentially Decaying Counter

Author: Minsu Jang (minsu@etri.re.kr)
'''

from threading import Thread
from math import exp
import time

class DecayingPotential(Thread):
    def __init__(self, tau=1.0, initial_value=0):
        Thread.__init__(self)
        self.potential = initial_value
        self.tau = tau
        self.time = time.time()
        self.suspend_ = False
        self.exit_ = False

    def decay(self):
        cur_time = time.time()
        delta_time = cur_time - self.time
        if delta_time > 0:
            self.potential *= exp(delta_time * -0.69 / self.tau)
        self.time = cur_time

    def spike(self):
        self.potential += 0.3

    def run(self):
        while True:
            while self.suspend_:
                time.sleep(0.5)

            self.decay()
            if self.exit_ == True:
                break
            time.sleep(0.01)

    def get_potential(self):
        return self.potential
    
    def suspend(self):
        self.suspend_ = True
        
    def resume(self):
        self.suspend_ = False
        
    def stop(self):
        self.exit_ = True