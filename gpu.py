#!/usr/bin/python
import time
t0 = time.time()
from Electromagnetism import PIC
from MathTools import Vector
import random
import numpy as np

# E and B are in Normalized units
E = Vector(0,0,0)
B = Vector(0,0,-1)

numParticles = 100000
numTimeSteps = 100000
step = 0.1

ionPosition = np.array([[random.uniform(0,1),
                         random.uniform(0,1),
                         random.uniform(0,1)] for i in xrange(numParticles)],dtype=np.float32).flatten()

ionVelocity = np.array([[random.normalvariate(0,1),
                         random.normalvariate(0,1),
                         random.normalvariate(0,1)] for i in xrange(numParticles)],dtype=np.float32).flatten()

mySim = PIC(numParticles,E,B,step)
mySim.stepN(ionPosition, ionVelocity, numTimeSteps)
t1 = time.time()
print t1-t0
