from CAVSimulator0728 import Simulator
import numpy as np

env = Simulator(3,3)

s = env.reset()

v, x, a = env.CACC(s,env.num_leading_cars)

print(s)
print(v,x,a)