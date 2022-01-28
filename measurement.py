import math
import random
import numpy

# classical 1-qubit states
ket0 = numpy.array([1 + 0j, 0 + 0j])
ket1 = numpy.array([0 + 0j, 1 + 0j])

'''Takes in any 1-qubit state vector and performs a measurement operation to
return either ket0 or ket1'''
def measurement(state):
    if random.random() <= abs(state[0])**2:
        return ket0
    else:
        return ket1

'''This function measures the vector psi = 0.6ket0 + 0.8ket1 m times and returns
the proportion of times that the measurement results in ket1. The probability of
any measurement of psi returning ket1 is 0.8^2 = 0.64, so for large m we
expect this test to return about 0.64.'''
def measurementTest345(m):
    psi = 0.6 * ket0 + 0.8 * ket1
    def f():
        if (measurement(psi) == ket0).all():
            return 0
        else:
            return 1
    acc = 0
    for i in range(m):
        acc += f()
    return acc / m

print(measurementTest345(10000))
