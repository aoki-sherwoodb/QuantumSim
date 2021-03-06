


import random
import math
import numpy

import qConstants as qc
import qUtilities as qu
import qGates as qg



def first(state):
    '''Assumes n >= 1. Given an n-qbit state, measures the first qbit.
    Returns a pair (tuple of two items) consisting of a classical one-qbit state
    (either |0> or |1>) and an (n - 1)-qbit state.'''
    sigma0, sigma1 = 0,0
    for i in range(len(state) // 2):
        sigma0 += abs(state[i])**2
    for i in range(len(state) // 2, len(state)):
        sigma1 += abs(state[i])**2
    sigma0, sigma1 = math.sqrt(sigma0), math.sqrt(sigma1)
    if qu.equal(sigma0, qc.zero, 0.000001):    #first two cases avoid division by 0
        ketChi = numpy.zeros(len(state) // 2, dtype='complex128')
        ketPsi = state[len(state) // 2:]
    elif qu.equal(sigma1, qc.zero, 0.000001):
        ketPsi = numpy.zeros(len(state) // 2, dtype='complex128')
        ketChi = state[:len(state) // 2]
    else:
        ketChi = (1 / sigma0) * state[:len(state) // 2]
        ketPsi = (1 / sigma1) * state[len(state) // 2:]
    if len(ketChi) == 1:
        ketChi = qc.one
    if len(ketPsi) == 1:
        ketPsi = qc.one
    if random.random() <= abs(sigma0)**2:
        return (qc.ket0, ketChi)
    else:
        return (qc.ket1, ketPsi)

def last(state):
    '''Assumes n >= 1. Given an n-qbit state, measures the last qbit.
    Returns a pair consisting of an (n - 1)-qbit state and a classical 1-qbit state
    (either |0> or |1>).'''
    sigma0, sigma1 = 0,0
    for i in range(len(state)):
        if i % 2 == 0:
            sigma0 += abs(state[i])**2
        else:
            sigma1 += abs(state[i])**2
    ketChi = numpy.fromiter((state[i] for i in range(len(state)) if i % 2 == 0), dtype='complex128')
    ketPsi = numpy.fromiter((state[i] for i in range(len(state)) if i % 2 != 0), dtype='complex128')
    sigma0, sigma1 = math.sqrt(sigma0), math.sqrt(sigma1)
    if qu.equal(sigma0, qc.zero, 0.000001):    #first two cases avoid division by 0
        ketChi = numpy.zeros(len(state) // 2, dtype='complex128')
    elif qu.equal(sigma1, qc.zero, 0.000001):
        ketPsi = numpy.zeros(len(state) // 2, dtype='complex128')
    else:
        ketChi = (1 / sigma0) * ketChi
        ketPsi = (1 / sigma1) * ketPsi
    if len(ketChi) == 1:
        ketChi = qc.one
    if len(ketPsi) == 1:
        ketPsi = qc.one
    if random.random() <= abs(sigma0)**2:
        return (ketChi, qc.ket0)
    else:
        return (ketPsi, qc.ket1)



### DEFINING SOME TESTS ###

def firstTest(n):
    # Assumes n >= 1. Constructs an unentangled (n + 1)-qbit state |0> |psi> or |1> |psi>, measures the first qbit, and then reconstructs the state.
    ketPsi = qu.uniform(n)
    state = qg.tensor(qc.ket0, ketPsi)
    meas = first(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed firstTest first part")
    else:
        print("failed firstTest first part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))
    ketPsi = qu.uniform(n)
    state = qg.tensor(qc.ket1, ketPsi)
    meas = first(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed firstTest second part")
    else:
        print("failed firstTest second part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))

def firstTest345(n, m):
    # Assumes n >= 1. n + 1 is the total number of qbits. m is how many tests to run. Should return a number close to 0.64 --- at least for large m.
    psi0 = 3 / 5
    ketChi = qu.uniform(n)
    psi1 = 4 / 5
    ketPhi = qu.uniform(n)
    ketOmega = psi0 * qg.tensor(qc.ket0, ketChi) + psi1 * qg.tensor(qc.ket1, ketPhi)
    def f():
        if (first(ketOmega)[0] == qc.ket0).all():
            return 0
        else:
            return 1
    acc = 0
    for i in range(m):
        acc += f()
    print("check firstTest345 for frequency near 0.64")
    print("    frequency = ", str(acc / m))

def lastTest(n):
    # Assumes n >= 1. Constructs an unentangled (n + 1)-qbit state |psi> |0> or |psi> |1>, measures the last qbit, and then reconstructs the state.
    psi = qu.uniform(n)
    state = qg.tensor(psi, qc.ket0)
    meas = last(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed lastTest first part")
    else:
        print("failed lastTest first part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))
    psi = qu.uniform(n)
    state = qg.tensor(psi, qc.ket1)
    meas = last(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed lastTest second part")
    else:
        print("failed lastTest second part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))

def lastTest345(n, m):
    # Assumes n >= 1. n + 1 is the total number of qbits. m is how many tests to run. Should return a number close to 0.64 --- at least for large m.
    psi0 = 3 / 5
    ketChi = qu.uniform(n)
    psi1 = 4 / 5
    ketPhi = qu.uniform(n)
    ketOmega = psi0 * qg.tensor(ketChi, qc.ket0) + psi1 * qg.tensor(ketPhi, qc.ket1)
    def f():
        if (last(ketOmega)[1] == qc.ket0).all():
            return 0
        else:
            return 1
    acc = 0
    for i in range(m):
        acc += f()
    print("check lastTest345 for frequency near 0.64")
    print("    frequency = ", str(acc / m))



### RUNNING THE TESTS ###

def main():
    firstTest(1)
    firstTest(1)
    firstTest345(1, 10000)
    firstTest345(1, 10000)
    lastTest(1)
    lastTest(1)
    lastTest345(1, 10000)
    lastTest345(1, 10000)

if __name__ == "__main__":
    main()
