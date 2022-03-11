


import math
import random
import numpy

import qConstants as qc


def directSum(A, B):
    '''takes in two complex matrices A (p * q) and B (m * n) and returns a single
    complex matrix with dimensions (p + m) * (q + n) with A and B arranged
    along the diagonal and zeros elsewhere'''
    top = numpy.concatenate((A, numpy.zeros((A.shape[0], B.shape[1]), dtype='complex128')), axis=1)
    bottom = numpy.concatenate((numpy.zeros((B.shape[0], A.shape[1]), dtype='complex128'), B), axis=1)
    return numpy.concatenate((top, bottom))

def equal(a, b, epsilon):
    '''Assumes that n >= 0. Assumes that a and b are both n-qbit states or n-qbit gates. Assumes that epsilon is a positive (but usually small) real number. Returns whether a == b to within a tolerance of epsilon. Useful for doing equality comparisons in the floating-point context. Warning: Does not consider global phase changes; for example, two states that are global phase changes of each other may be judged unequal. Warning: Use this function sparingly, for inspecting output and running tests. Probably you should not use it to make a crucial decision in the middle of a big computation. In past versions of CS 358, this function has not existed. I have added it this time just to streamline the tests.'''
    diff = a - b
    if len(diff.shape) == 0:
        # n == 0. Whether they're gates or states, a and b are scalars.
        return abs(diff) < epsilon
    elif len(diff.shape) == 1:
        # a and b are states.
        return sum(abs(diff)) < epsilon
    else:
        # a and b are gates.
        return sum(sum(abs(diff))) < epsilon

def uniform(n):
    '''Assumes n >= 0. Returns a uniformly random n-qbit state.'''
    if n == 0:
        return qc.one
    else:
        psiNormSq = 0
        while psiNormSq == 0:
            reals = numpy.array(
                [random.normalvariate(0, 1) for i in range(2**n)])
            imags = numpy.array(
                [random.normalvariate(0, 1) for i in range(2**n)])
            psi = numpy.array([reals[i] + imags[i] * 1j for i in range(2**n)])
            psiNormSq = numpy.dot(numpy.conj(psi), psi).real
        psiNorm = math.sqrt(psiNormSq)
        return psi / psiNorm

def bitValue(state):
    '''Given a one-qbit state assumed to be exactly classical --- usually because a classical state was just explicitly assigned to it --- returns the corresponding bit value 0 or 1.'''
    if (state == qc.ket0).all():
        return 0
    else:
        return 1

def powerMod(k, l, m):
    '''Given non-negative integer k, non-negative integer l, and positive integer m. Computes k^l mod m. Returns an integer in {0, ..., m - 1}.'''
    kToTheL = 1
    curr = k
    while l >= 1:
        if l % 2 == 1:
            kToTheL = (kToTheL * curr) % m
        l = l // 2
        curr = (curr * curr) % m
    return kToTheL

def lowest_terms(c, d):
    '''Converts the fraction c/d to lowest terms'''
    gcd = math.gcd(c, d)
    while gcd != 1:
        c = c / gcd
        d = d / gcd
        gcd = math.gcd(c, d)
    return c, d

def fraction(x0, j):
    '''returns a rational number approximation of x0 up to depth j'''
    if x0 == 0:
        return 0, 1
    a0 = math.floor(1 / x0)
    if j == 0:
        return 1, a0
    else:
        j = j - 1
        num, den = fraction((1 / x0) - a0, j)
        return lowest_terms(den, (a0 * den + num))

def continuedFraction(n, m, x0):
    '''x0 is a float in [0, 1). Tries probing depths j = 0, 1, 2, ... until
    the resulting rational approximation x0 ~ c / d satisfies either d >= m or
    |x0 - c / d| <= 1 / 2^(n + 1). Returns a pair (c, d) with gcd(c, d) = 1.'''
    j = 0
    if x0 == 0:
        return (0, 1)
    c, d = 1, math.floor(1 / x0)
    while not(abs(x0 - c / d) <= 1 / 2**(n + 1) or d >= m):
        c, d = fraction(x0, j)
        j += 1
    return (c, d)

def continuedFractionTest():
    #check 1 / pi
    c, d = fraction(1 / math.pi, 0)
    if c == 1 and d == 3:
        print("passed pi j = 0")
    else:
        print("failed pi j = 0")
    c, d = fraction(1 / math.pi, 1)
    if c == 7 and d == 22:
        print("passed pi j = 1")
    else:
        print("failed pi j = 1")
    c, d = fraction(1 / math.pi, 2)
    if c == 106 and d == 333:
        print("passed pi j = 2")
    else:
        print("failed pi j = 2")

    #check 0
    c, d = fraction(0, 100)
    if c == 0 and d == 1:
        print("passed 0")
    else:
        print("failed 0")

    #check rational number
    x = random.randint(0, 9999)
    x0 = 1 / x
    if c == 1 and d == x:
        print("passed rational")
    else:
        print("failed rational")
        print("x:", x)
        c, d = fraction(x0, 2)
        print("c:", c, "d:", d)

if __name__ == '__main__':
    continuedFractionTest()
