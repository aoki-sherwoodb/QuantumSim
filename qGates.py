
import random
import numpy

import qConstants as qc
import qUtilities as qu
import qBitStrings as qb


def application(u, ketPsi):
    '''Assumes n >= 1. Applies the n-qbit gate U to the n-qbit state |psi>, returning the n-qbit state U |psi>.'''
    return numpy.dot(u, ketPsi)

def tensor(a, b):
    '''Assumes that n, m >= 1. Assumes that a is an n-qbit state and b is an
    m-qbit state, or that a is an n-qbit gate and b is an m-qbit gate. Returns
    the tensor product of a and b, which is an (n + m)-qbit gate or state.'''
    n = a.shape[0]
    m = b.shape[0]
    if len(a.shape) == 1:    #a, b are states
        product = numpy.zeros(n*m, dtype='complex128')
        product_index = 0
        for i in range(n):
            for j in range(m):
                product[product_index] = a[i] * b[j]
                product_index += 1
        return product
    else:                       #a, b are gates
        product = numpy.zeros((n*m, n*m), dtype='complex128')
        for row in range(n*m):
            for col in range(n*m):
                product[row,col] = a[row // m, col // m] * b[row % m, col % m]
        return product

def function(n, m, f):
    '''Assumes n,m >= 1. Given a Python function f : {0,1}^n -> {0,1}^m,
    that is, f takes as input an n-bit string and produces as output an m-bit
    string, as defined in qBitStrings.py. Returns the corresponding
    (n+m)-qbit gate F.'''
    F = numpy.zeros((2**(n+m), 2**(n+m)), dtype='complex128')
    for i in range(2**n):
        alpha = qb.string(n, i)
        ketAlpha = qb.string_to_state(alpha)
        for k in range(2**m):
            beta = qb.string(m, k)
            F[qb.integer(alpha + beta)] = tensor(ketAlpha, qb.string_to_state(qb.addition(beta, f(alpha))))
    return F.T

def power(stateOrGate, m):
    '''Assumes n >= 1. Given an n-qbit gate or state and m >= 1, returns the
    mth tensor power, which is an (n * m)-qbit gate or state. For the sake of
    time and memory, m should be small.'''
    result = stateOrGate
    for i in range(m - 1):
        result = tensor(result, stateOrGate)
    return result

### DEFINING SOME TESTS ###

def applicationTest():
    # These simple tests detect type errors but not much else.
    answer = application(qc.h, qc.ketMinus)
    if qu.equal(answer, qc.ket1, 0.000001):
        print("passed applicationTest first part")
    else:
        print("FAILED applicationTest first part")
        print("    H |-> = " + str(answer))
    ketPsi = qu.uniform(2)
    answer = application(qc.swap, application(qc.swap, ketPsi))
    if qu.equal(answer, ketPsi, 0.000001):
        print("passed applicationTest second part")
    else:
        print("FAILED applicationTest second part")
        print("    |psi> = " + str(ketPsi))
        print("    answer = " + str(answer))

def tensorTest():
    # Pick two gates and two states.
    u = qc.x
    v = qc.h
    ketChi = qu.uniform(1)
    ketOmega = qu.uniform(1)
    # Compute (U tensor V) (|chi> tensor |omega>) in two ways.
    a = tensor(application(u, ketChi), application(v, ketOmega))
    b = application(tensor(u, v), tensor(ketChi, ketOmega))
    # Compare.
    if qu.equal(a, b, 0.000001):
        print("passed tensorTest")
    else:
        print("FAILED tensorTest")
        print("    a = " + str(a))
        print("    b = " + str(b))

def functionTest(n, m):
    # 2^n times, randomly pick an m-bit string.
    values = [qb.string(m, random.randrange(0, 2**m)) for k in range(2**n)]
    # Define f by using those values as a look-up table.
    def f(alpha):
        a = qb.integer(alpha)
        return values[a]
    # Build the corresponding gate F.
    ff = function(n, m, f)
    # Helper functions --- necessary because of poor planning.
    def g(gamma):
        if gamma == 0:
            return qc.ket0
        else:
            return qc.ket1

    def ketFromBitString(alpha):
        ket = g(alpha[0])
        for gamma in alpha[1:]:
            ket = tensor(ket, g(gamma))
        return ket
    # Check 2^n - 1 values somewhat randomly.
    alphaStart = qb.string(n, random.randrange(0, 2**n))
    alpha = qb.next(alphaStart)
    while alpha != alphaStart:
        # Pick a single random beta to test against this alpha.
        beta = qb.string(m, random.randrange(0, 2**m))
        # Compute |alpha> tensor |beta + f(alpha)>.
        ketCorrect = ketFromBitString(alpha + qb.addition(beta, f(alpha)))
        # Compute F * (|alpha> tensor |beta>).
        ketAlpha = ketFromBitString(alpha)
        ketBeta = ketFromBitString(beta)
        ketAlleged = application(ff, tensor(ketAlpha, ketBeta))
        # Compare.
        if not qu.equal(ketCorrect, ketAlleged, 0.000001):
            print("failed functionTest")
            print(" alpha = " + str(alpha))
            print(" beta = " + str(beta))
            print(" ketCorrect = " + str(ketCorrect))
            print(" ketAlleged = " + str(ketAlleged))
            print(" and here’s F...")
            print(ff)
            return
        else:
            alpha = qb.next(alpha)
    print("passed functionTest")


### RUNNING THE TESTS ###

def main():
    applicationTest()
    applicationTest()
    tensorTest()
    tensorTest()
    functionTest(1,1)
    functionTest(2,2)
    functionTest(3,3)

if __name__ == "__main__":
    main()
