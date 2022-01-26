


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
                product[row,col] = a[row // n, col // n] * b[row % m, col % m]
        return product

def function(n, m, f):
    '''Assumes n,m == 1. Given a Python function f : {0,1}^n -> {0,1}^m,
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



### RUNNING THE TESTS ###

def main():
    applicationTest()
    applicationTest()
    tensorTest()
    tensorTest()

if __name__ == "__main__":
    main()
