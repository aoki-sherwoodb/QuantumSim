
import random
import numpy
import math

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
                # multiply all values in b by each value in a in order
                product[product_index] = a[i] * b[j]
                product_index += 1
        return product
    else:                       #a, b are gates
        product = numpy.zeros((n*m, n*m), dtype='complex128')
        for row in range(n*m):
            for col in range(n*m):
                # multiply all values in b by each value in a in a grid
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

def fourier(n):
    '''builds the n-qbit QFT T using the definition'''
    qft = numpy.zeros((2**n, 2**n), dtype='complex128')
    for a in range(2**n):
        for b in range(2**n):
            qft[a, b] = numpy.exp(1j * 2 * numpy.pi * a * b / 2**n)
    return (1 / 2**(n / 2)) * qft

def buildS(n):
    '''Generates the n-qubit S matrix that swaps the last bit of a state to be
    the first bit'''
    if n == 2:
        return qc.swap
    else:
        s = tensor(power(qc.i, n-2), qc.swap)
        for k in range(1, n-2):
            sLayer = tensor(tensor(power(qc.i, n-2-k), qc.swap), power(qc.i, k))
            s = numpy.dot(sLayer, s)
        sLayer = tensor(qc.swap, power(qc.i, n-2))
        s = numpy.dot(sLayer, s)
        return s

def fourierRecursive(n):
    '''Assumes n >= 1. Returns the n-qbit quantum Fourier transform gate T.
    Computes T recursively rather than from the definition.'''
    if n == 1:
        return qc.h
    else:
        r = tensor(qc.i, fourierRecursive(n-1))
        omega = numpy.exp(1j * 2 * numpy.pi / 2**(n))
        s = buildS(n)
        d = numpy.zeros((2**(n-1), 2**(n-1)), dtype='complex128')
        for i in range(2**(n-1)):
            d[i,i] = omega**i
        I = power(qc.i, n-1)
        q = numpy.dot(tensor(qc.h, I), qu.directSum(I, d))
        return numpy.dot(q, numpy.dot(r, s))

def distant(gate):
    '''Given an (n + 1)-qbit gate U (such as a controlled-V gate, where V is
    n-qbit), performs swaps to insert one extra wire between the first qbit and
    the other n qbits. Returns an (n + 2)-qbit gate.'''
    n = gate.shape[0] // 4
    swapLayer = tensor(qc.swap, power(qc.i, n))
    return numpy.dot(swapLayer, numpy.dot(tensor(qc.i, gate), swapLayer))

def ccNot():
    '''Returns the three-qbit ccNOT (i.e., Toffoli) gate. The gate is
    implemented using five specific two-qbit gates and some SWAPs.'''
    u = numpy.array([
        [1 + 0j, 0 + 0j],
        [0 + 0j, 0 - 1j]])
    v = (1 / math.sqrt(2)) * numpy.array([
        [1 + 0j, 0 + 1j],
        [0 - 1j, -1 + 0j]])
    cU = qu.directSum(qc.i, u)
    cV = qu.directSum(qc.i, v)
    cZ = qu.directSum(qc.i, qc.z)
    cULayer = tensor(cU, qc.i)
    cVLayer = distant(cV)
    cZLayer = tensor(qc.i, cZ)
    ccNOT = application(cULayer, application(cZLayer, application(cVLayer, application(cZLayer, cVLayer))))
    return ccNOT

def groverR3():
    '''Assumes that n = 3. Returns -R, where R is Grover’s n-qbit gate for
    reflection across |rho>. Builds the gate from one- and two-qbit gates,
    rather than manually constructing the matrix.'''
    hLayer = power(qc.h, 3)
    xLayer = power(qc.x, 3)
    bottomH = tensor(power(qc.i, 2), qc.h)
    cZ = application(bottomH, application(ccNot(), bottomH))
    minusR = application(hLayer, application(xLayer, application(cZ, application(xLayer, hLayer))))
    return minusR

### DEFINING SOME TESTS ###

def distantTest():
    distantCNot = distant(qc.cnot)
    zeros = tensor(qc.ket0, tensor(qc.ket0, qc.ket0))
    middleOne = tensor(qc.ket0, tensor(qc.ket1, qc.ket0))
    firstOne = tensor(qc.ket1, tensor(qc.ket0, qc.ket0))
    middleZero = tensor(qc.ket1, tensor(qc.ket0, qc.ket1))
    if not qu.equal(application(distantCNot, zeros), zeros, 0.000001):
        print("failed distantTest on zeros")
    if not qu.equal(application(distantCNot, middleOne), middleOne, 0.000001):
        print("failed distantTest on middleOne")
    if not qu.equal(application(distantCNot, firstOne), middleZero, 0.000001):
        print("failed distantTest on firstOne")

def ccNotTest():
    gate = numpy.array([
            [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j]])
    if qu.equal(ccNot(), gate, 0.000001):
        print("passed ccNot test")
    else:
        print("failed ccNot test")
        print(ccNot())

def groverR3Test():
    triplePlus = power(qc.ketPlus, 3)
    def groverR3Result(state):
        return state - (2 * triplePlus * application(triplePlus.T, state))

    minusR = groverR3()
    testState = qu.uniform(3)
    if not qu.equal(application(minusR, testState), groverR3Result(testState), 0.00001):
        print("failed groverR3Test")
    else:
        print("passed groverR3Test")



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

def fourierTest(n):
    if n == 1:
        # Explicitly check the answer.
        t = fourier(1)
        if qu.equal(t, qc.h, 0.000001):
            print("passed fourierTest")
        else:
            print("failed fourierTest")
            print(" got T = ...")
            print(t)
    else:
        t = fourier(n)
        # Check the first row and column.
        const = pow(2, -n / 2) + 0j
        for j in range(2**n):
            if not qu.equal(t[0, j], const, 0.000001):
                print("failed fourierTest first part")
                print(" t = ")
                print(t)
                return
        for i in range(2**n):
            if not qu.equal(t[i, 0], const, 0.000001):
                print("failed fourierTest first part")
                print(" t = ")
                print(t)
                return
        print("passed fourierTest first part")
    # Check that T is unitary.
    tStar = numpy.conj(numpy.transpose(t))
    tStarT = numpy.matmul(tStar, t)
    id = numpy.identity(2**n, dtype=qc.one.dtype)
    if qu.equal(tStarT, id, 0.000001):
        print("passed fourierTest second part")
    else:
        print("failed fourierTest second part")
        print(" T^* T = ...")
        print(tStarT)

def fourierRecursiveTest(n):
    t = fourier(n)
    tRecursive = fourierRecursive(n)
    for i in range(2**n):
        for j in range(2**n):
            if not qu.equal(t[i,j], tRecursive[i,j], 0.000001):
                print("failed fourierRecursiveTest")
                print(" t =", t[i,j])
                print(" tRec =", tRecursive[i,j])
                return
    print("passed fourierRecursiveTest")
    return

### RUNNING THE TESTS ###

def main():
    applicationTest()
    applicationTest()
    tensorTest()
    tensorTest()
    functionTest(1,1)
    functionTest(2,2)
    functionTest(3,3)
    fourierTest(3)
    fourierRecursiveTest(1)
    fourierRecursiveTest(2)
    fourierRecursiveTest(3)
    fourierRecursiveTest(5)
    distantTest()
    ccNotTest()
    groverR3Test()
    return

if __name__ == "__main__":
    main()
