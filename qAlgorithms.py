
import random
import numpy
import math

import qConstants as qc
import qUtilities as qu
import qGates as qg
import qMeasurement as qm
import qBitStrings as qb



def bennett():
    '''Runs one iteration of the core algorithm of Bennett (1992). Returns a tuple of three items --- |alpha>, |beta>, |gamma> --- each of which is either |0> or |1>.'''
    if random.random() <= 0.5:  #A choosing |alpha> and sending |psi> to B
        ketAlpha = qc.ket0
        ketPsi = qc.ket0
    else:
        ketAlpha = qc.ket1
        ketPsi = qc.ketPlus
    if random.random() <= 0.5:  #B measures |psi> to yield |gamma>
        ketBeta = qc.ket0
        ketGamma = qm.first(qg.tensor(ketPsi, qc.ket0))[0]
    else:
        ketBeta = qc.ket1
        ketGamma = qm.first(qg.tensor(qg.application(qc.h, ketPsi), qc.ket0))[0]
    return (ketAlpha, ketBeta, ketGamma)



def deutsch(f):
    '''Implements the algorithm of Deutsch (1985). That is, given a two-qbit gate F representing a function f : {0, 1} -> {0, 1}, returns |1> if f is constant, and |0> if f is not constant.'''
    state = qg.tensor(qc.ket1, qc.ket1)
    hh = qg.tensor(qc.h, qc.h)
    state = qg.application(hh, state)
    state = qg.application(f, state)
    state = qg.application(hh, state)
    return qm.first(state)[0]

def bernsteinVazirani(n, f):
    '''Given n >= 1 and an (n + 1)-qbit gate F representing a function
    f : {0, 1}^n -> {0, 1} defined by mod-2 dot product with an unknown delta
    in {0, 1}^n, returns the list or tuple of n classical one-qbit states (each
    |0> or |1>) corresponding to delta.'''
    state = qg.power(qc.ket0, n)
    state = qg.tensor(state, qc.ket1)
    hLayer = qg.power(qc.h, n + 1)
    state = qg.application(hLayer, state)
    state = qg.application(f, state)
    state = qg.application(hLayer, state)
    delta = []
    for i in range(n):
        meas = qm.first(state)
        delta.append(meas[0])
        state = meas[1]
    return delta

def simon(n, f):
    '''The inputs are an integer n >= 2 and an (n + (n - 1))-qbit gate F
    representing a function f: {0, 1}^n -> {0, 1}^(n - 1) hiding an n-bit
    string delta as in the Simon (1994) problem. Returns a list or tuple of n
    classical one-qbit states (each |0> or |1>) corresponding to a uniformly
    random bit string gamma that is perpendicular to delta.'''

    state = qg.power(qc.ket0, n + n - 1)
    hLayer = qg.power(qc.h, n)
    firstLayer = qg.tensor(hLayer, qg.power(qc.i, n - 1))
    state = qg.application(firstLayer, state)
    state = qg.application(f, state)
    # measure last n - 1 qbits to disentangle the input and output registers
    for i in range(n - 1):
        state = qm.last(state)[0]
    state = qg.application(hLayer, state)
    # measure the input register to get gamma
    gamma = []
    for i in range(n):
        measurement = qm.first(state)
        gamma.append(measurement[0])
        state = measurement[1]
    return gamma

def shor(n, f):
    '''Assumes n >= 1. Given an (n + n)-qbit gate F representing a function
    f: {0, 1}^n -> {0, 1}^n of the form f(l) = k^l % m, returns a list or tuple
    of n classical one-qbit states (|0> or |1>) corresponding to the output of
    Shor’s quantum circuit.'''
    state = qg.power(qc.ket0, 2 * n)
    hLayer = qg.tensor(qg.power(qc.h, n), qg.power(qc.i, n))
    state = qg.application(hLayer, state)
    state = qg.application(f, state)
    for i in range(n):
        state = qm.last(state)[0]
    state = qg.application(qg.fourier(n), state)
    output = []
    for i in range(n):
        measurement = qm.first(state)
        output.append(measurement[0])
        state = measurement[1]
    return output

### DEFINING SOME TESTS ###

def shorTest(n, m):
    k = m
    while math.gcd(k, m) != 1:
        k = random.randint(1, m)

    def f(l):
        int_l = qb.integer(l)
        kToTheL = qu.powerMod(k, int_l, m)
        return qb.string(n, kToTheL)

    gate = qg.function(n, n, f)
    output = shor(n, gate)
    b = qb.integer(qb.statelist_to_string(output))
    print(b)
    return

def bennettTest(m):
    # Runs Bennett's core algorithm m times.
    trueSucc = 0
    trueFail = 0
    falseSucc = 0
    falseFail = 0
    for i in range(m):
        result = bennett()
        if qu.equal(result[2], qc.ket1, 0.000001):
            if qu.equal(result[0], result[1], 0.000001):
                falseSucc += 1
            else:
                trueSucc += 1
        else:
            if qu.equal(result[0], result[1], 0.000001):
                trueFail += 1
            else:
                falseFail += 1
    print("check bennettTest for false success frequency exactly 0")
    print("    false success frequency = ", str(falseSucc / m))
    print("check bennettTest for true success frequency about 0.25")
    print("    true success frequency = ", str(trueSucc / m))
    print("check bennettTest for false failure frequency about 0.25")
    print("    false failure frequency = ", str(falseFail / m))
    print("check bennettTest for true failure frequency about 0.5")
    print("    true failure frequency = ", str(trueFail / m))

def deutschTest():
    def fNot(x):
        return (1 - x[0],)
    resultNot = deutsch(qg.function(1, 1, fNot))
    if qu.equal(resultNot, qc.ket0, 0.000001):
        print("passed deutschTest first part")
    else:
        print("failed deutschTest first part")
        print("    result = " + str(resultNot))
    def fId(x):
        return x
    resultId = deutsch(qg.function(1, 1, fId))
    if qu.equal(resultId, qc.ket0, 0.000001):
        print("passed deutschTest second part")
    else:
        print("failed deutschTest second part")
        print("    result = " + str(resultId))
    def fZero(x):
        return (0,)
    resultZero = deutsch(qg.function(1, 1, fZero))
    if qu.equal(resultZero, qc.ket1, 0.000001):
        print("passed deutschTest third part")
    else:
        print("failed deutschTest third part")
        print("    result = " + str(resultZero))
    def fOne(x):
        return (1,)
    resultOne = deutsch(qg.function(1, 1, fOne))
    if qu.equal(resultOne, qc.ket1, 0.000001):
        print("passed deutschTest fourth part")
    else:
        print("failed deutschTest fourth part")
        print("    result = " + str(resultOne))

def bernsteinVaziraniTest(n):
    delta = qb.string(n, random.randrange(0, 2**n))
    def f(s):
        return (qb.dot(s, delta),)
    gate = qg.function(n, 1, f)
    qbits = bernsteinVazirani(n, gate)
    bits = tuple(map(qu.bitValue, qbits))
    diff = qb.addition(delta, bits)
    if diff == n * (0,):
        print("passed bernsteinVaziraniTest")
    else:
        print("failed bernsteinVaziraniTest")
        print("   delta = " + str(delta))

def simonTest(n):
    # Pick a non-zero delta uniformly randomly.
    delta = qb.string(n, random.randrange(1, 2**n))
    # Build a certain matrix M.
    k = 0
    while delta[k] == 0:
        k += 1
    m = numpy.identity(n, dtype=int)
    m[:, k] = delta
    mInv = m
    # This f is a linear map with kernel {0, delta}. So it’s a valid example.
    def f(s):
        full = numpy.dot(mInv, s) % 2
        full = tuple([full[i] for i in range(len(full))])
        return full[:k] + full[k + 1:]
    gate = qg.function(n, n - 1, f)

    gamma_matrix = []
    while len(gamma_matrix) < n - 1:
        gamma = simon(n, gate)
        gamma_row = [int(elt is qc.ket1) for elt in gamma]
        gamma_matrix.append(gamma_row)
        reduction = qb.reduction(gamma_matrix)
        if reduction[-1] == [0] * n:
            reduction.remove(reduction[-1])
        gamma_matrix = reduction

    print("FINAL GAMMA MATRIX: ", gamma_matrix)
    prediction = [1] * n
    #setting digits in prediction for rows with a single 1
    for i in range(len(gamma_matrix)):
        if gamma_matrix[i].count(1) == 1:
            zero_index = gamma_matrix[i].index(1)
            prediction[zero_index] = 0
    prediction = tuple(prediction)

    if delta == prediction:
        print("passed simonTest")
    else:
        print("failed simonTest")
        print(" delta = " + str(delta))
        print(" prediction = " + str(prediction))

### RUNNING THE TESTS ###

def main():
    # bennettTest(100000)
    # deutschTest()
    # bernsteinVaziraniTest(5)
    # simonTest(2)
    # simonTest(4)
    # simonTest(6)
    shorTest(4, 3)
    shorTest(4, 4)
    shorTest(5, 5)


if __name__ == "__main__":
    main()
