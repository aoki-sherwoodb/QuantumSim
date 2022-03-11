

import random
import numpy

# Import the student's library.
import qConstants as qc
import qUtilities as qu
import qBitStrings as qb
import qGates as qg
import qMeasurement as qm
import qAlgorithms as qa

# Import my library. qBitStringsSol should be identical to qBitStrings.
# import qConstantsSol as qcs
# import qUtilitiesSol as qus
# import qBitStringsSol as qbs
# import qGatesSol as qgs
# import qMeasurementSol as qms
# import qAlgorithmsSol as qas

def simonTestSimple(n):
    # Pick a non-zero delta uniformly randomly.
    delta = qb.string(n, random.randrange(1, 2**n))
    # Let k be the index of the first 1 in delta.
    k = 0
    while delta[k] == 0:
        k += 1
    # This matrix M is always its own inverse mod 2.
    m = numpy.identity(n, dtype=int)
    m[:, k] = delta
    mInv = m
    # This f is a linear map with kernel {0, delta}. So it's a valid example.
    def f(s):
        full = numpy.dot(mInv, s) % 2
        full = tuple([full[i] for i in range(len(full))])
        return full[:k] + full[k + 1:]
    gate = qg.function(n, n - 1, f)
    # Check whether simon outputs a bit string perpendicular to delta.
    kets = qa.simon(n, gate)
    bits = tuple(map(qu.bitValue, kets))
    if qb.dot(bits, delta) == 0:
        print("passed simonTestSimple")
    else:
        print("failed simonTestSimple")
        print("    delta = " + str(delta))
        print("    bits = " + str(bits))

def gateTestSimple():
    first = numpy.matmul(qc.swap, numpy.matmul(qc.cnot, qc.swap))
    hh = qg.tensor(qc.h, qc.h)
    second = numpy.matmul(hh, numpy.matmul(qc.cnot, hh))
    if qu.equal(first, second, 0.000001):
        print("passed gateTestSimple")
    else:
        print("failed gateTestSimple")
        print("    first = " + str(first))
        print("    second = " + str(second))

def main():
    try:
        gateTestSimple()
    except:
        print("failed gateTestSimple")
        print("    fatal error")
    try:
        simonTestSimple(5)
    except:
        print("failed simonTestSimple")
        print("    fatal error")
    try:
        qa.shorTest(5, 5)
    except:
        print("failed shorTest")
        print("    fatal error")

if __name__ == "__main__":
    main()
