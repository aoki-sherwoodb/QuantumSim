


import random

import qConstants as qc
import qUtilities as qu
import qGates as qg
import qMeasurement as qm



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



### DEFINING SOME TESTS ###

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



### RUNNING THE TESTS ###

def main():
    bennettTest(100000)
    deutschTest()

if __name__ == "__main__":
    main()
