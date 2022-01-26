import math
import numpy

import qConstants as qc
import qBitStrings as qb
import qUtilities as qu
import qGates as qg

# It is conventional to have a main() function. Change it to do whatever you want. On Day 06 you could put your entanglement experiment in here.
def main():
    circuit1 = numpy.dot(qg.tensor(qc.h, qc.h), numpy.dot(qc.cnot, qg.tensor(qc.h, qc.h)))
    print(circuit1)
    circuit2 = numpy.dot(qc.swap, numpy.dot(qc.cnot, qc.swap))
    print(circuit2)

# If the user imports this file into another program as a module, then main() does not run. But if the user runs this file directly as a program, then main() does run.
if __name__ == "__main__":
    main()
