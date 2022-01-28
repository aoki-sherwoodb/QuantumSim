# HW03, Ben Aoki-Sherwood, CS358

# Exercise A

import numpy

'''takes in two reals r, t corresponding to the magnitude and phase of a complex
number and returns the complex number re^(it)'''
def rect(r, t):
    return r * numpy.exp(numpy.array(0 + t * 1j))

# Exercise B

'''takes in two complex matrices A (p * q) and B (m * n) and returns a single
complex matrix with dimensions (p + m) * (q + n) with A and B arranged
along the diagonal and zeros elsewhere'''
def directSum(A, B):

    top = numpy.concatenate((A, numpy.zeros((A.shape[0], B.shape[1]), dtype='complex128')), axis=1)
    bottom = numpy.concatenate((numpy.zeros((B.shape[0], A.shape[1]), dtype='complex128'), B), axis=1)
    return numpy.concatenate((top, bottom))

print("Exercise A: testing rect()...")

print(rect(1, 0))   #expect (1 + 0j)
print(rect(1, numpy.pi / 4))    #expect (0.7 + 0.7j)
print(rect(78, numpy.pi))   #expect (-78 + 0j)
print(rect(1, numpy.pi * 1.25)) #expect (-0.7 - 0.7j)

print("Exercise B: testing directSum()...")

A = numpy.array([[1,2,3]])
B = numpy.array([[4,5],[6,7]])
C = numpy.array([[8],[9]])

print(directSum(A, B))  #expect 3x5 matrix with A in top left, B in bottom right
print(directSum(B, A))  #same as above but A,B swap position
print(directSum(C, B))  #expect 3x3 matrix with C in top left, B in bottom right
print(directSum(A, C))  #expect 3x4 with A in top left, C in bottom right
