import math
import numpy
import matplotlib.pyplot as plt

import qConstants as qc
import qUtilities as qu

differences = [0] * 10000

for i in range(10000):
    ket_psi = qu.uniform(2)
    complex_diff = (ket_psi[0] * ket_psi[3]) - (ket_psi[1] * ket_psi[2])
    diff_psi = abs(numpy.conj(complex_diff) * complex_diff)
    differences[i] = diff_psi

differences.sort()
fig = plt.figure(figsize = (10, 7))
plt.hist(differences)
plt.title("Magnitude of difference from unentangled state")
plt.savefig("untentangled_hist.png")

print("Percentiles of differences:")
print("0%:", differences[0])
print("1%:", differences[99])
print("10%:", differences[999])
print("25%:", differences[2499])
print("50%:", differences[4999])
