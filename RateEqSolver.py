import numpy as np
import matplotlib.pyplot as plt
import inspect as ins


class RESolver(object):
    """Class that makes an instance of a system of first
       order 1D differential equations. Works by reading the matrix
       corresponding to the system and optionally adds free terms.
       necessary variables:
       matrix: coefficient matrix, array of array (see readme for more
       information)
       initCond: the initial (or boundary) conditions, array
       extra variables:
       extra: array of free terms, array of functions and/ or scalar
       customrange; range of times within which the equations have
       to be solved, if default an automatic range is used, two float array."""
    def __init__(self, matrix, initCond, extra=[], customrange=[0, 0]):
        super(RESolver, self).__init__()
        self.matrix = matrix
        self.NLevels, self.couplings = np.shape(matrix)
        self.maxtau = float('inf')
        self.customrange = customrange
        for array in matrix:
            for r in array:
                if np.abs(r) < self.maxtau and r != 0:
                    self.maxtau = np.abs(r)
                else:
                    continue
        self.maxtau = 5 / self.maxtau
        self.mintau = 0.0
        if np.any(self.customrange) > 0:
            self.mintau = self.customrange[0]
            self.maxtau = self.customrange[1]

        self.t, self.dt = np.linspace(self.mintau, self.maxtau,
                                      1000, retstep=True)
        self.pop = np.zeros((self.NLevels, len(self.t)))
        self.diffPop = np.zeros((self.NLevels))
        for i, value in enumerate(initCond):
            self.pop[i, 0] = value
        for i in range(1, len(self.t)):
            for j in range((self.NLevels)):
                for k in range(self.couplings):
                    self.diffPop[j] += self.matrix[j][k] * self.pop[k, i - 1]
                if extra:
                    if ins.isfunction(extra[j]):
                        tmp = extra[j](self.t[i])
                        self.diffPop[j] += tmp
                    else:
                        self.diffPop[j] += extra[j]
                self.pop[j, i] = (self.pop[j, i - 1] +
                                  self.diffPop[j] * self.dt)
                self.diffPop = np.zeros((self.NLevels))






