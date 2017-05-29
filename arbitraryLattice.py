import numpy as np
from math import floor, ceil
import pickle

class Lattice:

    def __init__(self, vec_a, vec_b, offset):
        self.vec_a = np.array(vec_a)
        self.vec_b = np.array(vec_b)
        self.vec_c = self.vec_a - self.vec_b
        self.len_a = np.linalg.norm(self.vec_a)
        self.len_b = np.linalg.norm(self.vec_b)
        self.len_c = np.linalg.norm(self.vec_c)
        self.ang_a = np.angle(self.vec_a[0] + 1j*self.vec_a[1])
        self.ang_b = np.angle(self.vec_b[0] + 1j*self.vec_b[1])
        self.ang_c = np.angle(self.vec_c[0] + 1j*self.vec_c[1])
        self.offset = np.array(offset)

    def getParams(self):
        return self.vec_a, self.vec_b, self.offset

    def getMinLatticeDist(self):
         return min(np.linalg.norm(self.vec_a), np.linalg.norm(self.vec_b))

    def getMaxLatticeDist(self):
         return max(np.linalg.norm(self.vec_a), np.linalg.norm(self.vec_b))

    def decompose(self, subject, vec_a, vec_b):
        x = np.transpose(np.array([vec_a, vec_b]))
        y = np.array(subject)
        ans = np.linalg.solve(x, y)

        return ans

    def getLatticePoints(self, x_min, x_max, y_min, y_max):
        offset = np.array(self.offset)
        corners = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]] # coordinates of the corners of the region
        displacements = [np.array(corner) - offset for corner in corners] # displacements of the region corners from the offset point

        dd = np.array([self.decompose(displacement, self.vec_a, self.vec_b) for displacement in displacements]) # dd = decomposed displacements

        a_min = floor(min(dd[:,0]))
        a_max = ceil(max(dd[:,0]))
        b_min = floor(min(dd[:,1]))
        b_max = ceil(max(dd[:,1]))

        points = []
        for i in range(a_min, a_max + 1):
            for j in range(b_min, b_max + 1):
                point = offset + self.vec_a*i + self.vec_b*j
                if point[0] > x_min and point[0] < x_max and point[1] > y_min and point[1] < y_max:
                    points.append(offset + self.vec_a*i + self.vec_b*j)

        return np.array(points)

    def getCoordinates(self, a, b):
        """Return the coordinates of a lattice point with the given indeces"""
        return self.offset + a*self.vec_a + b*self.vec_b

    def getIndices(self, x, y, roundIndices=True):
        """Return the indices of a lattice point with the given coordinates"""
        displacement = np.array([x, y]) - self.offset
        indices = self.decompose(displacement, self.vec_a, self.vec_b)
        if roundIndices:
            rounded = [round(index) for index in indices]
            indices = rounded

        return indices

    def save(self, filename):
        params = [self.vec_a, self.vec_b, self.offset]
        pickle.dump(params, open(filename, 'wb'))

def loadLattice(filename):
    vec_a, vec_b, offset = pickle.load(open( filename, 'rb'))
    lattice = Lattice(vec_a, vec_b, offset)

    return lattice

def makeLatticeByAngles(mag_a, ang_a, mag_b, ang_b, offset):
    from math import sin, cos

    vec_a = mag_a * np.array([cos(ang_a), sin(ang_a)])
    vec_b = mag_b * np.array([cos(ang_b), sin(ang_b)])
    lattice = Lattice(vec_a, vec_b, offset)

    return lattice
