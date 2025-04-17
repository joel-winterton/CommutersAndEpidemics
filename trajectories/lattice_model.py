import numpy as np

""" 
Implements a hexagonal lattice to be used in mEPR.
Contains coordinate system classes as well as lattice implementation.
"""


class Offset:
    def __init__(self, r, c):
        r, c = int(r), int(c)
        self.r = r
        self.c = c
        self.array = np.array([r, c])
        self.tuple = (r, c)


class Cube:
    def __init__(self, r, q, s):
        r, q, s = int(r), int(q), int(s)
        self.r = r
        self.q = q
        self.s = s
        self.array = np.array([r, q, s])
        self.tuple = (r, q, s)


class Coordinate:
    cube: Cube
    offset: Offset

    def __init__(self, offset: Offset = None, cube: Cube = None):
        if offset is None and cube is None:
            raise ValueError("You need to pass either offset or cube coordinates")

        if offset is not None:
            self.offset = offset
            self.cube = Cube(*oddr_to_cube(*offset.tuple))
        if cube is not None:
            self.cube = cube
            self.offset = Offset(*cube_to_oddr(*cube.tuple))

    def cartesian(self):
        """
        Return cartesian coordinates from offsets.
        This assumes an outer circle radius of 1 to avoid passing parameters everywhere.
        :return:
        """
        dx = np.sqrt(3)
        dy = 3 / 2
        a = self.offset.r % 2
        x = (a / 2 + self.offset.c) * dx
        y = self.offset.r * dy
        return x, y


def oddr_to_cube(r, c):
    """
    Convert (r,c) offset coordinates to cube coordinates (q,r,s).
    :param r:
    :param c:
    :return:
    """
    q = c - (r - r & 1) / 2
    return q, r, -q - r


def cube_to_oddr(q, r, s):
    """
    Convert cube coordinates (q,r,s) to (r,c) offset coordinates.
    :param q:
    :param r:
    :param s:
    :return:
    """
    return q + (r - (r & 1)) / 2, r


class Lattice:
    def __init__(self, width: int, population_avg: int = 100):
        assert (width > 0 & width % 2 == 0)
        self.r = width
        self.c = width
        self.grid = np.zeros((self.r, self.c))
        print("Assigning populus")
        self.population = np.random.normal(loc=population_avg, scale=2, size=self.grid.shape).astype(int)
        if np.sum(self.population.size > 0) < 0.5 * self.r * self.c:
            print("More than 50% of sites unpopulated!")

        print("Calculating distance matrix")
        self.distance_matrix = self.calculate_distance_matrix()
        print("Calculating exploration probabilities")
        self.exploration_probabilities = self.gravity_law()

    def validate_coord(self, coords: Coordinate):
        r, c = coords.offset.array
        return 0 <= coords.offset.r < self.r - 1 and 0 <= c < self.c - 1

    def gravity_law(self):
        """
        Pre-calculate gravity law as a probability of moving from site i to site j.
        """
        r, c = self.grid.shape
        mass1 = self.population[:, :, None, None]
        mass2 = self.population[None, None, :, :]

        gravity_matrix = mass1 * mass2 / np.power(self.distance_matrix, 2)

        # Create broadcasted index grids
        rows = np.arange(r)
        cols = np.arange(c)

        # Get all (r1, c1) and (r2, c2) coordinates
        r1, c1, r2, c2 = np.ix_(rows, cols, rows, cols)

        # Create mask: True where r1 == r2 and c1 == c2
        diag_mask = (r1 == r2) & (c1 == c2)

        # Zero out diagonals
        gravity_matrix[diag_mask] = 0

        row_sums = gravity_matrix.sum(axis=(2, 3), keepdims=True)
        row_sums[row_sums == 0] = 1
        return gravity_matrix / row_sums

    def calculate_distance_matrix(self):
        """
        Pre-calculate distance matrix between all lattice sites, index by offset coordinates.
        """
        rows, cols = self.grid.shape
        r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        q = c - (r - np.bitwise_and(r, 1)) / 2

        # arrays of cube coords
        x, y, z = q, r, -q - r

        # Reshape for broadcasting: (r1, c1, 1, 1)
        x1 = x[:, :, None, None]
        y1 = y[:, :, None, None]
        z1 = z[:, :, None, None]

        # Reshape for broadcasting: (1, 1, r2, c2)
        x2 = x[None, None, :, :]
        y2 = y[None, None, :, :]
        z2 = z[None, None, :, :]
        # Compute hex distance: max of component-wise differences
        dist = np.maximum.reduce([
            np.abs(x1 - x2),
            np.abs(y1 - y2),
            np.abs(z1 - z2)
        ])
        # make diagonals 1 since we'll chop them off later anyway, avoids division by zero.
        dist[dist == 0] = 1
        return dist.astype(int)

    def ball(self, center: Coordinate, distance: int):
        """
        Return all coordinates on lattice that are distance from center coordinate.
        :param center:
        :param distance:

        :return:
        """

        r1, q1, s1 = center.cube.array
        result = []

        for dr in range(-distance, distance + 1):
            for dq in range(-distance, distance + 1):
                ds = -dr - dq
                if abs(dr) + abs(dq) + abs(ds) == 2 * distance:
                    coord = Coordinate(cube=Cube(r1 + dr, q1 + dq, s1 + ds))
                    if self.validate_coord(coord):
                        result.append(coord)
        return np.array(result)
