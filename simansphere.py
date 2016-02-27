import numpy as np
from math import exp, sqrt


def _delta_func(temperature):
    if temperature < 10 ** (-3):
        iterations_per_T = 10000
        if temperature > 10 ** (-9):
            delta = 0.0001
        else:
            delta = 0.000001
    else:
        iterations_per_T = 2000
        delta = 0.1

    return delta, iterations_per_T


def _points_on_sphere(N):
    """Chooses N coordinates with uniform distribution over sphere surface.
    Returns an array of said coordinates.
    """
    points = []
    while len(points) < N:
        x_i = np.random.uniform(-1, 1)
        y_i = np.random.uniform(-1, 1)
        if x_i ** 2 + y_i ** 2 < 1:
            x = 2 * x_i * sqrt(1 - x_i ** 2 - y_i ** 2)
            y = 2 * y_i * sqrt(1 - x_i ** 2 - y_i ** 2)
            z = 1 - 2 * (x_i ** 2 + y_i ** 2)
            points.append((x, y, z))
    return points


def _energy_of_charges_on_sphere(N, points):
    """
    Computes the electrostatic energy of a system of point charges on a
    sphere. Each term in the sum over particle pairs is stored in an N x N
    matrix. The matrix is kept upper-triangular since the central diagonal
    consists of only  zeroes and as the lower-triangle is indentical to the
    upper triangle, it need not be calculated.

    :param: N: int - number of charges
    :param: points: array - coordinates of point charges

    Returns the total electrostatic energy of the system (int) and a matrix, M
    (a numpy array of arrays) where the i-j-th element of M is the component of
    this total due to the electrostatic energy between charges i and j.
    """
    W_matrix = np.zeros((N, N))
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            diff_init = [
                points[i][0] - points[j][0],
                points[i][1] - points[j][1],
                points[i][2] - points[j][2]
            ]
            W_matrix[i][j] = 1 / sqrt(
                diff_init[0] ** 2 + diff_init[1] ** 2 + diff_init[2] ** 2)
    W = np.sum(W_matrix)

    return W, W_matrix


def _recompute_energy_components(N, energy_matrix, points, charge_index):
    """Given the new charge positions, replaces the elements of energy_matrix
    involving the given charge with the recomputed energy components.
    energy_matrix is modified in place.
    """
    for j in range(charge_index + 1, N):
        diff = [
            points[charge_index][0] - points[j][0],
            points[charge_index][1] - points[j][1],
            points[charge_index][2] - points[j][2]
        ]
        energy_matrix[charge_index][j] = \
            1 / sqrt((diff[0]) ** 2 + diff[1] ** 2 + diff[2] ** 2)
    for k in range(0, charge_index):
        diff = [
            points[charge_index][0] - points[k][0],
            points[charge_index][1] - points[k][1],
            points[charge_index][2] - points[k][2]
        ]
        energy_matrix[k][charge_index] = \
            1 / sqrt((diff[0]) ** 2 + (diff[1]) ** 2 + (diff[2]) ** 2)


def _move_charge_and_recompute_energy(N, W_matrix, points, delta):
    """Moves a randomly chosen charge in a random direction, by a distance delta
    (normalising it back onto the sphere's surface) and computes the
    electrostatic energy of the modified system.

    :param: N: int - number of charges
    :param: W_matrix: numpy array - energy matrix
    :param: points: numpy array - list of charge positions
    :param: delta: int - distance by which to move charge

    Returns the new total energy, energy matrix and array of charge coordinates.
    """
    W_matrix_mod = np.copy(W_matrix)
    points_mod = np.copy(points)
    charge_index = np.random.randint(0, N)
    move_vector = np.array([
        np.random.uniform(-1, 1),
        np.random.uniform(-1, 1),
        np.random.uniform(-1, 1)
    ])
    # normalise to delta
    move_vector = delta * move_vector / sqrt(
        move_vector[0] ** 2 + move_vector[1] ** 2 +
        move_vector[2] ** 2
    )
    # move the charge
    points_mod[charge_index] = points_mod[charge_index] + move_vector
    # keep the charge on the surface of the sphere
    norm = (points_mod[charge_index][0]) ** 2 + \
        (points_mod[charge_index][1]) ** 2 + (points_mod[charge_index][2]) ** 2
    points_mod[charge_index] /= sqrt(norm)

    # recompute changed elements of W_matrix for charge's row and column
    _recompute_energy_components(N, W_matrix_mod, points_mod, charge_index)
    # recompute the energy
    W_mod = np.sum(W_matrix_mod)

    return W_mod, W_matrix_mod, points_mod


def point_charges_on_unit_sphere_min_energy(
        N, initial_T, P, T_terminate, delta_func):
    u"""
    The minimum energy configuration of N point unit charges on the surface of a
    unit sphere is found by simulated annealing.

    The method is as follows:
    1. The initial configuration of N point charges on the unit sphere is
    chosen at random and a sufficiently high temperature T is chosen.

    2. One of the charges is chosen at random and moved in a random direction
    by a distance delta.

    3. The new energy of the system, W is evaluated. If the energy has reduced
    the move is accepted, otherwise the move is accepted with a probability
    proportional to exp(−∆W/T) in order to allow the system to escape non-global
    minima.

    4. Steps 2–3 are repeated a number of times given by iterations_per_temp.

    5. T is reduced by a percentage P and steps 1–4 are repeated until T is
    greater than a termination temperature, end_temp.

    :param: N: int - number of charges
    :param: initial_T: int - initial temperature
    :param: P: int - percentage by which to reduce T after iterations_per_temp
    iterations (controls the rate of annealing)
    :param: T_terminate: int - temperature at which to terminate annealing
    :param: delta_func: function(T) , T dependent function returning the
    distance, (delta - int) by which to move the charge and the number of
    iterations per temperature (iterations_per_T - int)

    (T is in units KR/boltzmann_constant*q**2
     W is in units q**2/gas_constant)

    - - - - -
    More on the delta_func:

    To increase the likelihood of finding the global minima, the system must be
    annealed slowly. As N increases, the rate of annealing must be reduced by
    choosing a smaller P value. This is a consequence of the number of
    local minima being greatly increased for larger N. A greater number of
    iterations per temperature value, iterations_per_T, also affords the system
    more opportunity to escape non-optimal minima and hence, in general,
    iterations_per_T must also be increased with N.

    Choosing a small delta value allows for more detailed exploration but
    increases the probability of the system becoming unable to escape a
    sub-optimal configuration. At high temperatures, the charges should be
    allowed to "explore" the full extent of the surface, with a smaller delta
    being chosen at lower temperatures in order to refine the configuration.
    _ _ _ _ _

    :return: minimum energy and the associated configuration.
    """
    assert T_terminate < initial_T, "termination temperature must be lower " \
        "than initial temperature"

    points = np.array(_points_on_sphere(N))

    # store all the configurations and associated energies as we go
    config_log = []
    W_log = []

    T = initial_T

    W, W_matrix = _energy_of_charges_on_sphere(N, points)

    while T > T_terminate:
        delta, iterations_per_T = delta_func(T)

        for i in range(1, iterations_per_T):

            W_mod, W_matrix_mod, points_mod = \
                _move_charge_and_recompute_energy(N, W_matrix, points, delta)

            # is the move accepted?
            if W_mod < W or np.random.random() < exp(-(W_mod - W) / T):
                W = W_mod
                W_log.append(W)
                W_matrix = np.copy(W_matrix_mod)
                points = np.copy(points_mod)
                config_log.append(points)

        T = 1 - (P * 0.01) * T

    # Take the optimum of all explored configurations
    min_w = min(W_log)
    min_points = config_log[W_log.index(min_w)]

    return min_w, min_points


# VARIABLES
initial_T = 1000
T_terminate = 10 ** (-18)
P = 0.1
Nmin = 1
Nmax = 4


# RUN
def run():
    min_energies = {}
    for N in range(Nmin, Nmax):
        min_energies[N] = point_charges_on_unit_sphere_min_energy(
            N, initial_T, P, T_terminate, _delta_func)
        fo = open("output_{0}to{1}_2.txt".format(Nmin, Nmax), "w")
        for N, data in min_energies.items():
            energy, config = data
            fo.write('{N}: {energy}' .format(N=N, energy=energy))
            fo.write('{config}\n'.format(config=config))
        fo.close()
