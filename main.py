# Slow remagnetization of ferrofluids. Effect of chain-like aggregates


import timeit
START = timeit.default_timer()


import espressomd
import espressomd.magnetostatics
espressomd.assert_features(['DIPOLES', 'DP3M', 'LENNARD_JONES'])
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import espressomd.observables
from espressomd.io.writer import vtf


print ('----------------------------\n\n')


PI = np.pi
LJ_SIGMA = 1.0
LJ_EPSILON = 1.0
LJ_CUT = 2 ** (1.0 / 6.0) * LJ_SIGMA
#N = 10000 # particles number
N = 512
#PHI = 0.03 # volume fraction
PHI = 0.03 # volume fraction
KT = 1.0 # Temperature
GAMMA = 1.0 # Friction coefficient
TIME_STEP = 0.01 # Time step
SEED = 3


#D = 13.4E-9 # particle diameter
D = 15E-9 # particle diameter
ETA = 0.13 # carrier fluid viscosity
H_0 = 0.1E3 # magnetic field
H_1 = 2E3 # magnetic field
H_2 = 3E3 # magnetic field
K_B = 1.38E-23 # Boltzmann constant
M_S = 5E5 # saturation magnetization of particle materials
MU_0 = 4.0 * PI * 1E-7 # vacuum permeability
S = 2E-9 # steric layer thickness
T = 298.0 # temperature


def FUNCTION_V(D, PI):
    return PI * D ** 3 / 6.0


# particle magnetic moment
def FUNCTION_M(M_S, V):
    return M_S * V


def FUNCTION_DIP_LAMBDA(D, K_B, M, MU_0, PI, T):
    return MU_0 / (4.0 * PI) * M ** 2 / (D ** 3 * K_B * T)


def FUNCTION_KAPPA(H, K_B, M, MU_0, T):
    return MU_0 * M * H / (K_B * T)


V = FUNCTION_V(D, PI)
M = FUNCTION_M(M_S, V)
DIP_LAMBDA = FUNCTION_DIP_LAMBDA(D, K_B, M, MU_0, PI, T)
KAPPA_0 = FUNCTION_KAPPA(H_0, K_B, M, MU_0, T)
KAPPA_1 = FUNCTION_KAPPA(H_1, K_B, M, MU_0, T)
KAPPA_2 = FUNCTION_KAPPA(H_2, K_B, M, MU_0, T)
TAU = 3.0 * PI * ETA * D ** 3 / (K_B * T)


print(f'tau = {TAU:.2f}')
print(f'dip_lambda = {DIP_LAMBDA:.2f}')
print(f'kappa_0 = {KAPPA_0:.2f}')
print(f'kappa_1 = {KAPPA_1:.2f}')
print(f'kappa_2 = {KAPPA_2:.2f}')


BOX_SIZE = LJ_SIGMA * (PI * N / (6.0 * PHI)) ** (1.0 / 3.0) # box size 3d


SYSTEM = espressomd.System(box_l = (BOX_SIZE, BOX_SIZE, BOX_SIZE))
SYSTEM.time_step = TIME_STEP


# Lennard-Jones interaction
SYSTEM.non_bonded_inter[0, 0].lennard_jones.set_params(epsilon = LJ_EPSILON, sigma = LJ_SIGMA, cutoff = LJ_CUT,
    shift = "auto")


# Random dipole moments
np.random.seed(seed = SEED)
DIP_PHI = 2.0 * PI * np.random.random((N, 1))
DIP_COS_THETA = 2.0 * np.random.random((N, 1)) - 1.0
DIP_SIN_THETA = np.sin(np.arccos(DIP_COS_THETA))
DIP = np.hstack((DIP_SIN_THETA * np.sin(DIP_PHI), DIP_SIN_THETA * np.cos(DIP_PHI), DIP_COS_THETA))


# Random positions in system volume
POS = BOX_SIZE * np.random.random((N, 3))


# Add particles
PARTICLES = SYSTEM.part.add(pos = POS, rotation = N * [(1, 1, 1)], dip = DIP)


# Remove overlap between particles by means of the steepest descent method
SYSTEM.integrator.set_steepest_descent(f_max = 0.0, gamma = 0.1, max_displacement = 0.05)


while SYSTEM.analysis.energy()["total"] > 5 * KT * N:
    SYSTEM.integrator.run(20)


# Switch to velocity Verlet integrator
SYSTEM.integrator.set_vv()
SYSTEM.thermostat.set_langevin(kT = KT, gamma = GAMMA, seed = SEED)


# tune verlet list skin
SYSTEM.cell_system.skin = 0.8


# Setup dipolar P3M
ACCURACY = 5E-4
SYSTEM.actors.add(espressomd.magnetostatics.DipolarP3M(accuracy = ACCURACY, prefactor = DIP_LAMBDA))


for i in tqdm.trange(10):
    SYSTEM.integrator.run(100)


STATE = open('vmd/state.vtf', mode = 'w+t')


# write structure block as header
vtf.writevsf(SYSTEM, STATE)


# remove all constraints
SYSTEM.constraints.clear()


LOOPS = 20

H_DIPM = KAPPA_0
TIME_MAX = 0.04
STEPS = int(TIME_MAX / (TAU * TIME_STEP * LOOPS))
H_FIELD = [H_DIPM, 0, 0]
H_CONSTRAINT = espressomd.constraints.HomogeneousMagneticField(H = H_FIELD)
SYSTEM.constraints.add(H_CONSTRAINT)


print('Establishment of equilibrium with an included magnetic field')
for i in tqdm.trange(LOOPS):
    SYSTEM.integrator.run(STEPS)
vtf.writevcf(SYSTEM, STATE)


# remove all constraints
SYSTEM.constraints.clear()


H_DIPM = KAPPA_2
TIME_MAX = 0.04
STEPS = int(TIME_MAX / (TAU * TIME_STEP * LOOPS))
H_FIELD = [H_DIPM, 0, 0]
H_CONSTRAINT = espressomd.constraints.HomogeneousMagneticField(H = H_FIELD)
SYSTEM.constraints.add(H_CONSTRAINT)


DIPM_TOT_CALC = espressomd.observables.MagneticDipoleMoment(ids = PARTICLES.id)
SYSTEM.time = 0.0
TIME = np.zeros(LOOPS + 1)
TIME_SI = np.zeros(LOOPS + 1)
MAGNETIZATION = np.zeros(LOOPS + 1)
MAGNETIZATION_SI = np.zeros(LOOPS + 1)


print('Calculation of magnetization')
for i in tqdm.trange(LOOPS):
    TIME[i] = SYSTEM.time
    MAGNETIZATION[i] = DIPM_TOT_CALC.calculate()[0] / N
    SYSTEM.integrator.run(STEPS)
vtf.writevcf(SYSTEM, STATE)

TIME[LOOPS] = SYSTEM.time
MAGNETIZATION[LOOPS] = DIPM_TOT_CALC.calculate()[0] / N


STATE.close()


for i in range(LOOPS + 1):
    MAGNETIZATION_SI[i] = PHI * M_S * MAGNETIZATION[i] / 1000.0
    TIME_SI[i] = TAU * TIME[i]



np.savetxt('data/magn_vs_time_SI.txt', np.c_[TIME_SI, MAGNETIZATION_SI],
    header = 'Time, sec             Magnetization, kA/m')


FIG = plt.figure(figsize = (6.0, 4.5))
plt.plot(TIME_SI, MAGNETIZATION_SI, 'o--', color = 'black', linewidth = 1.0, markersize = 4.0)
plt.xlabel('$t (s)$', fontsize = 16)
plt.ylabel('$M_F (kA/m)$', fontsize = 16)
plt.tight_layout()
plt.grid(True)
plt.savefig('figs/magn_vs_time_SI.pdf')


STOP = timeit.default_timer()
RUNTIME = STOP - START
print('\n\nTotal runtime: {:5.2f} sec = {:5.2f} min = {:5.2f} hours\n' .format(RUNTIME, RUNTIME / 60.0, RUNTIME / 3600.0))
