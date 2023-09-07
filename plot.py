# Slow remagnetization of ferrofluids. Effect of chain-like aggregates


import matplotlib.pyplot as plt
import numpy as np


TIME_SI, MAGNETIZATION_SI = np.loadtxt('data/magn_vs_time_SI.txt', comments = '#', unpack = True)
t_3, sum, M_F_3 = np.loadtxt('theoretical_data/M_F_vs_t_d_18_H_01_3.txt', comments = '#', unpack = True)
FIG = plt.figure(figsize = (6.0, 4.5))
plt.plot(TIME_SI, MAGNETIZATION_SI, 'o--', color = 'black', linewidth = 1.0, markersize = 4.0)
plt.plot(t_3, M_F_3, color = 'black', linewidth = 2.0)
plt.xlabel('$t (s)$', fontsize = 16)
plt.ylabel('$M_F (kA/m)$', fontsize = 16)
plt.tight_layout()
plt.grid(True)
plt.savefig('figs/magn_vs_time_SI.pdf')
