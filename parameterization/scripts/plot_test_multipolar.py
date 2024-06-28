import pickle
import numpy as np
from matplotlib import pyplot as plt
"""
No longer used.

Script for comparing the potential values across the various simulations in 
test_linearity.test_linearity_multipolar()

The goal of this script was to ensure that the resutls were qualitatively 
similar for each single-contact monopolar simulation. These plots helped identify 
the need for making homogenous brain models (e.g., segmask_white_matter.nii)
"""

exp_dir = 'results/experiment_linear_multipolar'
filename = 'data.pkl'
with open(exp_dir + '/' + filename, 'rb') as f:
    contacts, experiment_list = pickle.load(f)


names = []
vs = []
for name, expt in experiment_list.items():
    names.append(name)
    vs.append(expt['values'])

plt.figure(figsize=(18, 12))
ct = 0
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        ct += 1
        plt.subplot(2, 3, ct)
        plt.plot(vs[i], vs[j], '.')
        plt.xscale('log')
        plt.yscale('log')
        plt.axis('equal')
        plt.plot(plt.xlim(), plt.ylim(), 'k-')
        plt.xlabel(names[i])
        plt.ylabel(names[j])
plt.show()
