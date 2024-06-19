import numpy as np
import pyvista as pv
import csv
from matplotlib import pyplot as plt



dir = 'results/electrode_1_20240619_100924'
vtu_file = dir + '/potential.vtu'
csv_file = dir + '/oss_potentials_Lattice.csv'

# vtu
mesh = pv.read(vtu_file)
v_vtu = mesh['potential_real']
p_vtu = mesh.points

# csv
# index,x-pt,y-pt,z-pt,potential,inside_csf,inside_encap,frequency
vlist = []
plist = []
with open(csv_file) as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        if i>1:
            vlist.append(float(row[4]))
            plist.append(np.array([float(row[1]), float(row[2]), float(row[3])]))
v_csv = np.array(vlist)
p_csv = np.vstack(plist)

labels = ['x', 'y', 'z']
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot(p_vtu[:, i], v_vtu, 'b.', label='')
    plt.xlabel(labels[i])
    plt.plot(p_csv[:, i], v_csv, 'r.', label='CSV' )
    plt.xlabel(labels[i])
plt.show(block=True)
