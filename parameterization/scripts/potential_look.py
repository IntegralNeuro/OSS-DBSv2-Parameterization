import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import csv
from matplotlib import pyplot as plt


dir = 'results/electrode_1_20240618_182403'
vtu_file = dir + '/potential.vtu'
csv_file = dir + '/oss_potentials_Lattice.csv'

# vtu
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(vtu_file)
reader.Update()  # Needed because of GetScalarRange
output = reader.GetOutput()
datv = vtk_to_numpy(output.GetPointData().GetArray("potential_real"))

# csv
datc = []
with open(csv_file) as f:
    reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(reader):
        if i>1:
            datc.append(float(row[4]))
datc = np.array(datc)
datv = np.array(datv)

plt.plot(datv, 'r')
plt.plot(datc, 'b')
plt.show()