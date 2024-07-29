import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvista as pv
import simulation_analysis as sa
import impedance_model as im
import json
import os
from scipy.interpolate import make_smoothing_spline, griddata, RBFInterpolator
import warnings
import time


# Set this
dir = '/Volumes/sim_data/results/experiment_n_contacts_1.5_0.5_0.1_0.0_floating_01M/'
json_filename = 'generated_ossdbs_parameters.json'


def get_geometry(config):
    # Get electrode geometry
    geom = {}
    if config['Electrodes'][0]['CustomParameters']['segmented_levels'] != []:
        geom['n_contacts'] = config['Electrodes'][0]['CustomParameters']['n_segments_per_level']
        geom['contact_angle'] = config['Electrodes'][0]['CustomParameters']['segment_contact_angle']
        geom['gap_angle'] = 360.0 / geom['n_contacts'] - geom['contact_angle']
    else:
        geom['n_contacts'] = 1
        geom['contact_angle'] = 360
        geom['gap_angle'] = 0
    geom['center_angle'] = geom['gap_angle'] / 2 + geom['contact_angle'] / 2
    geom['lead_diam'] = config['Electrodes'][0]['CustomParameters']['lead_diameter']
    geom['tip_length'] = config['Electrodes'][0]['CustomParameters']['tip_length']
    geom['contact_length'] = config['Electrodes'][0]['CustomParameters']['contact_length']
    geom['lead_rotation'] = config['Electrodes'][0]['Rotation[Degrees]']
    geom['tip_x'] = config['Electrodes'][0]['TipPosition']['x[mm]']
    geom['tip_y'] = config['Electrodes'][0]['TipPosition']['y[mm]']
    geom['tip_z'] = config['Electrodes'][0]['TipPosition']['z[mm]']
    return geom


def set_coordinate_frame(mesh, geom):
    x = mesh.points[:, 0] - geom['tip_x']
    y = mesh.points[:, 1] - geom['tip_y']
    z = mesh.points[:, 2] - geom['tip_z'] - geom['tip_length'] - geom['contact_length'] / 2
    a = np.arctan2(y, x) * 180 / np.pi - geom['lead_rotation']
    r = np.sqrt(x**2 + y**2)
    # shift mesh coordiante frame to match
    mesh.points[:, 0] = x
    mesh.points[:, 1] = y
    mesh.points[:, 2] = z
    return x, y, z, a, r


# Get data from all of the directories
dir_list = sa.recursive_dir_list(dir)
dir_list.sort()

# Iterate over dir_list and plot the data with corresponding color
i = 4
d = dir_list[i]

# Load the json file
with open(os.path.join(d, json_filename), 'r') as f:
    config = json.load(f)

# Dict
data = get_geometry(config)

print(f'Loading unstructured datasets {i}...')
p_mesh = pv.read(os.path.join(d, 'potential.vtu'))
mesh = pv.read(os.path.join(d, 'E-field.vtu'))
mesh['potential_real'] = p_mesh['potential_real']
x, y, z, a, r = set_coordinate_frame(mesh, data)

# re-grid to regular mesh
R_MIN = data['lead_diam'] / 2
R_MAX = 5.0
# gr = np.linspace(R_MIN, R_MAX, 100)
gr = 10**np.linspace(np.log10(R_MIN), np.log10(R_MAX), 100)
ga = np.array([data['contact_angle']])
# ga = np.linspace(-180, 179, 360)
gz = np.array([0])
# gz = np.linspace(-2 * data['contact_length'], 2 * data['contact_length'], 100)
gr, ga, gz = np.meshgrid(gr, ga, gz)
gx = gr * np.cos(ga * np.pi / 180)
gy = gr * np.sin(ga * np.pi / 180)
grid = pv.StructuredGrid(gx, gy, gz)

#  Mesh with only potential (for grid.interpolate)
v_mesh = pv.StructuredGrid(mesh.points)
v_mesh['potential_real'] = mesh['potential_real']

# COMPARE Interpolation Schemes
start = time.time()
g_lin = griddata(v_mesh.points, v_mesh['potential_real'], grid.points, method='linear')
print(f'Scipy\ttime (v): {time.time() - start}\t{(g_lin > 0).mean()}')

start = time.time()
g_mesh_01 = grid.interpolate(v_mesh, radius=1e-1)
g_mesh_01 = g_mesh_01['potential_real']
print(f'Interp 0.1\ttime (v): {time.time() - start}\t{(g_mesh_01 > 0).mean()}')

start = time.time()
g_mesh_05 = grid.interpolate(v_mesh, radius=5e-1)
g_mesh_05 = g_mesh_05['potential_real']
print(f'Interp 0.5\ttime (v): {time.time() - start}\t{(g_mesh_05 > 0).mean()}')

start = time.time()
g_mesh_1 = grid.interpolate(v_mesh, radius=1e0)
g_mesh_1 = g_mesh_1['potential_real']
print(f'Interp 1.0\ttime (v): {time.time() - start}\t{(g_mesh_1 > 0).mean()}')

start = time.time()
g_mesh_5 = grid.interpolate(v_mesh, radius=5e0)
g_mesh_5 = g_mesh_5['potential_real']
print(f'Interp 5.0\ttime (v): {time.time() - start}\t{(g_mesh_5 > 0).mean()}')

start = time.time()
g_mesh_10 = grid.interpolate(v_mesh, radius=1e2)
g_mesh_10 = g_mesh_10['potential_real']
print(f'Interp 10.0\ttime (v): {time.time() - start}\t{(g_mesh_10 > 0).mean()}')

vals = [g_lin, g_mesh_01, g_mesh_05, g_mesh_1, g_mesh_5, g_mesh_10]
strs = ['g_lin', 'g_mesh_01', 'g_mesh_05', 'g_mesh_1', 'g_mesh_5', 'g_mesh_10']

plt.figure()
col = 'krgbcm'
ct = 0
for i1, m1 in enumerate(vals):
    plt.plot(m1, color=col[i1], label=f"{strs[i1]}")
plt.legend()
plt.show()


"""
potential
Scipy	time (v): 80.4965648651123	1.0
Interp 0.1	time (v): 0.547684907913208	0.42
Interp 0.5	time (v): 1.7643358707427979	0.94
Interp 1.0	time (v): 5.053267955780029	1.0
Interp 5.0	time (v): 21.8784077167511	1.0
Interp 10.0	time (v): 25.311754941940308	1.0
"""