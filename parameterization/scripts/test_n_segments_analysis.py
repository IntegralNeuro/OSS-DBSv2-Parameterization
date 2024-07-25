import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvista as pv
import simulation_analysis as sa
import impedance_model as im
import json
import os
from scipy.interpolate import make_smoothing_spline
import warnings
import time

DEBUG = False
THRESHOLD = 0.185
STIM_AMP = 0.005
R_MIN = 1.05
R_MAX = 1.1
DA = 5
DZ = 0.2

da = 1
dz = 0.1
wa = 3
wz = 0.3

json_filename = 'generated_ossdbs_parameters.json'
dir = '/Volumes/sim_data/results/experiment_n_contacts_1.5_0.5_0.1_0.0_floating_01M/'


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
data_dict = {}
for i, d in enumerate(dir_list):
    # Load the json file
    with open(os.path.join(d, json_filename), 'r') as f:
        config = json.load(f)

    # Dict
    data = get_geometry(config)
    # mesh data
    p_mesh = pv.read(os.path.join(d, 'potential.vtu'))
    mesh = pv.read(os.path.join(d, 'E-field.vtu'))
    mesh['potential_real'] = p_mesh['potential_real']
    x, y, z, a, r = set_coordinate_frame(mesh, data)

    R_MAX = 6.0

    # re-grid to horizontal slice
    hr = np.linspace(data['lead_diam'] / 2, R_MAX, 100)
    ha = np.linspace(-180, 180, 180)
    hz = np.linspace(0, 0, 1)
    hr, ha, hz = np.meshgrid(hr, ha, hz)
    hx = hr * np.cos(ha * np.pi / 180)
    hy = hr * np.sin(ha * np.pi / 180)
    h_grid = pv.StructuredGrid(hx, hy, hz)

    # re-grid to vertical slice
    vx = np.linspace(data['lead_diam'] / 2, R_MAX, 100)
    vy = np.linspace(0, 0, 1)
    vz = np.linspace(-3 * data['contact_length'], 3 * data['contact_length'], 150)
    vx, vy, vz = np.meshgrid(vx, vy, vz)
    v_grid = pv.StructuredGrid(vx, vy, vz)

    # Create the structured grids from the original meshes
    print(f'Interpolating set {i}...')
    start = time.time()
    h_mesh = h_grid.interpolate(mesh)
    print(f'Interpolation time (h): {time.time() - start}')
    start = time.time()
    v_mesh = v_grid.interpolate(mesh)
    print(f'Interpolation time (v): {time.time() - start}')

    data_dict[data['n_contacts']] = data
    data['horizontal'] = h_mesh
    data['vertical'] = v_mesh


# Make some plots
# Create a colormap with one color for each entry in dir_list
colors = mpl.colormaps['viridis'].resampled(len(data_dict))
plt.figure()
n_contacts_list = list(data_dict.keys())
n_contacts_list.sort()
current = np.zeros(len(n_contacts_list))
impedance = np.zeros(len(n_contacts_list))
power = 0
for i, n_contacts in enumerate(n_contacts_list):
    data = data_dict[n_contacts]

    # get electrode impedance, power, and current
    width = (data['contact_angle'] / 360) * data['lead_diam'] * np.pi
    area = width * data['contact_length'] * 1e6  # mm^2 to um^2
    impedance_rs_factor = 0.6
    impedance_dict = im.impedance_dict(area, material='Pt')
    impedance[i] = impedance_dict['Z'] - impedance_dict['Rs'] * (1 - impedance_rs_factor)
    if i == 0:
        power = STIM_AMP**2 * impedance[i]
        current[i] = STIM_AMP
    else:
        current[i] = np.sqrt(power / impedance[i])
        current[i] = STIM_AMP
    # loop the two through slices
    v_mesh = data_dict[n_contacts][slice_label]
    for j, slice_label in enumerate(['horizontal', 'vertical']):
        mesh = data_dict[n_contacts][slice_label]
        if slice_label == 'horiztonal':
            x_label = 'a [deg]'
            x_var = a
            x_binned = np.arange(-180 - da/2, 180 + da/2, da)
            x_plot = x_binned[:-1] + da / 2
            x_sm_win = da
            x_th_win = wa
        else:
            x_label = 'z [mm]'
            x_var = z
            x_binned = np.arange(-data['contact_length'] * 2 - dz/2, data['contact_length'] * 2 + dz/2, dz)
            x_plot = x_binned[:-1] + dz / 2
            x_sm_win = dz
            x_th_win = wz

        # Smooth potential and E-field
        sm_idx = ((r < R_MAX * data['lead_diam'] / 2) &
                  (r > R_MIN * data['lead_diam'] / 2))
        with warnings.catch_warnings(action='ignore'):
            v_binned = (np.histogram(x_var[sm_idx], x_binned, weights=v_mesh['potential_real'][sm_idx])[0] /
                        np.histogram(x_var[sm_idx], x_binned)[0]) * current[i]
            e_binned = (np.histogram(x_var[sm_idx], x_binned, weights=e_mesh['E_field_mag'][sm_idx])[0] /
                        np.histogram(x_var[sm_idx], x_binned)[0]) * current[i]

        # Threshold
        vta = np.full(np.size(x_plot), np.nan)
        th = (e_mesh['E_field_mag'] * current[i] >= THRESHOLD)
        for k, x_value in enumerate(x_plot):
            th_idx = (np.abs(x_var - x_value) < x_th_win / 2) & th
            if np.sum(th_idx) > 0:
                vta[k] = r[th_idx].max()
        # Smooth VTA
        spl_idx = np.isfinite(vta)
        # vta[spl_idx] = make_smoothing_spline(x_plot[spl_idx], vta[spl_idx])(x_plot[spl_idx])
        vta[spl_idx] = make_smoothing_spline(x_plot[spl_idx], vta[spl_idx], lam =(10000 if j==0 else 0.1))(x_plot[spl_idx])
        # mask and threshold
        # mask = (r < R_MAX * data['lead_diam'] / 2) & (r > R_MIN * data['lead_diam'] / 2)

        plt.subplot(2, 3, 3 * j + 1)
        # plt.plot(x_var[mask], v_mesh['potential_real'][mask], '.', color=colors(i), label=f'{n_contacts} contacts')
        plt.plot(x_plot, v_binned, color=colors(i), label=f'{n_contacts} contacts')
        plt.title('Potential')
        plt.ylabel('mV')
        plt.xlabel(x_label)

        plt.subplot(2, 3, 3 * j + 2)
        # plt.plot(x_var[mask], e_mesh['E_field_mag'][mask], '.', color=colors(i), label=f'{n_contacts} contacts')
        plt.plot(x_plot, e_binned, color=colors(i), label=f'{n_contacts} contacts')
        plt.title('E-Field')
        plt.ylabel('mV/m')
        plt.xlabel(x_label)

        plt.subplot(2, 3, 3 * j + 3)
        plt.plot(x_plot, vta, color=colors(i), label=f'{n_contacts} @ {1e3 * current[i]:0.2f}mA')
        plt.title('VTA')
        plt.ylabel('VTA')
        plt.xlabel(x_label)
plt.legend()
plt.show(block=False)

# Angle validation plot
if DEBUG:
    mask = ((r < R_MAX * data['lead_diam'] / 2) & (r > R_MIN * data['lead_diam'] / 2))
    # plt.plot(a[mask], z[mask], '.')
    plt.hist(a[mask], 360)
    for i in range(data['n_contacts']):
        a1 = data['gap_angle'] / 2 + i * (data['contact_angle'] + data['gap_angle'])
        if a1 > 180:
            a1 -= 360
        a2 = a1 + data['contact_angle']
        if a2 > 180:
            a2 -= 360
        plt.axvline(a1, color='r')
        plt.axvline(a2, color='b')
    plt.show()



# ################################################################################################### 


# re-grid to horizontal slice
R_MAX = 6.0
gr = np.linspace(data['lead_diam'] / 2, R_MAX, 100)
ga = np.linspace(-180, 180, 180)
gz = np.linspace(-3 * data['contact_length'], 3 * data['contact_length'], 150)
gr, ga, gz = np.meshgrid(gr, ga, gz)
gx = gr * np.cos(ga * np.pi / 180)
gy = gr * np.sin(ga * np.pi / 180)
grid = pv.StructuredGrid(gx, gy, gz)

# Create the structured grids from the original meshes
v_mesh = pv.StructuredGrid(mesh.points)
v_mesh['potential_real'] = mesh['potential_real']


# PV
print(f'Interpolating set {i}...')
start = time.time()
g_mesh_1 = grid.interpolate(v_mesh, radius=1)
print(f'Interpolation 1 - time (v): {time.time() - start}')
# 8655.8

print(f'Interpolating set {i}...')
start = time.time()
g_mesh_10 = grid.interpolate(v_mesh, radius=10)
print(f'Interpolation 10 - time (v): {time.time() - start}')
#

# Scipy
from scipy.interpolate import griddata, RBFInterpolator
import time

# Linear interpolation
print(f'Interpolating set {i}...')
start = time.time()
g_linear = griddata(v_mesh.points, mesh['potential_real'], grid.points, method='linear')
print(f'Linear time (v): {time.time() - start}')
# 70.90339


# Nearest interpolation
print(f'Interpolating set {i}...')
start = time.time()
g_nearest = griddata(v_mesh.points, mesh['potential_real'], grid.points, method='nearest')
print(f'Nearest time (v): {time.time() - start}')
# 6.676684617996216




fig = plt.figure()
dd = 0.04
vidx = (np.abs(v_mesh.points[:, 1]) < dd) & (np.abs(v_mesh.points[:, 2]) < dd)
gidx = (np.abs(grid.points[:, 1]) < dd) & (np.abs(grid.points[:, 2]) < dd)

plt.subplot(2,2,1)
plt.plot(v_mesh.points[vidx, 0], v_mesh['potential_real'][vidx], 'b.')
plt.plot(grid.points[gidx, 0], g_mesh_1['potential_real'[gidx], 'r.', alpha=0.1)
xlim = [grid.points[gidx, 0].min(), grid.points[gidx, 0].max()]
plt.xlim(xlim)
plt.yscale('log')
plt.title('1 vs. V')

plt.subplot(2,2,2)
plt.plot(v_mesh.points[vidx, 0], v_mesh['potential_real'][vidx], 'b.')
plt.plot(grid.points[gidx, 0], g_mesh_10['potential_real'][gidx], 'r.', alpha=0.1)
xlim = [grid.points[gidx, 0].min(), grid.points[gidx, 0].max()]
plt.xlim(xlim)
plt.yscale('log')
plt.title('10 vs. V')

plt.subplot(2,2,3)
plt.plot(v_mesh.points[vidx, 0], v_mesh['potential_real'][vidx], 'b.')
plt.plot(grid.points[gidx, 0], g_linear[gidx], 'r.', alpha=0.5)
xlim = [grid.points[gidx, 0].min(), grid.points[gidx, 0].max()]
plt.xlim(xlim)
plt.yscale('log')
plt.title('Linear vs. V')

plt.subplot(2,2,4)
plt.plot(v_mesh.points[vidx, 0], v_mesh['potential_real'][vidx], 'b.')
plt.plot(grid.points[gidx, 0], g_nearest[gidx], 'r.', alpha=0.5)
xlim = [grid.points[gidx, 0].min(), grid.points[gidx, 0].max()]
plt.xlim(xlim)
plt.title('Nearest vs. V')
plt.yscale('log')
plt.show(block=False)



plt.plot(v_mesh.points[:, 2], v_mesh['potential_real'], 'k.', alpha=0.1, label='linear')
plt.plot(grid.points[:, 2], g_mesh_10['potential_real'], 'g.', label='10')
plt.plot(grid.points[:, 2], grid_v_nearest, 'b.', label='nearest')
plt.plot(grid.points[:, 2], grid_v_linear, 'r.', label='linear')
plt.plot(v_mesh.points[:, 2], v_mesh['potential_real'], 'k.', alpha=0.1, label='linear')
plt.legend()
plt.show(block=False)



import pickle
with open('data_dict.pkl', 'wb') as f:
    pickle.dump(data_dict, f)