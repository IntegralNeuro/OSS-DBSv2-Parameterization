import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import json
import os
import glob
import time
import pandas as pd
import pickle
import datetime

import simulation_analysis as sa
import impedance_model as im
import numpy as np


# Parse a dict from an input json to get simulation geometry
def get_geometry(config):
    # Get electrode geometry
    geom = {}
    geom['contact_length'] = config['Electrodes'][0]['CustomParameters']['contact_length']
    geom['contact_spacing'] = config['Electrodes'][0]['CustomParameters']['contact_spacing']
    if config['Electrodes'][0]['CustomParameters']['segmented_levels'] != []:
        geom['n_contacts_per_level'] = config['Electrodes'][0]['CustomParameters']['n_segments_per_level']
        geom['contact_angle'] = config['Electrodes'][0]['CustomParameters']['segment_contact_angle']
        geom['gap_angle'] = 360.0 / geom['n_contacts_per_level'] - geom['contact_angle']
    else:
        geom['n_contacts_per_level'] = 1
        geom['contact_angle'] = 360
        geom['gap_angle'] = 0
    geom['encapsulation'] = config['Electrodes'][0]['EncapsulationLayer']['Thickness[mm]']
    geom['tip_x'] = config['Electrodes'][0]['TipPosition']['x[mm]']
    geom['tip_y'] = config['Electrodes'][0]['TipPosition']['y[mm]']
    geom['tip_z'] = config['Electrodes'][0]['TipPosition']['z[mm]']
    geom['lead_diam'] = config['Electrodes'][0]['CustomParameters']['lead_diameter']
    geom['tip_length'] = config['Electrodes'][0]['CustomParameters']['tip_length']
    geom['lead_rotation'] = config['Electrodes'][0]['Rotation[Degrees]']
    geom['tot_contacts'] = len(config['Electrodes'][0]['Contacts'])
    geom['levels'] = config['Electrodes'][0]['CustomParameters']['levels']
    geom['stim_contact'] = [i + 1 for i, c in enumerate(config['Electrodes'][0]['Contacts'])
                            if c['Current[A]'] > 0][0]
    geom['stim_level'] = 1 + (geom['stim_contact'] - 1) // geom['n_contacts_per_level']
    stim_segment = (geom['stim_contact'] - 1) % geom['n_contacts_per_level']  # 0-indexed
    geom['center_angle'] = (
        (geom['gap_angle'] + geom['contact_angle']) * (stim_segment + 0.5) +
        geom['lead_rotation']
    )
    # z is the center of the contact re elctrode, not taking tip location into account
    geom['center_z'] = (
        geom['tip_length'] +
        (geom['stim_level'] - 0.5) * geom['contact_length'] +
        (geom['stim_level'] - 1) * geom['contact_spacing']
    )
    return geom

# Read the input json files and make a dataframe with geometry data
def get_geom_df(dir_list, json_filename='generated_ossdbs_parameters.json'):
    geom_df = pd.DataFrame()
    for i, d in enumerate(dir_list):
        # Load the json file
        with open(os.path.join(d, json_filename), 'r') as f:
            config = json.load(f)
        geom = get_geometry(config)
        geom['set_dir'] = d.split('/')[-2]
        geom['expt_dir'] = d.split('/')[-1]
        geom['dir'] = d
        geom_df = geom_df.append(geom, ignore_index=True)
    return geom_df


def process_raw_experiment(results_dir='/Volumes/sim_data/results/n_contacts/',
                           json_filename='generated_ossdbs_parameters.json',
                           grid_filename='grid.vtk',
                           output_dir=None):
    RND = 8
    # Get data from all of the directories
    dir_list = sa.recursive_dir_list(results_dir)
    gdf = get_geom_df(dir_list, json_filename=json_filename)
    gdf = gdf.sort_values(by=['encapsulation', 'contact_length', 'n_contacts_per_level'], ignore_index=True)


    # Get the stim-contact-center slices for each experimen
    x_slices = []
    y_slices = []
    z_slices = []
    for i, gr in gdf.iterrows():
        # Load the json file
        # Get grid cylindrical coords - (r, angle, z)
        print(f'Loading grid for set {i}...', end='', flush=True)
        grid = pv.read(os.path.join(gr['dir'], grid_filename))
        grid.points -= np.array([gr['tip_x'], gr['tip_y'], gr['tip_z']])

        # Rototate so that stim center is at 0 degrees
        avec = np.arctan2(grid.y[1, :, 1], grid.x[1, :, 1]) * 180 / np.pi
        if gr['n_contacts_per_level'] == 1:
            aidx = 1  # any number will do
        else:
            aidx = np.argmin(np.abs(avec - gr['center_angle']))
        rgrid = grid.rotate_z(-avec[aidx])

        # make slices
        rgrid.points = np.round(rgrid.points, RND)  # otherwise slicing includes spurious points
        slices = rgrid.slice_orthogonal(0, 0, gr['center_z'])
        x_slices.append(slices[1])  # along the x-axis - orthgonal to y-axis
        y_slices.append(slices[0])  # along the y-axis - orthgonal to x-axis
        z_slices.append(slices[2])
        print(f' done.')


    # Save the data to a pickle file
    now = datetime.datetime.now().strftime("%y%m%d_%H%M")
    if output_dir == None:
        output_dir = results_dir
    output_file = os.path.join(output_dir, f'test_n_data_{now}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump({'x_slices': x_slices, 'y_slices': y_slices, 
                    'z_slices': z_slices, 'gdf': gdf}, f)

    return x_slices, y_slices, z_slices, gdf


"""
Make a figure with a grid of panels, where the panel rows are for each value of
'n_contacts_per_level' the pcolumns are for each value of 'contact_length'.
On each plot, the potential as a color map and isocontours of the E-field magnitude, for a set
values of the E-field magnitude.  The countours of the E-field magnitude the volumes of tissue
activated for that threshold.
"""
def make_contour_plots(gdf, x_slices, z_slices, output_dir=None):
    # Define the preset values of the E-field magnitude for isocontours
    stimulation_current = 5.0  # in mA
    thresholds = np.arange(0.8, 4.5, 0.6)  # in mV/mm

    # Get unique values of 'n_contacts_per_level' and 'contact_length' from gdf
    n_contact_vals = gdf['n_contacts_per_level'].unique()
    contact_length_vals = gdf['contact_length'].unique()

    # Create a grid of panels
    nrows = len(n_contact_vals)
    ncols = len(contact_length_vals)

    for slices in [x_slices, z_slices]:
        for encapsulation in [0.0, 0.1]:
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))
            # Loop through each combination of 'n_contacts_per_level' and 'contact_length'
            for i, n_contacts_per_level in enumerate(n_contact_vals):
                for j, contact_length in enumerate(contact_length_vals):

                    # Filter the data based on 'n_contacts_per_level' and 'contact_length'
                    gr = gdf[(gdf['n_contacts_per_level'] == n_contacts_per_level) &
                             (gdf['contact_length'] == contact_length) &
                             (gdf['encapsulation'] == encapsulation)]
                    index = gr.index[0]
                    s = slices[index]

                    # Transform data to mV and mV/mm
                    potential = np.array(s['potential_real']) * stimulation_current * 1e-3 # in V
                    efield = np.linalg.norm(s['E_field_real'], axis=1) * stimulation_current * 1e-3 # in V/mm

                    if slices == x_slices:
                        rvec = np.unique(s.points[:, 0])
                        zvec = np.unique(s.points[:, 2])
                        nr = len(rvec)
                        nz = len(zvec)
                        potential_m = potential.reshape(nz, nr)
                        efield_m = efield.reshape(nz, nr)
                        # only plot the contact side
                        potential_m = potential_m[:, rvec > 0]
                        efield_m = efield_m[:, rvec > 0]
                        rvec_pos = rvec[rvec > 0]

                        Y = zvec
                        X = rvec_pos
                        Cp = potential_m.T
                        Ce = efield_m.T
                    elif slices == z_slices:
                        a = np.arctan2(s.points[:, 1], s.points[:, 0]) * 180 / np.pi
                        r = np.sqrt(s.points[:, 0]**2 + s.points[:, 1]**2)
                        avec = np.unique(np.round(a, 5))
                        rvec = np.unique(np.round(r, 5))
                        na = len(avec)
                        nr = len(rvec)
                        if na == 361:
                            avec = avec[:-1]
                            na -= 1
                        potential_m = potential.reshape(na, nr)
                        efield_m = efield.reshape(na, nr)
                        Y = avec / 360 * gr['lead_diam'].values[0] * np.pi
                        X = rvec
                        Cp = potential_m.T
                        Ce = efield_m.T

                    # Plot
                    ax = axs[i, j]
                    ax.pcolormesh(Y, X, Cp[:-1, :-1], cmap='viridis')
                    ax.contour(Y, X, Ce, levels=thresholds, cmap='binary')

                    if slices == x_slices:
                        # Set axis labels and title
                        z_center = gr.center_z.values[0]
                        ax.set_xlim([z_center - 3, z_center + 3])
                        ax.set_ylim([0.65, 3])
                        if i == len(n_contact_vals) - 1:
                            ax.set_xlabel('x (mm)')
                        if j == 0:
                            ax.set_ylabel('z (mm)')
                        ax.set_aspect('equal')
                    elif slices == z_slices:
                        ax.set_ylim([0.65, 3])
                        if i == len(n_contact_vals) - 1:
                            ax.set_xlabel('circumference (mm)')
                        if j == 0:
                            ax.set_ylabel('r (mm)')
                        ax.set_aspect('equal')
                    ax.text(0.02, 0.95, f'n_contacts={n_contacts_per_level}\ncontact_length={contact_length}',
                            transform=ax.transAxes, color='white', fontsize=8,
                            verticalalignment='top', horizontalalignment='left')

            # Title
            if slices == x_slices:
                axstr = "Vertical-radial slice at contact-center"
                sstr = 'vertical'
            else:
                axstr = "Horizonal slice at contact-center, in polar-coords"
                sstr = 'horizontal'
            if encapsulation == 0:
                estr = "No encapsulation"
            else:
                estr = f"Encapsulation: {encapsulation}mm "
            fig.suptitle(f'Potential & E-field VTAs: {axstr}\n' + 
                         f'Stim: {stimulation_current} mA, Thresholds: {thresholds}V/mm,  {estr}', fontsize=14)
            plt.show(block=False)
            if output_dir is not None:
                now = datetime.datetime.now().strftime("%y%m%d_%H%M")
                outfile = os.path.join(output_dir, f'contours_{sstr}_{encapsulation}_{now}.pdf')
                plt.savefig(outfile)


def rotate_grid_array(angle, grid, array=None):
    # Shift the angle-index of the data
    # to effectively rotate array data by the
    # specified angle (rounded to integer index shift)
    #
    # This uses the fact that the grid was set up as a
    # cylindrical StucturedGrid with angle as the second index
    # (see json_modification_api.py)
    nz, na, nr = grid.x.shape
    shape = grid[array].shape
    if len(shape) == 1:
        data = grid[array].reshape(nr, na, nz)
    else:
        data = grid[array].reshape(nr, na, nz, shape[1])
    aidx_shift = - np.round(angle * 360 / na).astype(int)
    data =  np.roll(data, aidx_shift, axis=1)
    return data.reshape(shape)

def replicate_frey_2022(gdf, x_slices, z_slices, output_dir=None,
                        threshold_type='E_field',  # []'E_field', 'Hessian']
                        threshold_efield=0.5,  # in V/mm
                        threshold_hessian=0.10, # in V/mm^2
                        threshold=None,
                        encapsulation=0.0):
    # Define the preset values of the E-field magnitude for isocontours
    stimulation_currents = [1.0, 2.0, 3.0]   # in mA
    if threshold is not None:
        if threshold_type == 'E_field':
            threshold_efield = threshold
        elif threshold_type == 'Hessian':
            threshold_hessian = threshold

    # Get unique values of 'n_contacts_per_level' and 'contact_length' from gdf
    electrodes = ['Standard', 'Vercise']
    n_contacts = [1, 3]
    contact_length = 1.5

    thresh = [[[], [], []], [[], [], []]]
    plotter = pv.Plotter(shape=(2, 2), window_size=[2048, 1536])
    for ic, contact in enumerate(n_contacts):
        # Filter the data based on 'n_contacts_per_level' and 'contact_length'
        gr = gdf[(gdf['n_contacts_per_level'] == contact) &
                 (gdf['contact_length'] == contact_length) &
                 (gdf['encapsulation'] == encapsulation)]
        dir = gr['dir'].values[0]
        electrode = pv.read(os.path.join(dir, 'electrode_1.vtu'))
        grid = pv.read(os.path.join(dir, 'grid.vtk'))
        if np.abs(grid.points[:,0].mean()) > 1e-3:
            grid.points -= gr[['tip_x', 'tip_y', 'tip_z']].values
            electrode.points -= gr[['tip_x', 'tip_y', 'tip_z']].values

        # Grid manipulations
        if contact == 3:
            # If Vercise, rotate the grid twice to simulate the 3 contacts
            for array in grid.array_names:
                pass
                # Multipolar setup: 0.5 * (A + B) - C
                # ga120 = rotate_grid_array(120, grid, array=array)
                # ga240 = rotate_grid_array(240, grid, array=array)
                # grid[array] = 0.5 * grid[array] + 0.5 * ga120  - ga240
        if threshold_type == 'E_field':
            grid['E_field_mag'] = np.linalg.norm(grid['E_field_real'], axis=1) * 1e-3  # in V/mm @ 1mA stim
        elif threshold_type == 'Hessian':
            grid = sa.hessian_mesh_from_efield(grid)
            grid['eigenvalues'] = grid['eigenvalues'] * 1e-3  # in V/mm^2
        else:
            raise ValueError(f'Unknown threshold type: {threshold_type}')
        
        # Get the threshold meshes
        for ks, stim in enumerate(stimulation_currents):
            if threshold_type == 'E_field':
                threshold = threshold_efield
                thresh[ic][ks] = grid.threshold(value=threshold / stim,  # lower threshold for higher stim
                                                scalars='E_field_mag',
                                                continuous=True,
                                                method='lower')
            elif threshold_type == 'Hessian':
                threshold = threshold_hessian
                thresh[ic][ks] = grid.threshold(value=threshold / stim,  # lower threshold for higher stim
                                                scalars='eigenvalues',
                                                component_mode='component',
                                                component=0,
                                                continuous=True,
                                                method='lower')
            else:
                raise ValueError(f'Unknown threshold type: {threshold_type}')

        for js, slices in enumerate([x_slices, z_slices]):
            # Plot
            plotter.subplot(js, ic)
            plotter.add_mesh(electrode, scalars='boundaries')
            opacity = np.linspace(0.05, 0.15, len(stimulation_currents))[::-1]
            for ks in range(len(stimulation_currents)):
                plotter.add_mesh(thresh[ic][ks], color='blue', opacity=opacity[ks])
            DX = 5
            if slices == x_slices:
                # Set axis labels and title
                z_center = gr.center_z.values[0]
                plotter.set_position([0, -0.1, z_center])
                plotter.view_xz(negative=True)
                plotter.reset_camera(bounds=[-DX, DX, 0, 0, z_center - DX, z_center + DX])
            elif slices == z_slices:
                plotter.set_position([0, 0, -0.1])
                plotter.view_xy(negative=True)
                plotter.reset_camera(bounds=[-DX, DX, -DX, DX, 0, 0])
                # Pyvista can be sooooo frustrating: saved graphcs don't respect '\n' in strings
                # so let's recreate the newline!
                lines = [f'n_contacts: {contact}',
                         f'current: {stimulation_currents} mA',
                         f'threshold: {threshold} V/mm',
                         f'encapsulation: {encapsulation} mm']
                for il, line in enumerate(lines):
                    plotter.add_text(line, position=[20, 10 + 35 * il], color="black", font_size=12)
                plotter.add_text(f'{electrodes[ic]}', position="upper_left", color="black", font_size=12)

    if output_dir is not None:
        now = datetime.datetime.now().strftime("%y%m%d_%H%M")
        outfile = os.path.join(output_dir, f'frey_{threshold_type}{threshold}_encap{encapsulation}_{now}.pdf')
        plotter.save_graphic(outfile, painter=True)

    plotter.show() # interactive_update=True)



if __name__ == '__main__':

    # arguments
    RERUN = False
    RELOAD = True
    DO_CONTOURS = False  # make contour plots
    DO_FREY = True  # replicate Frey 2022
    results_dir = '/Volumes/sim_data/results/n_contacts/'
    json_filename = 'generated_ossdbs_parameters.json'
    grid_filename = 'grid.vtk'
    output_dir = os.path.join(results_dir, 'analysis')

    # Get processed data
    if (not RERUN) and (RELOAD or ('gdf' not in locals())):
        data_filename = glob.glob(os.path.join(output_dir, 'test_n_data_*.pkl'))
        if len(data_filename) == 0:
            RERUN = True
        else:
            data_filename.sort()
            data_filename = data_filename[-1]
            print(f'Loading data from {data_filename}...')
            with open(data_filename, 'rb') as f:
                data_dict = pickle.load(f)
            gdf = data_dict['gdf']
            x_slices = data_dict['x_slices']
            y_slices = data_dict['y_slices']
            z_slices = data_dict['z_slices']
    if RERUN:
        x_slices, y_slices, z_slices, gdf = \
            process_raw_experiment(results_dir=results_dir,
                                   json_filename=json_filename,
                                   grid_filename=grid_filename,
                                   output_dir=output_dir)

    # Do analyses
    if DO_CONTOURS:
        make_contour_plots(gdf, x_slices, z_slices, output_dir=output_dir)

    # Replicate Frey 2022
    if DO_FREY:
        for threshold_type in ['E_field', 'Hessian']:
            for encapsulation in [0.0, 0.1]:
                replicate_frey_2022(gdf, x_slices, z_slices, output_dir=output_dir,
                                    threshold_type=threshold_type,  # []'E_field', 'Hessian']
                                    threshold_efield=0.5,  # in V/mm
                                    threshold_hessian=0.10, # in V/mm^2
                                    threshold=None,
                                    encapsulation=encapsulation)
