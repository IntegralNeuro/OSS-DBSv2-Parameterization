import argparse
import numpy as np
from json_modification_api import CustomElectrodeModeler
import simulation_analysis as sa
from matplotlib import pyplot as plt
import pyvista as pv  # type: ignore
import os


def simulate_all_contacts(dir_name, args) -> None:
    """
    Look at single-contact stimulation for all contacts
    """
    # Which modes to simulate
    if args.stim_mode == 'current':
        stim_mode_list = ['current']
    elif args.stim_mode == 'voltage':
        stim_mode_list = ['voltage']
    else:
        stim_mode_list = ['current', 'voltage']

    # Set up electrode
    segment_contact_angle = args.contact_ratio * 360.0 / args.n_segments_per_level

    electrode = CustomElectrodeModeler("ossdbs_input_config")
    electrode.modify_electrode_custom_parameters(
        total_length=300.0,
        contact_length=args.contact_length,
        contact_spacing=args.contact_spacing,
        segment_contact_angle=segment_contact_angle,
        n_segments_per_level=args.n_segments_per_level,
        levels=args.levels,
        segmented_levels=list(range(1, args.levels + 1)),
        tip_contact=False,
    )
    # Simulate each contact
    n_contacts = electrode.get_electrode_custom_parameters()["_n_contacts"]
    for stim_mode in stim_mode_list:
        for contact in range(1, n_contacts + 1):
            name = f"{stim_mode}_{contact}"
            electrode.generate_output_path(f"{dir_name}/{name}")
            electrode.generate_current_contact(contact, 1.0)
            for i in range(n_contacts):
                if not (i + 1) == contact:
                    electrode.generate_voltage_contact(i + 1, 1.0)
            electrode.update_parameters()
            electrode.modify_json_parameters()
            electrode.run_ossdbs()


def read_experiment_output(dir_name: str) -> dict:
    """
    Read in the experiment output data.

    Args:
        dir_name (str): The directory name of the experiment output.

    Returns:
        dict: The experiment output data
            - keys: stim_mode
            - values: dict
                - keys: contact
                - values: dict
                    - pts: np.ndarray: The grid points, shape (n_points, 3)
                    - values: list of np.ndarray, each (n_points, dim)
                    - names: list of array names
    """
    data_dict = {}
    for subdir in os.listdir('results/' + dir_name):
        subdir_path = os.path.join('results', dir_name, subdir)
        stim_mode, contact, _, _ = subdir.split("_")
        contact = int(contact)
        if stim_mode not in data_dict:
            data_dict[stim_mode] = {}
        if os.path.isdir(subdir_path):
            vtu_file = os.path.join(subdir_path, "potential.vtu")
            if os.path.isfile(vtu_file):
                pts, values, names = sa.load_vtu(vtu_file)
                data_dict[stim_mode][contact] = {'pts': pts, 'values': values, 'names': names}
    return data_dict


def create_scatter_plots(data_dict):
    stim_modes = list(data_dict.keys())
    for m, stim_mode in enumerate(stim_modes):
        # Get contact list
        contacts = np.sort(list(data_dict[stim_mode].keys()))
        n_contacts = len(contacts)
        # Scatter plot grid
        n_plots = n_contacts * (n_contacts - 1) // 2
        nrow = np.floor(np.sqrt(n_plots)).astype(int)
        ncol = np.ceil(n_plots / nrow).astype(int)
        # Create scatter plots
        plt.figure(figsize=(15, 12))
        ct = 0
        for i in range(n_contacts):
            for j in range(i + 1, n_contacts):
                ct += 1
                plt.subplot(nrow, ncol, ct)
                plt.plot(data_dict[stim_mode][contacts[i]]['values'][0],
                         data_dict[stim_mode][contacts[j]]['values'][0], 'b.')
                plt.xscale('log')
                plt.yscale('log')
                plt.axis('equal')
                plt.plot(plt.xlim(), plt.ylim(), 'k-')
                plt.xlabel(f"Contact {contacts[i]}")
                plt.ylabel(f"Contact {contacts[j]}")
        plt.suptitle(f"Scatter Plots for {stim_mode} Stimulation")
        plt.show(block=(m == len(stim_modes) - 1))


# Run and plot the results of the experiment
def main() -> None:
    """
    This is the main function of the program.
    """
    args = parse_args()

    if args.data_directory is None:
        dir_name = f"experiment_all_contacts_{args.contact_length}_{args.contact_spacing}_" \
                f"{args.contact_ratio}_{args.n_segments_per_level}_{args.levels}"
        print(f'Writing data to {dir_name}')
        simulate_all_contacts(dir_name, args)
    else:
        print(f'Reading data from {args.data_directory}')
        data_dict = read_experiment_output(args.data_directory)

    create_scatter_plots(data_dict)


def parse_args() -> argparse.Namespace:
    """
    This function parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Simulate all contacts on a lead using OSS-DBS.")
    parser.add_argument(
        '-d',
        '--data_directory',
        type=str,
        default=None,
        help="If data directory given, just plot - don't rereun."
    )
    parser.add_argument(
        '-m',
        '--stim_mode',
        type=str,
        default="both",
        help="Stimulation mode (str) in [current, voltage, both]."
    )
    parser.add_argument(
        '-l',
        '--contact_length',
        type=float,
        default=0.5,
        help="Length (i.e., height) of each contact level."
    )
    parser.add_argument(
        '-s',
        '--contact_spacing',
        type=float,
        default=1.0,
        help="Spacing between contact levels."
    )
    parser.add_argument(
        '-p',
        '--n_segments_per_level',
        type=int,
        default=2,
        help="Number of segments per level."
    )
    parser.add_argument(
        '-v',
        '--levels',
        type=int,
        default=3,
        help="Number of levels."
    )
    parser.add_argument(
        '-r',
        '--contact_ratio',
        type=float,
        default=0.9,
        help="Ratio of total contact length to segment circumference."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
