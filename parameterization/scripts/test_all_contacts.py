import argparse
from json_modification_api import CustomElectrodeModeler
import simulation_analysis as sa
from matplotlib import pyplot as plt
import os


def main() -> None:
    """
    This is the main function of the program.
    """
    args = parse_args()

    # if running all, don't block on the first ones
    block = not args.experiment_name == "all"

    test_all_contacts(not args.no_rerun, block=block)


def parse_args() -> argparse.Namespace:
    """
    This function parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model electrodes for OSS-DBS.")
    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        default="all",
        help="Name of the experiment.",
    )
    parser.add_argument(
        '--no_rerun', '-n', action='store_true',
        help="Don't Rerun the simulation, just plot the results."
    )
    return parser.parse_args()


def test_all_contacts(rerun=True, block=True) -> None:
    """
    Look at single-contact stimulation for all contacts
    """
    # Parameters (put into args)
    contact_length = 0.5
    contact_spacing = 1.0
    n_segments_per_level = 2
    levels = 3
    contact_ratio = 0.9
    segment_contact_angle = contact_ratio * 360.0 / n_segments_per_level
    stim_mode_list = ["voltage", "current"]

    dir_name = f"experiment_all_contacts_{contact_length}_{contact_spacing}_{contact_ratio}_{n_segments_per_level}_{levels}"
    print(f'Writing data to {dir_name}')
    
    if rerun:
        electrode = CustomElectrodeModeler("ossdbs_input_config")
        electrode.modify_electrode_custom_parameters(
            total_length=300.0,
            contact_length=contact_length,
            contact_spacing=contact_spacing,
            segment_contact_angle=segment_contact_angle,
            n_segments_per_level=n_segments_per_level,
            levels=levels,
            segmented_levels=[1, 2, 3, 4, 5],
            tip_contact=False,
        )
        # Test whether linearity holds for monopolar stimulations
        n_contacts = electrode.get_electrode_custom_parameters()["_n_contacts"]
        for stim_mode in stim_mode_list:
            for contact in range(n_contacts):
                name = f"{stim_mode}_{contact}"
                electrode.generate_output_path(f"{dir_name}/{name}")
                for i in range(n_contacts):
                    if (i + 1) == contact:
                        if stim_mode == "current":
                            electrode.generate_current_contact(i + 1, 1.0)
                        else:
                            electrode.generate_voltage_contact(i + 1, 1.0)
                    else:
                        electrode.generate_floating_contact(i + 1)
                electrode.update_parameters()
                electrode.modify_json_parameters()
                electrode.run_ossdbs()
    else:
        data_dict = {}
        for subdir in os.listdir('results/'+dir_name):
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


if __name__ == "__main__":
    main()
