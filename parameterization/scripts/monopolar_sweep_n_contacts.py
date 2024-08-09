import argparse
import numpy as np
from json_modification_api import CustomElectrodeModeler
"""
Simulate monopolar stimulation for a range contact segment numbers on an
annulus (one contact simulated per geometry).
"""

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

    # Simulate first contact on each electrode
    for stim_mode in stim_mode_list:
        # for n_contacts in range(1, args.max_segments + 1, args.skip_segments):
        for n_contacts in args.segment_list:
            # Set up electrode
            electrode = CustomElectrodeModeler("ossdbs_input_config")
            tot_contacts = n_contacts * args.levels[0]
            stim_contact = n_contacts * (args.levels[1] - 1) + 1
            if n_contacts > 1:
                radius = electrode.input_dict["Electrodes"][0]["CustomParameters"]["lead_diameter"] / 2
                contact_angle = 360.0 / n_contacts - (args.contact_gap / radius) * 180.0 / np.pi
                electrode.modify_electrode_custom_parameters(
                    total_length=300.0,
                    contact_length=args.contact_length,
                    contact_spacing=args.contact_spacing,
                    segment_contact_angle=contact_angle,
                    n_segments_per_level=n_contacts,
                    levels=args.levels[0],
                    segmented_levels=list(range(1, args.levels[0] + 1)),
                    tip_contact=False,
                )
            else:
                electrode.modify_electrode_custom_parameters(
                    total_length=300.0,
                    contact_length=args.contact_length,
                    contact_spacing=args.contact_spacing,
                    levels=args.levels[0],
                    segmented_levels=[],
                    tip_contact=False,
                )

            # Simulate each contact
            name = f"{stim_mode}_{n_contacts:02d}"
            electrode.generate_output_path(f"{dir_name}/{name}", args.results_directory)
            for c in range(1, tot_contacts + 1):
                if c == stim_contact:
                    if stim_mode == 'current':
                        electrode.generate_current_contact(c, 1.0,
                                                           max_mesh_size=args.max_mesh_size,
                                                           max_mesh_size_edge=args.max_mesh_size_edge)
                    else:
                        electrode.generate_voltage_contact(c, 1.0,
                                                           max_mesh_size=args.max_mesh_size,
                                                           max_mesh_size_edge=args.max_mesh_size_edge)
                elif args.unused_contacts:
                    electrode.generate_unused_contact(c,
                                                      max_mesh_size=args.max_mesh_size,
                                                      max_mesh_size_edge=args.max_mesh_size_edge)
                else:
                    electrode.generate_floating_contact(c,
                                                        max_mesh_size=args.max_mesh_size,
                                                        max_mesh_size_edge=args.max_mesh_size_edge)

            if args.very_fine:
                electrode.input_dict["Mesh"]["MeshingHypothesis"]["Type"] = "VeryFine"

            if args.encapsulation_d > 0.0:
                electrode.input_dict["Electrodes"][0]["EncapsulationLayer"]["Thickness[mm]"] = args.encapsulation_d
            electrode.update_parameters()
            electrode.modify_json_parameters()
            electrode.run_ossdbs()


def parse_args() -> argparse.Namespace:
    """
    This function parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Simulate monopolar stimulation for a range contact segment numbers on an annulus (one contact simulated per geometry).")
    parser.add_argument('-m', '--stim_mode', type=str, default="current", help="Stimulation mode (str) in [*current*, voltage, both].")
    parser.add_argument('-l', '--contact_length', type=float, default=1.5, help="Length (i.e., height) of each contact level [1.5 mm].")
    parser.add_argument('-c', '--contact_spacing', type=float, default=0.5, help="Height between each contact level [0.5 mm].")
    parser.add_argument('-v', '--levels', type=int, nargs=2, default=[1, 1], help="Total number of levels and stimulation level (1-indexed)[1, 1].")
    # parser.add_argument('-p', '--max_segments', type=int, default=4, help="Max number of segments per level.")
    # parser.add_argument('-k', '--skip_segments', type=int, default=1, help="Simulate n_contacts in [:max_segments:skip_segments].")
    parser.add_argument('-s', '--segment_list', type=int, nargs='+', default=None, help="List of segments-per-level to run.")
    parser.add_argument('-g', '--contact_gap', type=float, default=0.1, help="Gap between segmented contacts in mm. [0.1 mm]")
    parser.add_argument('-u', '--unused_contacts', action='store_true', help="Make non-stim contacts unused, as opposed to floating.")
    parser.add_argument('-e', '--encapsulation_d', type=float, default=0.0, help="Thickness of encapsulation layer. [0.0 mm]")
    parser.add_argument('-M', '--max_mesh_size', type=float, default=1.0, help="Max mesh size. [1.0 mm]")
    parser.add_argument('-E', '--max_mesh_size_edge', type=float, default=0.35, help="Max edge mesh size. [0.35 mm]")
    parser.add_argument('-V', '--very_fine', action='store_true', help="Make mesh model 'VeryFine'.")
    parser.add_argument('-d', '--results_directory', type=str, default="./results", help="Results directory. [./results]")
    parser.add_argument('-D', '--data_directory', type=str, default=None, help="Data directory, within results. [manufactured]")
    return parser.parse_args()


# Run and plot the results of the experiment
def main() -> None:
    """
    This is the main function of the program.
    """
    args = parse_args()

    # Run the experiment, if necessary
    if args.data_directory is None:
        if args.unused_contacts:
            float_str = 'unused'
        else:
            float_str = 'floating'
        dir_name = f"experiment_n_contacts_{args.contact_length}_{args.contact_spacing}_" \
                   f"{args.contact_gap}_{args.encapsulation_d}_{float_str}"
        if args.very_fine:
            dir_name += "_vfine"
    else:
        dir_name = args.data_directory
    print(f'Writing data to {dir_name}')
    print(args)
    simulate_all_contacts(dir_name, args)


if __name__ == "__main__":
    main()
