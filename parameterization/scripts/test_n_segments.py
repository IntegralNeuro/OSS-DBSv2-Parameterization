import argparse
import numpy as np
from json_modification_api import CustomElectrodeModeler


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
        for n_contacts in range(1, args.max_segments + 1, args.skip_segments):
            # Set up electrode
            electrode = CustomElectrodeModeler("ossdbs_input_config")
            if n_contacts > 1:
                radius = electrode.input_dict["Electrodes"][0]["CustomParameters"]["lead_diameter"] / 2
                contact_angle = 360.0 / n_contacts - (args.contact_gap / radius) * 180.0 / np.pi
                electrode.modify_electrode_custom_parameters(
                    total_length=300.0,
                    contact_length=args.contact_length,
                    contact_spacing=args.contact_spacing,
                    segment_contact_angle=contact_angle,
                    n_segments_per_level=n_contacts,
                    levels=1,
                    segmented_levels=[1],
                    tip_contact=False,
                )
            else:
                electrode.modify_electrode_custom_parameters(
                    total_length=300.0,
                    contact_length=args.contact_length,
                    contact_spacing=args.contact_spacing,
                    segmented_levels=[],
                    levels=1,
                    tip_contact=False,
                )

            # Simulate each contact
            name = f"{stim_mode}_{n_contacts}"
            electrode.generate_output_path(f"{dir_name}/{name}", args.results_directory)
            electrode.generate_current_contact(1, 1.0,
                                               max_mesh_size=args.max_mesh_size, 
                                               max_mesh_size_edge=args.max_mesh_size_edge)
            for i in range(1, n_contacts):
                if args.floating_contacts:
                    electrode.generate_floating_contact(i + 1,
                                                        max_mesh_size=args.max_mesh_size, 
                                                        max_mesh_size_edge=args.max_mesh_size_edge)
                else:
                    electrode.generate_unused_contact(i + 1,
                                                      max_mesh_size=args.max_mesh_size, 
                                                      max_mesh_size_edge=args.max_mesh_size_edge)
            
            # electrode.input_dict["Mesh"]["MeshingHypothesis"]["Type"] = "VeryFine"
            
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
    parser = argparse.ArgumentParser(description="Simulate all contacts on a lead using OSS-DBS.")
    parser.add_argument('-m', '--stim_mode', type=str, default="both", help="Stimulation mode (str) in [current, voltage, both].")
    parser.add_argument('-l', '--contact_length', type=float, default=1.5, help="Length (i.e., height) of each contact level.")
    parser.add_argument('-s', '--contact_spacing', type=float, default=0.5, help="Height between each contact level.")
    parser.add_argument('-p', '--max_segments', type=int, default=4, help="Max number of segments per level.")
    parser.add_argument('-k', '--skip_segments', type=int, default=1, help="Simulate n_contacts in [:max_segments:skip_segments].")
    parser.add_argument('-g', '--contact_gap', type=float, default=0.1, help="Gap between segmented contacts in mm.")
    parser.add_argument('-f', '--floating_contacts', action='store_true', help="Make non-stim contacts floating, as opposed to unused.")
    parser.add_argument('-e', '--encapsulation_d', type=float, default=0.0, help="Thickness of encapsulation layer.")
    parser.add_argument('-M', '--max_mesh_size', type=float, default=1.0, help="Max mesh size.")
    parser.add_argument('-E', '--max_mesh_size_edge', type=float, default=0.35, help="Max edge mesh size.")
    parser.add_argument('-d', '--results_directory', type=str, default="./results", help="Results directory.")
    return parser.parse_args()


# Run and plot the results of the experiment
def main() -> None:
    """
    This is the main function of the program.
    """
    args = parse_args()

    # Run the experiment, if necessary
    if args.floating_contacts:
        float_str = 'floating'
    else:
        float_str = 'unused'
    dir_name = f"experiment_n_contacts_{args.contact_length}_{args.contact_spacing}_" \
            f"{args.contact_gap}_{args.encapsulation_d}_{float_str}"
    print(f'Writing data to {dir_name}')
    simulate_all_contacts(dir_name, args)


if __name__ == "__main__":
    main()
