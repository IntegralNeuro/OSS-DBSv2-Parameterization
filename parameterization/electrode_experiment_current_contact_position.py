from json_modification_api import CustomElectrodeModeler
import argparse

def main() -> None:
    """
    This is the main function of the program.
    It creates an instance of CustomElectrodeModeler and performs the necessary operations.
    """
    args = parse_args()
    model_electrodes(args.electrodes_to_generate, args.experiment_name)

def parse_args() -> argparse.Namespace:
    """
    This function parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model electrodes for OSS-DBS.")
    parser.add_argument(
        "--electrodes_to_generate", "-n",
        type=int,
        default=5,
        help="Number of electrodes to generate.",
    )
    parser.add_argument(
        "--experiment_name", "-e",
        type=str,
        default="current_contact_position",
        help="Name of the experiment.",
    )
    return parser.parse_args()

def model_electrodes(electrodes_to_generate: int, experiment_name: str) -> None:
    """
    This function models the electrode with the specified number of contacts.

    Args:
        modeler (CustomElectrodeModeler): The CustomElectrodeModeler instance.

    Returns:
        None
    """
    electrode = CustomElectrodeModeler("ossdbs_input_config")
    for i in range(electrodes_to_generate):
        electrode.modify_electrode_custom_parameters(
            total_length=300.0,
            segment_contact_angle=100.0,
            n_segments_per_level=3,
            levels=4,
            segmented_levels=[2, 3, 4],
            tip_contact=True,
        )
        for j in range(electrode.get_electrode_custom_parameters()["_n_contacts"]):
            if i == 0 and j == i:
                electrode.generate_floating_contact(j + 1)
                electrode.generate_current_contact(j + 2, 1.0)
            elif j == i:
                electrode.generate_current_contact(j + 1, 1.0)
            elif i == 0 and j != i+1:
                electrode.generate_floating_contact(j + 1)
            else:
                electrode.generate_floating_contact(j + 1)
        electrode.generate_output_path(f"/experiment_{experiment_name}/electrode_{i+1}")
        electrode.update_parameters()
        electrode.modify_json_parameters()
        electrode.run_ossdbs()
    electrode.plot_electrode()  # Right now this only plots the last electrode generated

if __name__ == "__main__":
    main()