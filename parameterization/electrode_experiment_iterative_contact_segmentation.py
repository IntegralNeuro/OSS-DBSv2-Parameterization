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
        default="iterative_contact_segmentation",
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
            segment_contact_angle=360/(i+2)-(20),
            n_segments_per_level=i+2,
            levels=4,
            segmented_levels=[2, 4],
            tip_contact=True,
        )
        for j in range(electrode.get_electrode_custom_parameters()["_n_contacts"]):
            if j % 2 == 0:
                electrode.generate_floating_contact(j + 1)
            else:
                electrode.generate_current_contact(j + 1, 1.0)
        electrode.generate_output_path(f"/experiment_{experiment_name}/electrode_{i+1}")
        electrode.update_parameters()
        electrode.modify_json_parameters()
        if i == 9:
            electrode.run_ossdbs()
            electrode.plot_electrode()  # Right now this only plots the last electrode generated

if __name__ == "__main__":
    main()