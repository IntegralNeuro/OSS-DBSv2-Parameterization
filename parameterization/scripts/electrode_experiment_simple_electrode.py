import argparse

from json_modification_api import CustomElectrodeModeler


def main() -> None:
    """
    This is the main function of the program.
    It creates an instance of CustomElectrodeModeler and performs the necessary operations.
    """
    args = parse_args()
    model_electrodes(args.experiment_name)


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
        default="simple_electrode",
        help="Name of the experiment.",
    )
    return parser.parse_args()


def model_electrodes(experiment_name: str) -> None:
    """
    This function models the electrode with the specified number of contacts.

    Args:
        modeler (CustomElectrodeModeler): The CustomElectrodeModeler instance.

    Returns:
        None
    """
    electrode = CustomElectrodeModeler("ossdbs_input_config")
    electrode.modify_electrode_custom_parameters(
        total_length=300.0,
        segment_contact_angle=100.0,
        n_segments_per_level=3,
        levels=4,
        segmented_levels=[2, 3, 4],
        tip_contact=True,
    )
    for i in range(electrode.get_electrode_custom_parameters()["_n_contacts"]):
        if i % 2 == 0:
            electrode.generate_floating_contact(i + 1)
        else:
            electrode.generate_current_contact(i + 1, 1.0)
    electrode.generate_output_path(f"experiment_{experiment_name}/electrode_1")
    electrode.update_parameters()
    electrode.modify_json_parameters()
    electrode.run_ossdbs()
    electrode.plot_electrode()


if __name__ == "__main__":
    main()
