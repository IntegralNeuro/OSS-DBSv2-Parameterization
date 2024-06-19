import argparse

import pandas as pd # type: ignore
from json_modification_api import CustomElectrodeModeler

DEFAULT_INPUT_FOLDER = "./results_example"

def main() -> None:
    """
    This is the main function of the program.
    It creates an instance of CustomElectrodeModeler and performs the necessary operations.
    """
    args = parse_args()
    plot_electrode(args.input_folder, args.analyze_flag)


def parse_args() -> argparse.Namespace:
    """
    This function parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model electrodes for OSS-DBS.")
    parser.add_argument(
        "--input_folder",
        "-i",
        type=str,
        default=DEFAULT_INPUT_FOLDER,
        help="Folder containing the relevant vtu input files",
    )
    parser.add_argument(
        "--analyze_flag",
        "-a",
        action="store_true",
        default=False,
        help="Flag to choose to only analyze the electrode",
    )
    return parser.parse_args()


def plot_electrode(input_folder: str, analyze_flag: bool) -> None:
    """
    This function models the electrode with the specified number of contacts.

    Args:
        modeler (CustomElectrodeModeler): The CustomElectrodeModeler instance.

    Returns:
        None
    """
    electrode = CustomElectrodeModeler()
    if analyze_flag:
        electrode.analyze_electrode(input_folder)
    else:    
        electrode.plot_electrode(input_folder)    
    for key in electrode.scalar_array_actor.keys():
        df = pd.DataFrame(electrode.scalar_array_actor[key])
        df.to_csv(f"{input_folder}/scalars_{electrode.scalars[key]}.csv")
        print(f"Saved scalars to {input_folder}/scalars_{electrode.scalars[key]}.csv")


if __name__ == "__main__":
    main()
