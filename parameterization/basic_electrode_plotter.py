import argparse

import numpy as np # type: ignore
import pandas as pd # type: ignore
from json_modification_api import CustomElectrodeModeler

DEFAULT_INPUT_FOLDER = "./plotter_example"

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
    i = 0
    electrode = CustomElectrodeModeler()
    if analyze_flag:
        points = electrode.analyze_electrode(input_folder)
    else:    
        electrode.plot_electrode(input_folder)
    print(electrode.scalar_array_actor.keys())
    for key in electrode.scalar_array_actor.keys():
        if key == "E-field.vtu":
            combined_df = pd.concat([pd.DataFrame(points[i]), pd.DataFrame(electrode.scalar_array_actor[key])], axis=1)
            combined_df.columns = ["x-pt", "y-pt", "z-pt", "x-field", "y-field", "z-field"]
            combined_df["magnitude"] = np.sqrt(combined_df["x-field"]**2 + combined_df["y-field"]**2 + combined_df["z-field"]**2)
            combined_df.to_csv(f"{input_folder}/combined_{electrode.scalars[i]}.csv")
            print(f"Saved combined data to {input_folder}/combined_{electrode.scalars[i]}.csv")
        elif key == "E-field_gradient":
            combined_df = pd.concat([pd.DataFrame(points[i]), pd.DataFrame(electrode.scalar_array_actor[key])], axis=1)
            combined_df.columns = ["x-pt", "y-pt", "z-pt", "dxx", "dxy", "dxz", "dyx", "dyy", "dyz", "dzx", "dzy", "dzz"]
            combined_df.to_csv(f"{input_folder}/combined_E_field_gradient.csv")
            print(f"Saved combined data to {input_folder}/combined_E_field_gradient.csv")
            i += 1
        else:
            combined_df = pd.concat([pd.DataFrame(points[i]), pd.DataFrame(electrode.scalar_array_actor[key])], axis=1)
            combined_df.columns = ["x-pt", "y-pt", "z-pt", "scalar"]
            combined_df.to_csv(f"{input_folder}/combined_{electrode.scalars[i]}.csv")
            print(f"Saved combined data to {input_folder}/combined_{electrode.scalars[i]}.csv")
            i += 1
    



if __name__ == "__main__":
    main()
