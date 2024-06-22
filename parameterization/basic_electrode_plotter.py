import argparse

import matplotlib.pyplot as plt
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
    generate_combined_files(input_folder, i, electrode, points) # Comment this out if you've already run it once for the folder you want
    gradient = pd.read_csv(f"{input_folder}/combined_E_field_gradient.csv")
    # Plotting the gradient
    dxx = gradient["dxx"]
    dxy = gradient["dxy"]
    dxz = gradient["dxz"]
    dyx = gradient["dyx"]
    dyy = gradient["dyy"]
    dyz = gradient["dyz"]
    dzx = gradient["dzx"]
    dzy = gradient["dzy"]
    dzz = gradient["dzz"]
    hessian = np.zeros((len(dxx), 3, 3))
    frobenius_norm = [0] * len(dxx)
    for i in range(len(dxx)):
        hessian[i] = np.array([[dxx[i], dxy[i], dxz[i]],
                               [dyx[i], dyy[i], dyz[i]],
                               [dzx[i], dzy[i], dzz[i]]])
        frobenius_norm[i] = np.linalg.norm(hessian[i], ord='fro')
    plt.figure()
    plt.plot(frobenius_norm)
    plt.title("Frobenius Norm of the Hessian Matrix of the E-field Gradient")
    plt.xlabel("Point Index")
    plt.ylabel("Frobenius Norm (V/mm^2)")
    plt.show()


def generate_combined_files(input_folder, i, electrode, points) -> pd.DataFrame:
    """
    Generate combined CSV files containing the coordinates and scalar values of the electrode.

    Args:
        input_folder (str): The path to the folder where the combined CSV files will be saved.
        i (int): The index of the electrode.
        electrode (object): The electrode object containing scalar array actors.
        points (list): The list of points corresponding to the electrode.

    Returns:
        pandas.DataFrame: The gradient dataframe if the key is "E-field_gradient", otherwise None.

    Raises:
        None

    """
    for key in electrode.scalar_array_actor.keys():
        combined_df = pd.concat([pd.DataFrame(points[i]), pd.DataFrame(electrode.scalar_array_actor[key])], axis=1)
        if key == "E-field.vtu":
            combined_df.columns = ["x-pt", "y-pt", "z-pt", "x-field", "y-field", "z-field"]
            combined_df["magnitude"] = np.sqrt(combined_df["x-field"]**2 + combined_df["y-field"]**2 + combined_df["z-field"]**2)
            filename = f"{input_folder}/combined_{electrode.scalars[i]}.csv"
        elif key == "E-field_gradient":
            combined_df.columns = ["x-pt", "y-pt", "z-pt", "dxx", "dxy", "dxz", "dyx", "dyy", "dyz", "dzx", "dzy", "dzz"]
            gradient = combined_df
            filename = f"{input_folder}/combined_E_field_gradient.csv"
            i += 1
        else:
            combined_df.columns = ["x-pt", "y-pt", "z-pt", "scalar"]
            filename = f"{input_folder}/combined_{electrode.scalars[i]}.csv"
            i += 1
        combined_df.to_csv(filename)
        print(f"Saved combined data to {filename}")
    return gradient
    
    



if __name__ == "__main__":
    main()
