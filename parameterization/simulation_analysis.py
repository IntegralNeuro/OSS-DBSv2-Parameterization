import numpy as np
import pyvista as pv
import argparse
import os
import json

"""
Some simple tools for analyzing the simulation results.
"""


def efield_magnitude(mesh: pv.PolyData) -> pv.PolyData:
    """
    Given a mesh of the E-field vector, add a new scalar which is its magnitude

    Args:
        mesh: contains array 'E_field_real'

    Returns:
        mesh: with new array 'E_field_mag'
    """
    # Get Hessian
    mesh['E_field_mag'] = np.linalg.norm(mesh['E_field_real'], axis=1)
    return mesh


def update_efield_file(dir: str) -> str:
    """
    Given a vtu with the E-field, add a new scalar with the magnitude of the E-field
    ('E_field_mag') and save the new file.

    Args:
        dir (str): Directory with an E-field (potiental gradient) vtu filename in it

    Returns:
        str: The path to the E-field file.
    """
    # Get E-field filenames
    e_field_files = ["E-field.vtu", "grid.vtk"]
    updated_files = []
    for file in e_field_files:
        filepath = os.path.join(dir, file)
        if not os.path.exists(filepath):
            continue
        # Load the mesh with E-field data and compute the magnitude
        mesh_e_field = pv.read(filepath)
        mesh_e_field = efield_magnitude(mesh_e_field)
        pv.save_meshio(filepath, mesh_e_field)
        updated_files.append(filepath)
    return updated_files


def load_vtu(vtu_file: str) -> tuple:
    """
    Load the vtu file and return the scalar values and the points.

    Args:
        vtu_file (str): The path to the vtu file.

    Returns:
        points: np.ndarray: The grid points, shape (n_points, 3)
        values: list of np.ndarray, each (n_points, dim)
        names: list of array names
    """
    mesh = pv.read(vtu_file)
    values = []
    for i, name in enumerate(mesh.array_names):
        values.append(mesh[mesh.array_names[i]])
    return mesh.points, values, mesh.array_names


def hessian_mesh_from_efield(mesh: pv.PolyData) -> pv.PolyData:
    """
    Given a mesh of the E-field, return a new mesh with the Hessian and its
    eigenvalues.

    Args:
        mesh: contains array 'E-field'

    Returns:
        hessian: mesh containing two arrays:
            'Hessian': (Npts x 9) - array whoes columns are the Hessian components:
                'dxx', 'dxy', 'dxz', 'dyx', 'dyy', 'dyz', 'dzx', 'dzy', 'dzz'
            'eigenvalues': (Npts x 3): array whose columns are the eigenvalues of the Hessian,
                                 in descending order
    """
    # Get Hessian
    mesh = mesh.compute_derivative(scalars='E_field_real', gradient='Hessian')
    # Convert to unstructured grid or leae as structured grid
    if type(mesh)==pv.PolyData:
        hessian = pv.UnstructuredGrid(mesh.cells, mesh.celltypes, mesh.points)
        hessian['Hessian'] = mesh['Hessian']
    elif type(mesh)==pv.StructuredGrid:
        hessian = mesh
    # Compute eigenvalues
    evals = np.linalg.eigvals(np.array(hessian['Hessian'].reshape(hessian.n_points, 3, 3)))
    evals = np.real(evals)  # take absolute value of eigenvalues
    evals.sort(axis=1)
    evals = evals[:, ::-1]  # store by abs values
    hessian['eigenvalues'] = evals
    return hessian


def make_hessian_file(dir: str) -> str:
    """
    Given a vtu of the gradient of a scalar field, compute the Hessian and
    save the a new file with the Hessian data:
        'gradient'

    Args:
        dir (str): Directory with an E-field (potiental gradient) vtu filename in it

    Returns:
        str: The path to the Hessian file.
    """
    # Get Hessian filename
    e_field_file = os.path.join(dir, "E-field.vtu")
    hessian_file = e_field_file.replace('E-field', 'Hessian')
    # Load the mesh with E-field data and compute the gradient
    mesh_e_field = pv.read(e_field_file)
    mesh_hessian = hessian_mesh_from_efield(mesh_e_field)
    pv.save_meshio(hessian_file, mesh_hessian)
    return hessian_file


def vta_from_hessian(hessian: pv.PolyData,
                     threshold: float,
                     axis: str = None,
                     eigenvalue: int = None,
                     vector: np.ndarray = None) -> pv.pyvista_ndarray:
    """
    Given a mesh with Hessian data, choose whether to look at a particular axis, one of the eigenvalues, or an arbitrary vector.
    Then, get the Hessian value, apply a threshold, and return a new mesh with the selected values.

    Args:
        hessian (pv.PolyData): The mesh with Hessian data.
        threshold (float, optional): The threshold value to apply. Defaults to None.
        axis (str, optional): The axis to select from the Hessian. Defaults to None.
        eigenvalue (int, optional): The eigenvalue index to select from the Hessian. Defaults to None.
        vector (np.ndarray, optional): The arbitrary 3-vector to select from the Hessian. Defaults to None.

    Returns:
        pv.PolyData: The new mesh with the selected values.
    """
    if axis is not None:
        axis_lookup = {'x': 0, 'y': 4, 'z': 8}  # diagonal elements of the Hessian
        vta_array = hessian['Hessian'].copy()
        vta_array = vta_array[:, axis_lookup[axis]]
    elif eigenvalue is not None:
        vta_array = hessian['eigenvalues'].copy()
        vta_array = vta_array[:, eigenvalue]
    elif vector is not None:
        vta_array = np.dot(np.dot(hessian['Hessian'].reshape(hessian.n_points, 3, 3), vector), vector)
    else:
        raise ValueError("Please specify either axis, eigenvalue, or vector.")

    if threshold is not None:
        in_vta = np.abs(vta_array) > threshold
        vta_array[~in_vta] = 0
        vta_array[in_vta] = 1

    return vta_array


def make_vta_file(hessian_vtu: str, vta_dict: dict) -> str:
    """
    Given a hessian vtu file, a dictionary of vta options, save a new vtu file containing
    vta array for each option.

    Args:
        hessian_vtu (str): The path to the hessian vtu file.
        vta_dict (dict): The dictionary of vta options:
            - keys: vta name
            - values: dict with keys:
                - axis: str: The axis to select from the Hessian. Defaults to None.
                - eigenvalue: int: The eigenvalue index to select from the Hessian. Defaults to None.
                - vector: np.ndarray: The arbitrary 3-vector to select from the Hessian. Defaults to None.
                - threshold: float: The threshold value to apply
            Each vta item must specify only one of axis, eigenvalue, or vector.

    Returns:
        str: The path to the new vtu file
    """
    # Get VTA filename
    vta_file = hessian_vtu.replace('Hessian', 'VTA')

    # Load the hessian vtu file
    hessian_mesh = pv.read(hessian_vtu)
    vta_mesh = pv.UnstructuredGrid(hessian_mesh.cells, hessian_mesh.celltypes,
                                   hessian_mesh.points)

    # Iterate over each vta option
    for vta_name, vta_option in vta_dict.items():
        if (not len(vta_option) == 2) or ('threshold' not in vta_option):
            raise ValueError("vta_dict entries must specify a threshold and " +
                             "only one of axis, eigenvalue, or vector.")
        if 'axis' in vta_option:
            new_vta = vta_from_hessian(hessian_mesh,
                                       vta_option['threshold'],
                                       axis=vta_option['axis'])
        elif 'eigenvalue' in vta_option:
            new_vta = vta_from_hessian(hessian_mesh,
                                       vta_option['threshold'],
                                       eigenvalue=vta_option['eigenvalue'])
        elif 'vector' in vta_option:
            new_vta = vta_from_hessian(hessian_mesh,
                                       vta_option['threshold'],
                                       vector=vta_option['vector'])
        else:
            raise ValueError("vta_dict must specify either axis, eigenvalue, or vector.")

        # Create a new mesh with the selected values
        vta_mesh[vta_name] = new_vta

    # Save the new vtu file
    pv.save_meshio(vta_file, vta_mesh)
    return vta_file


def recursive_dir_list(input_path: str, dir_list: list = None) -> None:
    if dir_list is None:
        dir_list = []
    contents = os.listdir(input_path)
    if "E-field.vtu" in contents:
        dir_list.append(input_path)
    else:
        contents.sort()
        subdir_list = [sub for sub in contents
                       if os.path.isdir(f"{input_path}/{sub}")]
        for sub in subdir_list:
            recursive_dir_list(f"{input_path}/{sub}", dir_list)
    return dir_list


def create_parser() -> argparse.ArgumentParser:
    """
    Create and return an instance of argparse. 
    """
    parser = argparse.ArgumentParser(description="Make Hessian and VTA files from. If none (--e_field, --hessian, or --vta are set, all will be done.")
    parser.add_argument("input_path", type=str, help="Path to the vtu file directory or a parent directory")
    parser.add_argument("-E", "--e_field", action="store_true", help="Update the E-field file")
    parser.add_argument("-H", "--hessian", action="store_true", help="Create the Hessian file")
    parser.add_argument("-v", "--vta", action="store_true", help="Create a VTA")
    parser.add_argument("-t", "--vta_threshold", type=float, default=None, help="Threshold value for VTA, if default behavior of computing thresholds for {x, y, z, eval1}")
    parser.add_argument("-j", "--vta_json", type=str, default=None, help="Path to a json file containing VTA option dictionaries")
    return parser


def main() -> None:
    """
    This is the main function of the program.
    It creates an instance of MeshPlotter and performs the necessary operations.
    """
    parser = create_parser()
    args = parser.parse_args()

    # Get the list of directories with E-field files
    dir_list = recursive_dir_list(args.input_path)

    for dir in dir_list:
        print(f"Processing directory: {dir}")

        if args.vta_json is not None:
            with open(args.vta_json, 'r') as f:
                vta_dict = json.load(f)
        else:
            vta_dict = {f'VTA(x,{args.vta_threshold})': {'axis': 'x', 'threshold': args.vta_threshold},
                        f'VTA(y,{args.vta_threshold})': {'axis': 'y', 'threshold': args.vta_threshold},
                        f'VTA(z,{args.vta_threshold})': {'axis': 'z', 'threshold': args.vta_threshold},
                        f'VTA(eval1,{args.vta_threshold})': {'eigenvalue': 0, 'threshold': args.vta_threshold}}

        if not args.e_field and not args.hessian and not args.vta:
            args.e_field = True
            args.hessian = True
            args.vta = True

        if args.e_field:
            e_field_files = update_efield_file(dir)
            print(f"Updated E-field files {e_field_files}")

        if args.hessian:
            hessian_file = make_hessian_file(dir)
            print(f"Saved Hessian file to {hessian_file}")

        if args.vta:
            if args.vta_threshold is None:
                raise ValueError("A threshold value must be specified to compute VTA.")
            hessian_file = os.path.join(dir, "Hessian.vtu")
            vta_file = make_vta_file(hessian_file, vta_dict)
            print(f"Saved VTA file to {vta_file}")

    print("Done!")


if __name__ == "__main__":
    main()
