import numpy as np  # type: ignore
import argparse
import pyvista as pv  # type: ignore
import os


class MeshPlotter:
    def __init__(self,
                 grid_plot: bool = False,
                 plot_E_field: bool = False,
                 plot_material: bool = False,
                 plot_conductivity: bool = False,
                 plot_potential: bool = False,
                 plot_Hessian: bool = False,
                 plot_VTA: bool = False,
                 e_field_scale: float = 1,
                 jitter_std: float = 0.0,
                 opacity: float = 1.0,
                 start_invisible: bool = False) -> None:

        self.colormaps = {
            "electrode": "cividis",  # Colormap for 'boundaries'
            "material": "tab10",  # Colormap for 'material_real'
            "conductivity": "plasma",  # Colormap for 'conductivity_real'
            "E-field": "Blues_r",  # Colormap for 'E_field_real'
            "potential": "RdBu",  # Colormap for 'potential_real'
            "Hessian": "PiYG",  # Colormap for 'Hessian'
            "VTA": "Greens"  # Colormap for 'VTA'
        }
        self.plot_electrode = True
        self.plot_E_field = plot_E_field
        self.plot_material = plot_material
        self.plot_conductivity = plot_conductivity
        self.plot_potential = plot_potential
        self.plot_Hessian = plot_Hessian
        self.plot_VTA = plot_VTA

        self.e_field_scale = e_field_scale
        self.grid_plot = grid_plot
        self.jitter_std = jitter_std
        self.opacity = opacity
        self.start_visible = not start_invisible

        self.mesh_actors = []
        self.input_dirs = []

    def _plotter_grid_init(self, n_plots: int) -> None:
        n_plots = len(self.input_dirs)
        self.ncol = np.ceil(np.sqrt(n_plots)).astype(int)
        self.nrow = np.ceil(n_plots / self.ncol).astype(int)
        self.plotter = pv.Plotter(shape=(self.nrow, self.ncol))
        self.plotter.set_background("slategray")

    def _plotter_single_init(self) -> None:
        self.ncol = 1
        self.nrow = 1
        self.subplot_toggle_count = 0
        self.subplot_scale_bar_count = 0
        self.plotter = pv.Plotter()
        self.plotter.set_background("slategray")

    def plotter_init(self, n_plots: int) -> None:
        self.results_count = 0
        self.toggle_count = 0
        if self.grid_plot:
            self._plotter_grid_init(n_plots)
        else:
            self._plotter_single_init()

    def _get_colormap(self, array_name: str) -> str:
        for key in self.colormaps.keys():
            if key in array_name:
                return self.colormaps[key]
        else:
            return "viridis"

    def _add_scalar(self,
                    mesh: pv.PolyData,
                    array_name: str,
                    label: str) -> None:
        if ("electrode" in array_name) or ("boundaries" in array_name):
            is_electrode = True
            kwargs = {
                "style": "surface",
                "show_edges": True,
                "cmap": self._get_colormap(array_name),
                "show_scalar_bar": False,
            }
        else:
            is_electrode = False
            kwargs = {
                "style": "points",
                "show_edges": True,
                "cmap": self._get_colormap(array_name),
                "scalar_bar_args": {
                    "title": label,
                    "title_font_size": 20,
                    "label_font_size": 15,
                    "position_x": 0.7,
                    "position_y": 0.02 + 0.06 * self.subplot_scale_bar_count,
                    "width": 0.25,
                    "height": 0.03
                }
            }
            mesh.points += np.random.normal(0, self.jitter_std, mesh.points.shape)
            self.subplot_scale_bar_count += 1

        mesh_actor = self.plotter.add_mesh(
            mesh,
            name=label,
            scalars=mesh[array_name],
            opacity=self.opacity,
            **kwargs
        )
        if not is_electrode:
            mesh_actor.SetVisibility(self.start_visible)
        return mesh_actor

    def _add_vector(self,
                    mesh: pv.PolyData,
                    array_name: str,
                    label: str) -> None:
        arrows = mesh.glyph(factor=self.e_field_scale, scale=array_name, orient=array_name)
        kwargs = {
            "show_edges": False,
            "color": "black",
            "show_scalar_bar": False,
        }
        mesh_actor = self.plotter.add_mesh(
            arrows,
            name=label,
            opacity=self.opacity,
            **kwargs
        )
        mesh_actor.SetVisibility(self.start_visible)
        return mesh_actor

    def add_mesh(self,
                 mesh: pv.PolyData,
                 array_name: str,
                 label: str = None) -> None:
        # Type specific kwargs
        label = label if label else array_name
        if mesh[array_name].ndim == 1:
            mesh_actor = self._add_scalar(mesh, array_name, label)
        elif mesh[array_name].shape[1] == 3:  # Vector mesh
            mesh_actor = self._add_vector(mesh, array_name, label)
        self.mesh_actors.append(mesh_actor)

    def add_toggle(self, label: str, initial_state: bool = True) -> None:
        self.plotter.add_checkbox_button_widget(
            lambda flag, index=self.toggle_count: self._toggle_visibility(flag, index),
            position=(12, 12 + self.subplot_toggle_count * 30),  # Adjust position for each checkbox
            color_on=(0.0, 1.0, 0.0),  # Optional: color when checked (green)
            color_off=(1.0, 0.0, 0.0),  # Optional: color when unchecked (red)
            size=20,  # Optional: size of the checkbox
            border_size=1,  # Optional: size of the border
            value=initial_state,  # Optional: initial state of the checkbox
        )
        self.plotter.add_text(label,
                              position=(40, 10 + self.subplot_toggle_count * 30),
                              font_size=12)
        self.toggle_count += 1
        self.subplot_toggle_count += 1

    def _toggle_visibility(self, flag, index):
        self.mesh_actors[index].SetVisibility(flag)
        self.plotter.render()

    def _get_vtu_filenames(self, vtu_directory: str) -> list:
        """
        Returns a list of vtu filenames that are to be plotted based on the "plot_x" flags.

        Parameters:
        - vtu_directory (str): The directory containing the vtu files.

        Returns:
        - list: A list of vtu filenames to be plotted.
        """
        # get the vtu files in the directory
        all_vtu_files = [filename for filename in os.listdir(vtu_directory)
                         if filename.endswith(".vtu")]
        # cycle through the vtu files
        vtu_files = []
        for filename in all_vtu_files:
            # only plot the files that are listed in the colormaps
            keys = [key for key in self.colormaps.keys() if key in filename]
            if len(keys) == 0:
                continue
            if keys[0] == "E-field":
                if self.plot_E_field:
                    vtu_files.append(filename)
            elif getattr(self, f"plot_{keys[0]}"):
                vtu_files.append(filename)
        return vtu_files

    def add_results_directory(self, vtu_directory: str) -> None:
        if self.grid_plot:
            # Initialize Subplot
            irow = self.results_count // self.ncol
            icol = self.results_count % self.ncol
            self.plotter.subplot(irow, icol)
            self.subplot_toggle_count = 0
            self.subplot_scale_bar_count = 0
        self.results_count += 1
        # Get the vtu files to plot in this vtu directory
        vtu_filenames = self._get_vtu_filenames(vtu_directory)
        # Add the arrays from each file to the plotter
        for filename in vtu_filenames:
            mesh = pv.read(f"{vtu_directory}/{filename}")
            for array_name in mesh.array_names:
                label = f'{filename.split(".")[0]}'
                if len(self.input_dirs) > 1:
                    if self.grid_plot:
                        label = f'{label}_{self.results_count}'
                    else:
                        label = f'{vtu_directory.split("/")[-1]}: {label}'
                self.add_mesh(mesh, array_name, label)
                # Add toggle for scalar mesh visibility, but electrodes always start visible
                self.add_toggle(label,
                                ("boundaries" in array_name) or self.start_visible)
        # Title
        if (not self.grid_plot) and (len(self.input_dirs) > 1):
            title = self.input_path.split('/')[-1]
        else:
            title = vtu_directory.split('/')[-1]
        self.plotter.title = title
        self.plotter.add_text(title, position="upper_left", color="black", font_size=16)

        self.plotter.show_grid()

    def _recursive_dir_list(self, input_path: str) -> None:
        contents = os.listdir(input_path)
        contents.sort()
        if "potential.vtu" in contents:
            self.input_dirs.append(input_path)
        else:
            dir_list = [sub for sub in contents
                        if os.path.isdir(f"{input_path}/{sub}")]
            for sub in dir_list:
                self.input_dirs.append(f"{input_path}/{sub}")

    def make_figure(self, input_path: str) -> None:
        self.input_path = input_path
        self._recursive_dir_list(input_path)
        self.input_dirs.sort()
        self.plotter_init(len(self.input_dirs))
        for dir in self.input_dirs:
            self.add_results_directory(dir)

        self.plotter.show()


def create_parser() -> argparse.ArgumentParser:
    """
    Create and return an instance of argparse.ArgumentParser with the necessary arguments.
    """
    parser = argparse.ArgumentParser(description="Plot OSS-DBS Simulation and Analysis Results. If none of the `plot_x` flags are set, all plots will be made.")
    parser.add_argument("input_path", type=str, help="Path to the vtu file directory or a parent directory")
    parser.add_argument("-g", "--grid_plot", action="store_true", help="Make a grid of plots, one for each result directory")
    parser.add_argument("-m", "--plot_material", action="store_true", help="Plot material")
    parser.add_argument("-c", "--plot_conductivity", action="store_true", help="Plot conductivity")
    parser.add_argument("-e", "--plot_E_field", action="store_true", help="Plot E-field")
    parser.add_argument("-p", "--plot_potential", action="store_true", help="Plot potential")
    parser.add_argument("-H", "--plot_Hessian", action="store_true", help="Plot Hessian of potential")
    parser.add_argument("-v", "--plot_VTA", action="store_true", help="Plot VTAs")
    parser.add_argument("-s", "--e_field_scale", type=float, default=1.0, help="Scale factor for E-field vectors")
    parser.add_argument("-j", "--jitter_std", type=float, default=0.0, help="Standard deviation for jittering the mesh points")
    parser.add_argument("-o", "--opacity", type=float, default=1.0, help="Plot opacity, in [0, 1]")
    parser.add_argument("-i", "--start_invisible", action="store_true", help="Start with scalar meshes toggeled to invisible")
    return parser


def main() -> None:
    """
    This is the main function of the program.
    It creates an instance of MeshPlotter and performs the necessary operations.
    """
    parser = create_parser()
    args = parser.parse_args()

    # If not plotting any specific data, plot all
    if np.sum([args.plot_material, args.plot_conductivity, args.plot_E_field,
               args.plot_potential, args.plot_Hessian, args.plot_VTA]) == 0:
        args.plot_material = True
        args.plot_conductivity = True
        args.plot_E_field = True
        args.plot_potential = True
        args.plot_Hessian = True
        args.plot_VTA = True

    plotter = MeshPlotter(
        grid_plot=args.grid_plot,
        plot_material=args.plot_material,
        plot_conductivity=args.plot_conductivity,
        plot_E_field=args.plot_E_field,
        plot_potential=args.plot_potential,
        plot_Hessian=args.plot_Hessian,
        plot_VTA=args.plot_VTA,
        e_field_scale=args.e_field_scale,
        jitter_std=args.jitter_std,
        opacity=args.opacity,
        start_invisible=args.start_invisible
    )
    plotter.make_figure(input_path=args.input_path)


if __name__ == "__main__":
    main()
