import argparse
import copy
import json
import os
import subprocess
from datetime import datetime

import pyvista as pv  # type: ignore
from hydra import compose, initialize  # type: ignore
from omegaconf import OmegaConf  # type: ignore

# Default contacts are neither floating or active, and hence "unused"
_CONTACT_DICT_TEMPLATE = {
    "Contact_ID": 0,
    "Active": False,  # whether voltage is clamped in FEM
    "Current[A]": False,  # whether the electrode is a current_source
    "Voltage[V]": False,  # voltage - only matters if active==True (should be)
    "Floating": False,  # required for used electrode if active==False
    "SurfaceImpedance[Ohmm]": {"real": 0.0, "imag": 0.0},
    "MaxMeshSize": 1.0,
    "MaxMeshSizeEdge": 0.035,
}


class CustomElectrodeModeler:
    def __init__(self, config_input: str = "ossdbs_input_config") -> None:
        self.custom_contacts = []
        self.output_path = ""
        self.file_path = ""
        self.filenames = [
            "electrode_1.vtu",
            "conductivity.vtu",
            "E-field.vtu",
            "material.vtu",
            "potential.vtu",
        ]
        self.scalars = [
            "boundaries",
            "conductivity_real",
            "E_field_real",
            "material_real",
            "potential_real",
        ]
        self.colormaps = [
            "cividis",  # Colormap for 'boundaries'
            "plasma",  # Colormap for 'conductivity_real'
            "Blues_r",  # Colormap for 'E_field_real'
            "magma",  # Colormap for 'material_real'
            "YlOrRd",  # Colormap for 'potential_real'
        ]
        self.meshes = []
        self.scalar_array_actor = {}
        self.plotter = pv.Plotter()
        self.plotter.set_background("slategray")
        self.config_name = config_input
        self.input_dict = self.generate_input_dictionary_template()
        self.initial_contacts = self.input_dict["Electrodes"][0]["Contacts"]

    def update_parameters(self) -> dict:
        """
        Updates parameters from the input dictionary based on the number of contacts specified.

        Args:
            contact_dict (dict): The input dictionary.

        Returns:
            None
        """
        self.input_dict["Electrodes"][0]["Contacts"] = self.custom_contacts
        n_contacts_existing = len(self.input_dict["Electrodes"][0]["Contacts"])
        self.modify_electrode_custom_parameters(_n_contacts=n_contacts_existing)
        total_current_a = 0.0
        for i in self.custom_contacts:
            if i["Current[A]"] is not False:
                if i["Current[A]"] != 0.0:
                    total_current_a += i["Current[A]"]
        total_current_a = -total_current_a
        self.modify_surface_parameters(name="BrainSurface", current_a=total_current_a)
        self.custom_contacts = []

    def generate_input_dictionary_template(self) -> dict:
        """
        Generates a template for the input dictionary.

        Returns:
            dict: The generated input dictionary template.
        """
        with initialize(config_path="configs"):
            cfg = compose(config_name=self.config_name)
        data = OmegaConf.to_container(cfg.ossdbs_input_json, resolve=True)
        return data

    def generate_output_path(self, custom_outputpath=None) -> None:
        """
        Generates the output path for the JSON file.

        Returns:
            None
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_outputpath:
            self.input_dict["OutputPath"] = f"results/{custom_outputpath}_{now}"
        else:
            self.input_dict["OutputPath"] = (
                f"results/{self.input_dict['OutputPath']}_{now}"
            )
        self.output_path = self.input_dict["OutputPath"]
        os.makedirs(self.output_path, exist_ok=True)

    def modify_electrode_custom_parameters(
        self,
        tip_length=None,
        contact_length=None,
        contact_spacing=None,
        lead_diameter=None,
        total_length=None,
        segment_contact_angle=None,
        n_segments_per_level=None,
        levels=None,
        segmented_levels=None,
        tip_contact=None,
        _n_contacts=None,
    ) -> None:
        custom_params = self.input_dict["Electrodes"][0]["CustomParameters"]
        for key in custom_params.keys():
            if key in [
                "tip_length",
                "contact_length",
                "contact_spacing",
                "lead_diameter",
                "total_length",
                "segment_contact_angle",
                "n_segments_per_level",
                "levels",
                "segmented_levels",
                "tip_contact",
                "_n_contacts",
            ]:
                if locals().get(key):
                    custom_params[key] = locals()[key]
        custom_params["_n_contacts"] = (
            custom_params["n_segments_per_level"]
            * (len(custom_params["segmented_levels"]))
        ) + (custom_params["levels"] - len(custom_params["segmented_levels"]))

    def modify_surface_parameters(
        self, name="BrainSurface", active=None, current_a=None, voltage_v=None
    ):
        custom_surfaces = self.input_dict["Surfaces"]
        for surface in custom_surfaces:
            if surface["Name"] == name:
                if active is not None:
                    surface["Active"] = active
                if current_a is not None:
                    surface["Current[A]"] = current_a
                if voltage_v is not None:
                    surface["Voltage[V]"] = voltage_v

    def get_electrode_custom_parameters(self) -> dict:
        """
        Returns the custom parameters of the electrode.

        Returns:
            dict: The custom parameters of the electrode.
        """
        return self.input_dict["Electrodes"][0]["CustomParameters"]

    def generate_floating_contact(
        self,
        contact_id,
        impedance_real=0.0,
        impedance_imag=0.0,
        max_mesh_size=1.0,
        max_mesh_size_edge=0.035,
    ) -> None:
        floating_contact = copy.deepcopy(_CONTACT_DICT_TEMPLATE)
        floating_contact["Contact_ID"] = contact_id
        floating_contact["Floating"] = True
        floating_contact["Current[A]"] = 0.0
        floating_contact["SurfaceImpedance[Ohmm]"]["real"] = impedance_real
        floating_contact["SurfaceImpedance[Ohmm]"]["imag"] = impedance_imag
        floating_contact["MaxMeshSize"] = max_mesh_size
        floating_contact["MaxMeshSizeEdge"] = max_mesh_size_edge
        self.custom_contacts.append(floating_contact)
        return floating_contact

    def generate_current_ground_contact(
        self,
        contact_id,
        current_a,
        impedance_real=0.0,
        impedance_imag=0.0,
        max_mesh_size=1.0,
        max_mesh_size_edge=0.035,
    ) -> None:
        current_ground_contact = copy.deepcopy(_CONTACT_DICT_TEMPLATE)
        current_ground_contact["Contact_ID"] = contact_id
        current_ground_contact["Active"] = True
        current_ground_contact["Current[A]"] = current_a
        current_ground_contact["Voltage[V]"] = 0.0
        current_ground_contact["SurfaceImpedance[Ohmm]"]["real"] = impedance_real
        current_ground_contact["SurfaceImpedance[Ohmm]"]["imag"] = impedance_imag
        current_ground_contact["MaxMeshSize"] = max_mesh_size
        current_ground_contact["MaxMeshSizeEdge"] = max_mesh_size_edge
        self.custom_contacts.append(current_ground_contact)
        return current_ground_contact

    def generate_current_contact(
        self,
        contact_id,
        current_a,
        impedance_real=0.0,
        impedance_imag=0.0,
        max_mesh_size=1.0,
        max_mesh_size_edge=0.035,
    ) -> None:
        current_contact = copy.deepcopy(_CONTACT_DICT_TEMPLATE)
        current_contact["Contact_ID"] = contact_id
        current_contact["Floating"] = True
        current_contact["Current[A]"] = current_a
        current_contact["SurfaceImpedance[Ohmm]"]["real"] = impedance_real
        current_contact["SurfaceImpedance[Ohmm]"]["imag"] = impedance_imag
        current_contact["MaxMeshSize"] = max_mesh_size
        current_contact["MaxMeshSizeEdge"] = max_mesh_size_edge
        self.custom_contacts.append(current_contact)
        return current_contact

    def generate_voltage_contact(
        self,
        contact_id,
        voltage_v,
        impedance_real=0.0,
        impedance_imag=0.0,
        max_mesh_size=1.0,
        max_mesh_size_edge=0.035,
    ) -> None:
        voltage_contact = copy.deepcopy(_CONTACT_DICT_TEMPLATE)
        voltage_contact["Contact_ID"] = contact_id
        voltage_contact["Active"] = True
        voltage_contact["Voltage[V]"] = voltage_v
        voltage_contact["SurfaceImpedance[Ohmm]"]["real"] = impedance_real
        voltage_contact["SurfaceImpedance[Ohmm]"]["imag"] = impedance_imag
        voltage_contact["MaxMeshSize"] = max_mesh_size
        voltage_contact["MaxMeshSizeEdge"] = max_mesh_size_edge
        self.custom_contacts.append(voltage_contact)
        return voltage_contact

    def modify_json_parameters(self) -> None:
        """
        Modifies the input dictionary and writes it to a JSON file.

        Args:
            results_path (str): The path where the JSON file will be saved.

        Returns:
            None
        """
        self.file_path = f"{self.output_path}/generated_ossdbs_parameters.json"
        with open(self.file_path, "w") as file:
            json.dump(self.input_dict, file, indent=2)

        print(f"JSON template data has been written to {self.file_path}")

    def run_ossdbs(self) -> None:
        """
        Runs the OSS-DBS command line tool with the specified file path.

        Args:
            file_path (str): The path to the input file.

        Returns:
            None
        """
        print("Generating OSS-DBS output...")
        try:
            subprocess.run(["ossdbs", self.file_path], check=True)
            print("OSS-DBS Output Complete")
        except subprocess.CalledProcessError as err:
            print("Error running oss-dbs:", err)

    def plot_electrode(self, output_path=None) -> None:
        if output_path:
            self.output_path = output_path
        title = f"OSS-DBS Electrode Model: {self.output_path}"
        for i, filename in enumerate(self.filenames):
            if filename == "electrode_1.vtu":
                electrode = pv.read(f"{self.output_path}/{filename}")
                self.scalar_array_actor[i] = electrode[self.scalars[i]]
                mesh_actor = self.plotter.add_mesh(
                    electrode,
                    show_edges=False,
                    name=filename,
                    scalars=self.scalar_array_actor[i],
                    scalar_bar_args={
                        "title": self.scalars[i],
                        "title_font_size": 15,
                        "label_font_size": 10,
                        "width": 0.15,
                        "height": 0.025,
                        "position_x": 0.8,
                        "position_y": 0.05 * i,
                    },
                    cmap=self.colormaps[i],
                )
            else:
                mesh = pv.read(f"{self.output_path}/{filename}")
                self.scalar_array_actor[i] = mesh[self.scalars[i]]
                mesh_actor = self.plotter.add_mesh(
                    mesh,
                    style="points",
                    show_edges=True,
                    name=filename,
                    scalars=self.scalar_array_actor[i],
                    scalar_bar_args={
                        "title": self.scalars[i],
                        "title_font_size": 15,
                        "label_font_size": 10,
                        "width": 0.15,
                        "height": 0.025,
                        "position_x": 0.8,
                        "position_y": 0.05 * i,
                    },
                    cmap=self.colormaps[i],
                )
                mesh_actor.SetVisibility(False)
                print(
                    f"{filename} MIN: {self.scalar_array_actor[i].min()}, MAX: {self.scalar_array_actor[i].max()}"
                )
            self.meshes.append(mesh_actor)
            self.plotter.add_checkbox_button_widget(
                lambda flag, idx=i: self.toggle_visibility(flag, idx),
                position=(10, 10 + i * 30),  # Adjust position for each checkbox
                color_on=(0.0, 1.0, 0.0),  # Optional: color when checked (green)
                color_off=(1.0, 0.0, 0.0),  # Optional: color when unchecked (red)
                size=20,  # Optional: size of the checkbox
            )
            self.plotter.add_text(filename, position=(40, 10 + i * 30), font_size=10)
        self.plotter.title = title
        self.plotter.add_text(title, position="upper_left", color="white", font_size=16)
        self.plotter.show_grid()
        self.plotter.show()

    def toggle_visibility(self, flag, index):
        self.meshes[index].SetVisibility(flag)
        self.plotter.render()


def main() -> None:
    """
    This is the main function of the program.
    It creates an instance of CustomElectrodeModeler and performs the necessary operations.
    """
    model_electrode()


def model_electrode():
    """
    This function models the electrode with the specified number of contacts and
    stimulates the first segmented electrode on each.

    Args:
        modeler (CustomElectrodeModeler): The CustomElectrodeModeler instance.

    Returns:
        None
    """
    electrode = CustomElectrodeModeler("ossdbs_input_config")
    electrodes_to_generate = 5
    for i in range(electrodes_to_generate):
        electrode.modify_electrode_custom_parameters(
            total_length=300.0,
            segment_contact_angle=360 / (i + 2) - (20),
            n_segments_per_level=i + 2,
            levels=4,
            segmented_levels=[2, 3, 4],
            tip_contact=True,
        )
        for j in range(electrode.get_electrode_custom_parameters()["_n_contacts"]):
            if j % 2 == 0:
                electrode.generate_floating_contact(j + 1)
            else:
                electrode.generate_current_contact(j + 1, 1.0)
        electrode.generate_output_path(f"electrode_{i+1}")
        electrode.update_parameters()
        electrode.modify_json_parameters()
        electrode.run_ossdbs()
    electrode.plot_electrode()  # Right now this only plots the last electrode generated


if __name__ == "__main__":
    main()