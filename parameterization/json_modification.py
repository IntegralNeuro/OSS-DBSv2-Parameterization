import argparse
import json
import os
import re
import subprocess
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

from hydra import compose, initialize  # type: ignore
from omegaconf import OmegaConf  # type: ignore

_CONTACT_DICT_CURRENT_TEMPLATE = {
    "Contact_ID": 8,
    "Active": True,
    "Current[A]": 1.0,
    "Voltage[V]": False,
    "Floating": False,
    "SurfaceImpedance[Ohmm]": {"real": 0.0, "imag": 0.0},
    "MaxMeshSize": 1.0,
    "MaxMeshSizeEdge": 0.035,
}
_CONTACT_DICT_FLOATING_TEMPLATE = {
    "Contact_ID": 8,
    "Active": False,
    "Current[A]": 0.0,
    "Voltage[V]": False,
    "Floating": True,
    "SurfaceImpedance[Ohmm]": {"real": 0.0, "imag": 0.0},
    "MaxMeshSize": 1.0,
    "MaxMeshSizeEdge": 0.035,
}
_WINDOW_WIDTH = 1000
_WINDOW_HEIGHT = 600


class ElectrodeGUI(tk.Tk):
    """
    A class representing the GUI for modifying OSS-DBS parameters.

    Attributes:
        screen_width (int): The width of the screen.
        screen_height (int): The height of the screen.
        x (int): The x-coordinate of the GUI window.
        y (int): The y-coordinate of the GUI window.
        entries (dict): A dictionary to store the created entry widgets.

    Methods:
        __init__(self, input_dict): Initializes the ElectrodeGUI object.
        generate_fields_from_dict(self): Generates entry fields with labels based on the input dictionary.
    """

    def __init__(self, input_dict) -> None:
        """
        Initializes the ElectrodeGUI object.

        Args:
            input_dict (dict): The input dictionary containing the initial values.

        Returns:
            None

        Description:
            - Sets up the GUI window with the title, size, and position.
            - Initializes the entries dictionary to store the created entry widgets.
            - Initializes the row_counter and entry_counter variables.
            - Stores the reference_dict for later use.
            - Creates a frame for the canvas and scrollbar.
            - Adds a canvas and scrollbar to the frame.
            - Configures the canvas with the scrollbar.
            - Creates another frame inside the canvas for content.
            - Initializes the entries and indexes dictionaries.
            - Generates entry fields with labels.
            - Configures columns to expand as the window resizes.
            - Configures row and column weights for the main window.
        """
        super().__init__()
        self.row_counter = 0
        self.entry_counter = 0
        self.outputpath = ""
        self.reference_dict = input_dict

        self.title("OSS-DBS Parameter Modifier")
        self.geometry(f"{_WINDOW_WIDTH}x{_WINDOW_HEIGHT*2}")

        # Create a frame for the canvas and scrollbar
        frame = ttk.Frame(self)
        frame.grid(row=0, column=0, sticky="nsew")

        # Add a canvas in the frame
        self.canvas = tk.Canvas(frame)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Add a scrollbar to the frame
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Configure the canvas with the scrollbar
        self.canvas.configure(
            yscrollcommand=scrollbar.set,
            width=_WINDOW_WIDTH * 0.95,
            height=_WINDOW_HEIGHT,
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        # Create another frame inside the canvas for content
        self.content_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        # Dictionary to store entries
        self.entries = {}
        self.indexes = {}

        # Generate entry fields with labels
        self.generate_fields_from_dict()

        # Configure columns to expand as window resizes
        for col in range(2):
            self.content_frame.grid_columnconfigure(col, weight=1)

        # Configure row and column weights for the main window
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def generate_fields_from_dict(self):
        """
        Generates entry fields with labels based on the input dictionary.

        Returns:
            None

        Description:
            - Iterates through the keys in the reference_dict.
            - If the value associated with a key is a dictionary or list, creates a label for the key.
            - If the value associated with a key is a dictionary, iterates through the secondary keys.
            - If the value associated with a secondary key is a dictionary or list, creates a label for the secondary key.
            - If the value associated with a secondary key is a dictionary, iterates through the tertiary keys.
            - If the value associated with a tertiary key is a dictionary or list, creates a label for the tertiary key.
            - If the value associated with a tertiary key is a dictionary, iterates through the quaternary keys.
            - If the value associated with a quaternary key is a dictionary or list, creates a label for the quaternary key.
            - If the value associated with a quaternary key is a dictionary, iterates through the quinary keys.
            - If the value associated with a quinary key is a dictionary or list, creates a label for the quinary key.
            - If the value associated with a quinary key is a dictionary, iterates through the senary keys.
            - Creates a label and an entry widget for each senary key.
        """
        secondary_key_index_counter = 0
        quaternary_key_index_counter = 0
        for key in self.reference_dict:
            if isinstance(self.reference_dict[key], (dict, list)):
                self.create_label(key)
                for secondary_key in self.reference_dict[key]:
                    if isinstance(secondary_key, dict):
                        for tertiary_key in secondary_key:
                            if isinstance(
                                self.reference_dict[key][secondary_key_index_counter][tertiary_key],
                                (dict, list),
                            ):
                                self.create_label(tertiary_key, level=2)
                                for quaternary_key in self.reference_dict[key][
                                    secondary_key_index_counter
                                ][tertiary_key]:
                                    if isinstance(quaternary_key, dict):
                                        for quinary_key in quaternary_key:
                                            if isinstance(
                                                self.reference_dict[key][
                                                    secondary_key_index_counter
                                                ][tertiary_key][quaternary_key_index_counter][
                                                    quinary_key
                                                ],
                                                (dict, list),
                                            ):
                                                self.create_label(quinary_key, level=4)
                                                for senary_key in self.reference_dict[key][
                                                    secondary_key_index_counter
                                                ][tertiary_key][quaternary_key_index_counter][
                                                    quinary_key
                                                ]:
                                                    self.create_label_and_entry(
                                                        senary_key,
                                                        self.reference_dict[key][
                                                            secondary_key_index_counter
                                                        ][tertiary_key][
                                                            quaternary_key_index_counter
                                                        ][
                                                            quinary_key
                                                        ][
                                                            senary_key
                                                        ],
                                                    )
                                            else:
                                                self.create_label_and_entry(
                                                    quinary_key,
                                                    self.reference_dict[key][
                                                        secondary_key_index_counter
                                                    ][tertiary_key][quaternary_key_index_counter][
                                                        quinary_key
                                                    ],
                                                )
                                        quaternary_key_index_counter += 1
                                    elif (
                                        self.reference_dict[key][secondary_key_index_counter][
                                            tertiary_key
                                        ][quaternary_key]
                                        is None
                                    ):
                                        self.create_label_and_entry(
                                            quaternary_key,
                                            str(
                                                self.reference_dict[key][
                                                    secondary_key_index_counter
                                                ][tertiary_key][quaternary_key]
                                            ),
                                        )
                                    else:
                                        self.create_label_and_entry(
                                            quaternary_key,
                                            self.reference_dict[key][secondary_key_index_counter][
                                                tertiary_key
                                            ][quaternary_key],
                                        )
                            else:
                                self.create_label_and_entry(
                                    tertiary_key,
                                    self.reference_dict[key][secondary_key_index_counter][
                                        tertiary_key
                                    ],
                                )
                        secondary_key_index_counter += 1
                    elif isinstance(self.reference_dict[key][secondary_key], (dict, list)):
                        self.create_label(secondary_key, level=1)
                        for tertiary_key in self.reference_dict[key][secondary_key]:
                            if isinstance(tertiary_key, int):
                                for tertiary_key in self.reference_dict[key][secondary_key]:
                                    self.create_label_and_entry(secondary_key, tertiary_key)
                            elif isinstance(
                                self.reference_dict[key][secondary_key][tertiary_key],
                                (dict, list),
                            ):
                                self.create_label(tertiary_key, level=2)
                                for quaternary_key in self.reference_dict[key][secondary_key][
                                    tertiary_key
                                ]:
                                    if isinstance(quaternary_key, dict):
                                        for quinary_key in quaternary_key:
                                            self.create_label_and_entry(
                                                quinary_key,
                                                self.reference_dict[key][secondary_key][
                                                    tertiary_key
                                                ][quaternary_key_index_counter][quinary_key],
                                            )
                                        quaternary_key_index_counter += 1
                                    else:
                                        self.create_label_and_entry(
                                            quaternary_key,
                                            self.reference_dict[key][secondary_key][tertiary_key][
                                                quaternary_key
                                            ],
                                        )
                                    quaternary_key_index_counter = 0
                            else:
                                self.create_label_and_entry(
                                    tertiary_key,
                                    self.reference_dict[key][secondary_key][tertiary_key],
                                )
                    elif self.reference_dict[key][secondary_key] is None:
                        self.create_label_and_entry(
                            secondary_key, str(self.reference_dict[key][secondary_key])
                        )
                    else:
                        self.create_label_and_entry(
                            secondary_key, self.reference_dict[key][secondary_key]
                        )
                    secondary_key_index_counter = 0
            elif self.reference_dict[key] is None:
                self.create_label_and_entry(key, str(self.reference_dict[key]))
            else:
                self.create_label_and_entry(key, self.reference_dict[key])

        for col in range(2):
            self.content_frame.grid_columnconfigure(col, weight=1)

    def create_label(self, text, level=0):
        """
        Creates a label widget and adds it to the GUI.

        Args:
            text (str): The text to display on the label.
            pady (int, optional): The padding on the y-axis. Defaults to 5.
        """
        label = ttk.Label(self.content_frame, text=text, font=f"Helvetica {12-level} bold")
        label.grid(row=self.row_counter, column=0, padx=5, pady=5, sticky="w")
        ttk.Separator(self.content_frame, orient="horizontal").grid(
            row=self.row_counter, column=1, sticky="ew"
        )
        self.row_counter += 1
        return label

    def create_button(self, text, command):
        """
        Creates a button widget and adds it to the GUI.

        Args:
            text (str): The text to display on the button.
            command (function): The function to call when the button is clicked.
            pady (int, optional): The padding on the y-axis. Defaults to 5.
        """
        button = ttk.Button(self.content_frame, text=text, command=command)
        button.grid(row=self.row_counter, column=0, columnspan=2, pady=5, sticky="ew")

    def create_label_and_entry(self, text, prefill):
        """
        Creates a label and an entry widget and adds them to the GUI.

        Args:
            text (str): The text to display on the label and as the key in the entries dictionary.
            pady (int, optional): The padding on the y-axis. Defaults to 5.

        Returns:
            tuple: A tuple containing the created label and entry widgets.
        """
        self.entry_counter += 1
        index = f"{text}_{self.entry_counter}"
        label = ttk.Label(self.content_frame, text=text)
        entry = ttk.Entry(self.content_frame)

        label.grid(row=self.row_counter, column=0, padx=5, pady=5, sticky="w")
        entry.grid(row=self.row_counter, column=1, padx=5, pady=5, sticky="ew")
        entry.insert(0, prefill)
        self.entries[text] = entry
        self.indexes[index] = entry
        self.row_counter += 1

    def print_entries(self):
        """
        Prints the values entered in the entry widgets.
        """
        print(self.entries)

    def print_indexes(self):
        """
        Prints the values entered in the entry widgets.
        """
        print(self.indexes)

    def get_entry_value(self, name):
        """
        Returns the value entered in the entry widget with the given name.

        Args:
            name (str): The name/key of the entry widget in the entries dictionary.

        Returns:
            str: The value entered in the entry widget, or None if the name is not found in the entries dictionary.
        """
        return self.entries[name].get() if name in self.entries else None

    def get_index_value(self, name, original_type="str"):
        """
        Returns the value entered in the entry widget with the given name.

        Args:
            name (str): The name/key of the entry widget in the entries dictionary.

        Returns:
            str: The value entered in the entry widget, or None if the name is not found in the entries dictionary.
        """
        if isinstance(original_type, bool):
            return bool(int(self.indexes[name].get()))
        if isinstance(original_type, list):
            integer_list = re.findall(r'\d+', self.indexes[name].get())
            integer_list = [int(num) for num in integer_list]
            return integer_list
        if original_type == "None":
            return None
        if "OutputPath" in name:
            self.outputpath = (
                f"{self.indexes[name].get()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(self.outputpath, exist_ok=True)
            return self.outputpath
        return type(original_type)(self.indexes[name].get() if name in self.indexes else None)

    def show_input(self, entry):
        """
        Displays a messagebox with the user's input from the given entry widget.

        Args:
            entry (tk.Entry): The entry widget to get the user's input from.
        """
        user_input = entry.get()
        messagebox.showinfo("Input", f"You entered: {user_input}")


def main():
    """
    This is the main function of the program.
    It generates an input dictionary template and creates a GUI using the input dictionary.
    """
    # args = parse_args()
    input_dict = generate_input_dictionary_template()
    create_gui(input_dict)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Parameters for OSS-DBS JSON Generation")
    return parser.parse_args()


def create_gui(input_dict: dict) -> None:
    """
    Creates a GUI for electrode parameters.

    Args:
        input_dict (dict): A dictionary containing input parameters.

    Returns:
        None
    """
    electrode_gui = ElectrodeGUI(input_dict)
    electrode_gui.create_button(
        "Generate", lambda: generate_output_dict_from_gui(electrode_gui, input_dict)
    )
    electrode_gui.mainloop()


def generate_output_dict_from_gui(gui: ElectrodeGUI, output_dict: dict) -> None:
    """
    Generate the output JSON by modifying the input dictionary based on the GUI entries.

    Args:
        gui (ElectrodeGUI): The GUI object containing the entries.
        output_dict (dict): The input dictionary to be modified.

    Returns:
        None
    """
    universal_counter = 0
    secondary_key_index_counter = 0
    quaternary_key_index_counter = 0

    for key in output_dict:
        if isinstance(output_dict[key], (dict, list)):
            for secondary_key in output_dict[key]:
                if isinstance(secondary_key, dict):
                    for tertiary_key in secondary_key:
                        if isinstance(
                            output_dict[key][secondary_key_index_counter][tertiary_key],
                            (dict, list),
                        ):
                            for quaternary_key in output_dict[key][secondary_key_index_counter][
                                tertiary_key
                            ]:
                                if isinstance(quaternary_key, dict):
                                    for quinary_key in quaternary_key:
                                        if isinstance(
                                            output_dict[key][secondary_key_index_counter][
                                                tertiary_key
                                            ][quaternary_key_index_counter][quinary_key],
                                            (dict, list),
                                        ):
                                            for senary_key in output_dict[key][
                                                secondary_key_index_counter
                                            ][tertiary_key][quaternary_key_index_counter][
                                                quinary_key
                                            ]:
                                                universal_counter += 1
                                                output_dict[key][secondary_key_index_counter][
                                                    tertiary_key
                                                ][quaternary_key_index_counter][quinary_key][
                                                    senary_key
                                                ] = gui.get_index_value(
                                                    f"{senary_key}_{universal_counter}",
                                                    output_dict[key][secondary_key_index_counter][
                                                        tertiary_key
                                                    ][quaternary_key_index_counter][quinary_key][
                                                        senary_key
                                                    ],
                                                )
                                        elif quinary_key in gui.entries:
                                            universal_counter += 1
                                            output_dict[key][secondary_key_index_counter][
                                                tertiary_key
                                            ][quaternary_key_index_counter][
                                                quinary_key
                                            ] = gui.get_index_value(
                                                f"{quinary_key}_{universal_counter}",
                                                output_dict[key][secondary_key_index_counter][
                                                    tertiary_key
                                                ][quaternary_key_index_counter][quinary_key],
                                            )
                                    quaternary_key_index_counter += 1

                                elif quaternary_key in gui.entries:
                                    universal_counter += 1
                                    if (
                                        output_dict[key][secondary_key_index_counter][tertiary_key][
                                            quaternary_key
                                        ]
                                        is None
                                    ):
                                        output_dict[key][secondary_key_index_counter][tertiary_key][
                                            quaternary_key
                                        ] = gui.get_index_value(
                                            f"{quaternary_key}_{universal_counter}",
                                            "None",
                                        )
                                    else:
                                        output_dict[key][secondary_key_index_counter][tertiary_key][
                                            quaternary_key
                                        ] = gui.get_index_value(
                                            f"{quaternary_key}_{universal_counter}",
                                            output_dict[key][secondary_key_index_counter][
                                                tertiary_key
                                            ][quaternary_key],
                                        )
                        elif tertiary_key in gui.entries:
                            universal_counter += 1
                            output_dict[key][secondary_key_index_counter][tertiary_key] = (
                                gui.get_index_value(
                                    f"{tertiary_key}_{universal_counter}",
                                    output_dict[key][secondary_key_index_counter][tertiary_key],
                                )
                            )
                    secondary_key_index_counter += 1

                elif isinstance(output_dict[key][secondary_key], (dict, list)):
                    for tertiary_key in output_dict[key][secondary_key]:
                        if isinstance(tertiary_key, int):
                            if secondary_key in gui.entries:
                                universal_counter += 1
                                output_dict[key][secondary_key] = [
                                    int(gui.get_index_value(f"{secondary_key}_{universal_counter}"))
                                ]
                        elif isinstance(
                            output_dict[key][secondary_key][tertiary_key], (dict, list)
                        ):
                            for quaternary_key in output_dict[key][secondary_key][tertiary_key]:
                                if isinstance(quaternary_key, dict):
                                    for quinary_key in quaternary_key:
                                        universal_counter += 1
                                        output_dict[key][secondary_key][tertiary_key][
                                            quaternary_key_index_counter
                                        ][quinary_key] = gui.get_index_value(
                                            f"{quinary_key}_{universal_counter}",
                                            output_dict[key][secondary_key][tertiary_key][
                                                quaternary_key_index_counter
                                            ][quinary_key],
                                        )
                                    quaternary_key_index_counter += 1
                                elif quaternary_key in gui.entries:
                                    universal_counter += 1
                                    output_dict[key][secondary_key][tertiary_key][
                                        quaternary_key
                                    ] = gui.get_index_value(
                                        f"{quaternary_key}_{universal_counter}",
                                        output_dict[key][secondary_key][tertiary_key][
                                            quaternary_key
                                        ],
                                    )
                                quaternary_key_index_counter = 0
                        elif tertiary_key in gui.entries:
                            universal_counter += 1
                            output_dict[key][secondary_key][tertiary_key] = gui.get_index_value(
                                f"{tertiary_key}_{universal_counter}",
                                output_dict[key][secondary_key][tertiary_key],
                            )

                elif secondary_key in gui.entries:
                    if output_dict[key][secondary_key] is None:
                        universal_counter += 1
                        output_dict[key][secondary_key] = gui.get_index_value(
                            f"{secondary_key}_{universal_counter}", "None"
                        )
                    else:
                        universal_counter += 1
                        output_dict[key][secondary_key] = gui.get_index_value(
                            f"{secondary_key}_{universal_counter}",
                            output_dict[key][secondary_key],
                        )
                secondary_key_index_counter = 0

        elif key in gui.entries:
            if output_dict[key] is None:
                universal_counter += 1
                output_dict[key] = gui.get_index_value(f"{key}_{universal_counter}", "None")
            else:
                universal_counter += 1
                output_dict[key] = gui.get_index_value(
                    f"{key}_{universal_counter}", output_dict[key]
                )
    # gui.destroy()
    modify_json_parameters(output_dict, gui.outputpath)


def generate_input_dictionary_template() -> dict:
    """
    Generates a template for the input dictionary.

    Returns:
        dict: The generated input dictionary template.
    """
    with initialize(config_path="configs"):
        cfg = compose(config_name="ossdbs_input_config")
    data = OmegaConf.to_container(cfg.ossdbs_input_json, resolve=True)
    return data


def modify_json_parameters(input_dict: dict, results_path: str) -> None:
    """
    Modifies the input dictionary and writes it to a JSON file.

    Args:
        input_dict (dict): The dictionary to be modified and written to a JSON file.
        results_path (str): The path where the JSON file will be saved.

    Returns:
        None
    """
    file_path = f"{results_path}/generated_oss-dbs_parameters.json"
    with open(file_path, "w") as file:
        json.dump(input_dict, file, indent=2)

    print(f"JSON template data has been written to {file_path}")

    run_ossdbs(file_path)


def run_ossdbs(file_path: str) -> None:
    """
    Runs the OSS-DBS command line tool with the specified file path.

    Args:
        file_path (str): The path to the input file.

    Returns:
        None
    """
    print("Generating OSS-DBS output...")
    try:
        subprocess.run(["ossdbs", file_path], check=True)
        print("OSS-DBS Output Complete")
    except subprocess.CalledProcessError as err:
        print("Error running oss-dbs:", err)


if __name__ == "__main__":
    main()
