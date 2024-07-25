import argparse
from json_modification_api import CustomElectrodeModeler
import simulation_analysis as sa
from matplotlib import pyplot as plt
import pickle
import numpy as np


def test_current_controlled(rerun=True, block=True) -> None:
    """
    Does the parameter named current_controlled have any effect?
    """
    dir_name = "current_controlled"
    if rerun:
        electrode = CustomElectrodeModeler("ossdbs_input_config")
        electrode.modify_electrode_custom_parameters(
            total_length=300.0,
            segment_contact_angle=60.0,
            n_segments_per_level=4,
            levels=2,
            segmented_levels=[1, 2],
            tip_contact=False,
        )
        active_contact_id = 5
        # Test the effect of "CurrentControlled"
        pts, values, names = [], [], []
        for current_controlled in [True, False]:
            electrode.generate_output_path(
                f"{dir_name}/current_controlled_{current_controlled}"
            )
            for i in range(electrode.get_electrode_custom_parameters()["_n_contacts"]):
                if i == active_contact_id + 1:
                    electrode.generate_current_contact(i + 1, 1.0)
                else:
                    electrode.generate_floating_contact(i + 1)
            electrode.input_dict["StimulationSignal"]["CurrentControlled"] = current_controlled
            electrode.update_parameters()
            electrode.modify_json_parameters()
            electrode.run_ossdbs()
            p, v, n = sa.load_vtu(electrode.output_path + '/potential.vtu')
            pts.append(p)
            values.extend(v)
            names.extend(n)

        with open('results/' + dir_name + '/data.pkl', 'wb') as f:
            pickle.dump((pts, values, names), f)
    else:
        with open('results/' + dir_name + '/data.pkl', 'rb') as f:
            pts, values, names = pickle.load(f)

    plt.plot(values[0], values[1], 'b.')
    plt.plot(plt.xlim(), plt.ylim(), 'k-')
    plt.xlabel('CurrentControlled == True')
    plt.ylabel('CurrentControlled == False')
    plt.title('CurrentControlled Effect')
    plt.show(block=block)
    return pts, values
    """
    Looks like current_controlled isn't doing much here??
    In [45]: (values[0]-values[1]).max()/np.median(values[0])
    Out[45]: 1.5538497782331267e-05

    In [46]: (values[0]-values[1]).max() , np.median(values[0])
    Out[46]: (7.939170711779298e-05, 5.109355372053328)
    """


def test_linearity_monopolar(rerun=True, block=True) -> None:
    """
    Is monopolar stimulation linear?
    """
    dir_name = "experiment_linear_mono"
    if rerun:
        electrode = CustomElectrodeModeler("ossdbs_input_config")
        electrode.modify_electrode_custom_parameters(
            total_length=300.0,
            segment_contact_angle=60.0,
            n_segments_per_level=4,
            levels=2,
            segmented_levels=[1, 2],
            tip_contact=False,
        )
        # Test whether linearity holds for monopolar stimulations
        pts_v, values_v, names_v = [], [], []
        pts_i, values_i, names_i = [], [], []
        contact_list = [[3], [4], [3, 4]]
        for stim_mode in ["voltage", "current"]:
            for run, contacts in enumerate(contact_list):
                electrode.generate_output_path(f"{dir_name}/{stim_mode}_{contacts}")
                for i in range(electrode.get_electrode_custom_parameters()["_n_contacts"]):
                    if (i + 1) in contacts:
                        if stim_mode == "current":
                            electrode.generate_current_contact(i + 1, 1.0)
                        else:
                            electrode.generate_voltage_contact(i + 1, 1.0)
                    else:
                        electrode.generate_floating_contact(i + 1)
                electrode.update_parameters()
                electrode.modify_json_parameters()
                electrode.run_ossdbs()
                p, v, n = sa.load_vtu(electrode.output_path + '/potential.vtu')
                if stim_mode == 'current':
                    pts_i.append(p)
                    values_i.extend(v)
                    names_i.extend(n)
                else:
                    pts_v.append(p)
                    values_v.extend(v)
                    names_v.extend(n)
        with open('results/' + dir_name + '/data.pkl', 'wb') as f:
            pickle.dump((pts_i, values_i, names_i,
                         pts_v, values_v, names_v, contact_list), f)
    else:
        with open('results/' + dir_name + '/data.pkl', 'rb') as f:
            pts_i, values_i, names_i, pts_v, values_v, names_v, contact_list = pickle.load(f)

    plt.figure(figsize=[15, 5])
    for i, values in enumerate([values_v, values_i]):
        # compare linearity
        plt.subplot(1, 3, i + 1)
        plt.plot(values[0] + values[1], values[2], 'b.')
        plt.plot(plt.xlim(), plt.ylim(), 'k-')
        if i == 0:
            str_ = 'Voltage'
        else:
            str_ = 'Current'
        c1 = contact_list[0][0]
        c2 = contact_list[1][0]
        plt.xlabel(f'V(c{c1})+V(c{c2})')
        plt.ylabel('V(c{c1}+c{c2})')
        plt.title(f'Linearity for {str_} stim\n' +
                  f'V(c{c1})+V(c{c2}) vs V(c{c1}+c{c2})')
    # compare voltage vs current stim
    plt.subplot(1, 3, 3)
    plt.plot(values_v[0], values_i[0], 'r.', label='Contact {c1}')
    plt.plot(values_v[1], values_i[1], 'b.', label='Contact {c2}')
    plt.xlabel('V_{Voltage Mode}')
    plt.ylabel('V_{Current Mode}')
    plt.title('Voltage vs Current stim')
    plt.legend()
    plt.show(block=block)
    return values_v, values_i

    """
    Linearity holds for both current and voltage stimulation!
    """


def test_linearity_bipolar(rerun=True, block=True) -> None:
    """
    Is bipolar stimulation linear?
    """
    dir_name = "experiment_linear_bipolar"
    if rerun:
        electrode = CustomElectrodeModeler("ossdbs_input_config")
        electrode.modify_electrode_custom_parameters(
            total_length=300.0,
            segment_contact_angle=60.0,
            n_segments_per_level=4,
            levels=2,
            segmented_levels=[1, 2],
            tip_contact=False,
        )
        # Test whether linearity holds for bipolar stimulations
        contacts = [1, 5]
        experiment_list = {
            f'voltage_monopolar_{contacts[0]}': {
                'contacts': [contacts[0]],
                'voltage': [1.0],
                'stim_mode': 'voltage'
            },
            f'voltage_monopolar_{contacts[1]}': {
                'contacts': [contacts[1]],
                'voltage': [1.0],
                'stim_mode': 'voltage'
            },
            f'voltage_bipolar_{contacts[0]}_{contacts[1]}': {
                'contacts': contacts,
                'voltage': [1.0, -1.0],
                'stim_mode': 'voltage'
            },
            f'current_monopolar_{contacts[0]}': {
                'contacts': [contacts[0]],
                'current': [1.0],
                'stim_mode': 'current'
            },
            f'current_monopolar_{contacts[1]}': {
                'contacts': [contacts[1]],
                'current': [1.0],
                'stim_mode': 'current'
            },
            f'current_bipolar_{contacts[0]}_{contacts[1]}': {
                'contacts': contacts,
                'current': [1.0, -1.0],
                'stim_mode': 'current'
            },
        }
        for name, expt in experiment_list.items():
            electrode.generate_output_path(f"{dir_name}/{name}")
            for i in range(electrode.get_electrode_custom_parameters()["_n_contacts"]):
                if (i + 1) in expt['contacts']:
                    contact_id = expt['contacts'].index(i + 1)
                    if expt['stim_mode'] == "current":
                        electrode.generate_current_contact(
                            i + 1, expt['current'][contact_id])
                    else:
                        electrode.generate_voltage_contact(
                            i + 1, expt['voltage'][contact_id])
                else:
                    electrode.generate_floating_contact(i + 1)
            electrode.update_parameters()
            electrode.modify_json_parameters()
            electrode.run_ossdbs()
            p, v, n = sa.load_vtu(electrode.output_path + '/potential.vtu')
            expt['pts'] = p
            expt['values'] = v[0]
            expt['names'] = n[0]

        with open('results/' + dir_name + '/data.pkl', 'wb') as f:
            pickle.dump((contacts, experiment_list), f)
    else:
        with open('results/' + dir_name + '/data.pkl', 'rb') as f:
            contacts, experiment_list = pickle.load(f)

    plt.figure(figsize=[10, 5])
    for i, stim_mode in enumerate(['voltage', 'current']):
        # bipolar
        bipolar_v = \
            experiment_list[f'{stim_mode}_bipolar_{contacts[0]}_{contacts[1]}']['values']
        # predicted bipolar - difference between to monomopolar stimulations
        predicted_v = (experiment_list[f'{stim_mode}_monopolar_{contacts[0]}']['values'] -
                       experiment_list[f'{stim_mode}_monopolar_{contacts[1]}']['values'])

        plt.subplot(1, 2, i + 1)
        plt.plot(bipolar_v, predicted_v, 'b.')
        plt.plot(plt.xlim(), plt.ylim(), 'k-')
        if i == 0:
            str_ = 'Voltage'
        else:
            str_ = 'Current'
        plt.xlabel('V_{Bipolar}')
        plt.ylabel('V_{Predicted}')
        plt.title(f'Bipolar vs predicted\n{str_} stim')
    plt.show(block=block)
    return experiment_list

    """
    Linearity holds for both current and voltage stimulation!
    """


def test_linearity_multipolar(args, rerun=True, block=True) -> None:
    """
    Is multipolar stimulation linear?
    """
    dir_name = "experiment_linear_multipolar"
    if rerun:
        electrode = CustomElectrodeModeler("ossdbs_input_config")
        electrode.modify_electrode_custom_parameters(
            total_length=300.0,
            segment_contact_angle=60.0,
            n_segments_per_level=4,
            levels=2,
            segmented_levels=[1, 2],
            tip_contact=False,
        )
        # Test whether linearity holds for bipolar stimulations
        contacts = [1, 2, 5]
        experiment_list = {
            f'current_monopolar_{contacts[0]}': {
                'contacts': [contacts[0]],
                'current': [1.0],
                'stim_mode': 'current'
            },
            f'current_monopolar_{contacts[1]}': {
                'contacts': [contacts[1]],
                'current': [1.0],
                'stim_mode': 'current'
            },
            f'current_monopolar_{contacts[2]}': {
                'contacts': [contacts[2]],
                'current': [1.0],
                'stim_mode': 'current'
            },
            f'current_multipolar_{contacts[0]}{contacts[1]}-{contacts[2]}': {
                'contacts': contacts,
                'current': [0.5, 0.5, -1.0],
                'stim_mode': 'current'
            },
        }
        for name, expt in experiment_list.items():
            electrode.generate_output_path(f"{dir_name}/{name}")
            for i in range(electrode.get_electrode_custom_parameters()["_n_contacts"]):
                if (i + 1) in expt['contacts']:
                    contact_id = expt['contacts'].index(i + 1)
                    if expt['stim_mode'] == "current":
                        electrode.generate_current_contact(
                            i + 1, expt['current'][contact_id],
                            max_mesh_size=args.max_mesh_size,
                            max_mesh_size_edge=args.max_mesh_size_edge)
                    else:
                        electrode.generate_voltage_contact(
                            i + 1, expt['voltage'][contact_id],
                            max_mesh_size=args.max_mesh_size,
                            max_mesh_size_edge=args.max_mesh_size_edge)
                else:
                    electrode.generate_floating_contact(
                        i + 1,
                        max_mesh_size=args.max_mesh_size,
                        max_mesh_size_edge=args.max_mesh_size_edge)
            electrode.update_parameters()
            electrode.modify_json_parameters()
            electrode.run_ossdbs()
            p, v, n = sa.load_vtu(electrode.output_path + '/potential.vtu')
            expt['pts'] = p
            expt['values'] = v[0]
            expt['names'] = n[0]

        with open('results/' + dir_name + '/data.pkl', 'wb') as f:
            pickle.dump((contacts, experiment_list), f)
    else:
        with open('results/' + dir_name + '/data.pkl', 'rb') as f:
            contacts, experiment_list = pickle.load(f)

    plt.figure()
    # multipolar
    multipolar_v = \
        experiment_list[f'current_multipolar_{contacts[0]}{contacts[1]}-{contacts[2]}']['values']
    # predicted multipolar - difference between to monomopolar stimulations
    predicted_v = \
        0.5 * experiment_list[f'current_monopolar_{contacts[0]}']['values'] + \
        0.5 * experiment_list[f'current_monopolar_{contacts[1]}']['values'] + \
        -1.0 * experiment_list[f'current_monopolar_{contacts[2]}']['values']
    plt.plot(multipolar_v, predicted_v, 'b.')
    plt.plot(plt.xlim(), plt.ylim(), 'k-')
    plt.xlabel('V_{Multipolar}')
    plt.ylabel('V_{Predicted}')
    plt.title('Multipolar vs predicted: current stim')
    plt.show(block=block)

    return experiment_list


def test_linearity_scale(block=True) -> None:
    """
    Is monopolar stimulation linear?
    """
    dir_name = "experiment_linear_scale"
    electrode = CustomElectrodeModeler("ossdbs_input_config")
    electrode.modify_electrode_custom_parameters(
        total_length=300.0,
        segment_contact_angle=60.0,
        n_segments_per_level=1,
        levels=3,
        segmented_levels=[],
        tip_contact=False,
    )
    # Test whether linearity holds for monopolar stimulations
    values = []
    contact = 1
    current_list = [0.001, 0.01, 0.1, 1]

    stim_mode = "current"
    for current in current_list:
        electrode.generate_output_path(f"{dir_name}/tmp")
        for i in range(electrode.get_electrode_custom_parameters()["_n_contacts"]):
            if (i + 1) == contact:
                if stim_mode == "current":
                    electrode.generate_current_contact(i + 1, current)
                else:
                    electrode.generate_voltage_contact(i + 1, current)
            else:
                electrode.generate_floating_contact(i + 1)
        electrode.update_parameters()
        electrode.modify_json_parameters()
        electrode.run_ossdbs()
        pts, v, n = sa.load_vtu(electrode.output_path + '/potential.vtu')
        values.extend(v)

    plt.figure(figsize=[15, 5])
    # compare linearity
    ct = 0
    for i in range(4):
        for j in range(i + 1, 4):
            ct += 1
            plt.subplot(3, 2, ct)
            plt.plot(values[i], values[j], 'b.')
            xlim = np.array(plt.xlim())
            plt.plot(xlim, xlim * current_list[j] / current_list[i], 'k-')
            plt.xlabel(f'V(c = {current_list[i]})')
            plt.ylabel(f'V(c = {current_list[j]})')
    plt.suptitle('Linearity in scale of current stim')
    plt.show(block=block)

    """
    Linearity holds for both current and voltage stimulation!
    """


def main() -> None:
    """
    This is the main function of the program.
    It creates an instance of CustomElectrodeModeler and performs the necessary operations.
    """
    args = parse_args()

    # if running all, don't block on the first ones
    block = not args.experiment_name == "all"

    if args.experiment_name == "all" or args.experiment_name == "current_controlled":
        test_current_controlled(block=block, rerun=not args.no_rerun)
    if args.experiment_name == "all" or args.experiment_name == "linearity_monopolar":
        test_linearity_monopolar(block=block, rerun=not args.no_rerun)
    if args.experiment_name == "all" or args.experiment_name == "linearity_bipolar":
        test_linearity_bipolar(block=block, rerun=not args.no_rerun)
    if args.experiment_name == "all" or args.experiment_name == "linearity_multipolar":
        test_linearity_multipolar(args, rerun=not args.no_rerun)
    if args.experiment_name == "all" or args.experiment_name == "linearity_scale":
        test_linearity_scale(block=block)


def parse_args() -> argparse.Namespace:
    """
    This function parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model electrodes for OSS-DBS.")
    parser.add_argument("--experiment_name", "-e", type=str, default="all", help="Name of the experiment.",)
    parser.add_argument('--no_rerun', '-n', action='store_true', help="Don't Rerun the simulation, just plot the results.")
    parser.add_argument('-M', '--max_mesh_size', type=float, default=0.5, help="Max mesh size.")
    parser.add_argument('-E', '--max_mesh_size_edge', type=float, default=0.35, help="Max edge mesh size.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
