
import ngsolve
import sys
from ossdbs.controller import Controller


def main() -> None:
    with ngsolve.TaskManager():
        ossdbs_fem(sys.argv[1])


def ossdbs_fem(json_path: str) -> None:

    controller = Controller(json_path=json_path)
    mesh = controller.mesh()
    solver = controller.solver()
    contacts = controller.contacts()
    conductivity = controller.conductivity()
    strategy = controller.spectrum_mode()
    volume_conductor = controller.volume_conductor()
    controller.coordinates()
    region = controller.region_of_interest()
    points = mesh.included_points(region.coordinates())

    mesh.refine_by_boundaries(contacts.active())
    volume_conductor = volume_conductor(conductivity=conductivity,
                                        mesh=mesh,
                                        contacts=contacts,
                                        solver=solver)
    output = strategy.result(signal=controller.stimulation_signal(),
                             volume_conductor=volume_conductor,
                             points=points)
    output.save(controller.output_path())
    output.save_mesh()


if __name__ == '__main__':
    main()
