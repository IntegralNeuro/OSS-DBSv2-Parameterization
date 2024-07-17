# Copyright 2023, 2024 Konstantin Butenko, Shruthi Chakravarthy
# Copyright 2023, 2024 Jan Philipp Payonk, Johannes Reding
# Copyright 2023, 2024 Tom Reincke, Julius Zimmermann
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass

import netgen
import netgen.occ as occ
import numpy as np

from .electrode_model_template import ElectrodeModel
from .utilities import get_highest_edge, get_lowest_edge


@dataclass
class BestElectrodeEverParameters:
    """Electrode geometry parameters."""

    # dimensions [mm]
    tip_length: float
    contact_length: float
    contact_spacing: float
    lead_diameter: float
    total_length: float

    # customization parameters
    segment_contact_angle: float  # grad
    n_segments_per_level: int
    levels: int
    segmented_levels: list[int]  # from 1 to levels
    tip_contact: bool
    _n_contacts: int
    #_n_contacts = n_segments_per_level * len(segmented_levels) + (levels - len(segmented_levels))

    def get_center_first_contact(self) -> float:
        """Returns distance between electrode tip and center of first contact."""
        if self.tip_contact:
            return 0.5 * self.tip_length
        else:
            return self.tip_length + 0.5 * self.contact_length

    def get_distance_l1_l4(self) -> float:
        """Returns distance between first level contact and the last level of contacts."""
        return (self.levels-1) * (self.contact_length + self.contact_spacing)

class BestElectrodeEverModel(ElectrodeModel):
    """Customize your lead.

    Attributes
    ----------
    parameters : BostonScientificVerciseParameters
        Parameters for the Sick fantasy Vercise geometry.

    rotation : float
        Rotation angle in degree of electrode.

    direction : tuple
        Direction vector (x,y,z) of electrode.

    position : tuple
        Position vector (x,y,z) of electrode tip.
    """

    # customization parameters
    # _n_contacts = 7

    #_n_contacts = n_segments_per_level * len(segmented_levels) + (levels - len(segmented_levels))

    def _construct_encapsulation_geometry(
        self, thickness: float
    ) -> netgen.libngpy._NgOCC.TopoDS_Shape:
        """Generate geometry of encapsulation layer around electrode.

        Parameters
        ----------
        thickness : float
            Thickness of encapsulation layer.

        Returns
        -------
        netgen.libngpy._NgOCC.TopoDS_Shape
        """
        radius = self._parameters.lead_diameter * 0.5 + thickness
        center = tuple(np.array(self._direction) * self._parameters.lead_diameter * 0.5)
        height = self._parameters.total_length - self._parameters.tip_length
        tip = netgen.occ.Sphere(c=center, r=radius)
        lead = occ.Cylinder(p=center, d=self._direction, r=radius, h=height)
        encapsulation = tip + lead
        encapsulation.bc("EncapsulationLayerSurface")
        encapsulation.mat("EncapsulationLayer")
        return encapsulation.Move(v=self._position) - self.geometry

    def _construct_geometry(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
        self._origin = (0, 0, 0)
        contacts = self._contacts()
        # TODO check
        body = self.__body() - contacts
        """
        for edge in contacts.edges:
            if edge.name is not None:
                print(f'CC {edge.name} ',
                      f'({self._radial_distance(edge.center):0.4f}, '
                      f'{np.arctan2(edge.center.y, edge.center.x) * 180 / np.pi:0.4f}, ',
                      f'{edge.center.z:0.4f})')

        for edge in body.edges:
            if edge.name is not None:
                print(f'BB {edge.name} ',
                      f'({self._radial_distance(edge.center):0.4f}, '
                      f'{np.arctan2(edge.center.y, edge.center.x) * 180 / np.pi:0.4f}, ',
                      f'{edge.center.z:0.4f})')
        """
        electrode = occ.Glue([body, contacts])
        """
        for edge in electrode.edges:
            if edge.name is not None:
                print(f'EE {edge.name} ',
                      f'({self._radial_distance(edge.center):0.4f}, '
                      f'{np.arctan2(edge.center.y, edge.center.x) * 180 / np.pi:0.4f}, ',
                      f'{edge.center.z:0.4f})')
        """
        axis = occ.Axis(p=self._origin, d=self._direction)
        rotated_electrode = electrode.Rotate(axis=axis, ang=self._rotation)

        return rotated_electrode.Move(v=self._position)

    def __body(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
        if self._parameters.tip_contact:
            radius = self._parameters.lead_diameter * 0.5
            center = tuple(np.array(self._direction) * radius)
            height = self._parameters.total_length - self._parameters.tip_length
            body = occ.Cylinder(p=center, d=self._direction, r=radius, h=height)
            body.bc(self._boundaries["Body"])
            return body
        else:
            radius = self._parameters.lead_diameter * 0.5
            center = tuple(np.array(self._direction) * radius)
            tip = occ.Sphere(c=center, r=radius)
            height = self._parameters.total_length - self._parameters.tip_length
            lead = occ.Cylinder(p=center, d=self._direction, r=radius, h=height)
            body = tip + lead
            body.bc(self._boundaries["Body"])
            return body

    def _contacts(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:

        # compute total number of contacts
        # self._n_contacts = self._parameters.n_segments_per_level * self._parameters.segmented_levels + (self._parameters.levels - self._parameters.segmented_levels)
        if 1 in self._parameters.segmented_levels and self._parameters.tip_contact:
            print("Tip contact cannot be segmented!")
            raise SystemExit

        vectors = []
        if self._parameters.tip_contact:
            distance = self._parameters.tip_length + self._parameters.contact_spacing
        else:
            distance = self._parameters.tip_length

        N_steps = self._parameters.levels

        for i in range(0, N_steps):
            if i == 0 and self._parameters.tip_contact:
                # insert null shift if tip is used for the code consistency
                vectors.append(tuple(np.array(self._direction) * 0.0))
                continue

            vectors.append(tuple(np.array(self._direction) * distance))
            distance += (
                self._parameters.contact_length + self._parameters.contact_spacing
            )
        radius = self._parameters.lead_diameter * 0.5
        direction = self._direction
        height = self._parameters.contact_length
        axis = occ.Axis(p=self._origin, d=self._direction)

        center = tuple(np.array(direction) * radius)
        # define half space at tip_center
        # to construct a hemisphere as part of the contact tip
        if self._parameters.tip_contact:
            half_space = netgen.occ.HalfSpace(p=center, n=direction)
            contact_tip = occ.Sphere(c=center, r=radius) * half_space
            h_pt2 = self._parameters.tip_length - radius
            contact_pt2 = occ.Cylinder(p=center, d=direction, r=radius, h=h_pt2)
            contact_1 = contact_tip + contact_pt2

        contact = occ.Cylinder(p=self._origin, d=self._direction, r=radius, h=height)
        contact_directed = self._contact_directed()

        contacts = []
        segmented_indices = []
        for level_i in range(0, N_steps):
            if level_i + 1 in self._parameters.segmented_levels:
                intersegment_angle =  360.0 / self._parameters.n_segments_per_level
                for segment_i in range(0, self._parameters.n_segments_per_level):
                    if segment_i == 0:
                        c = contact_directed.Move(v=vectors[level_i])
                    else:
                        c = contact_directed.Rotate(axis,intersegment_angle * segment_i).Move(v=vectors[level_i])
                    segmented_indices.append(len(contacts) + 1)
                    contacts.append(c)
            else:
                if level_i == 0 and self._parameters.tip_contact:
                    contacts.append(contact_1)
                else:
                    contacts.append(contact.Move(v=vectors[level_i]))

        for index, contact in enumerate(contacts, 1):
            if index in segmented_indices:
                self._label_directed_contact(contact, f"Contact_{index}")
            else:
                self._label_cylinder_contact(contact, f"Contact_{index}")
            """ Diagnostic to understand/refine geometry setting
            self._print_contact(contact)
            """

        return netgen.occ.Fuse(contacts)

    def _radial_distance(self, center):
        # Calculate the radial distance between from a point
        # to the center of the electrode
        # center: center of face or edge [occ.point]
        # return: float
        d_vec = np.array([center.x - self._origin[0], center.y - self._origin[1], center.z - self._origin[2]])
        a_vec = np.array(self._direction)
        return np.linalg.norm(np.cross(d_vec, a_vec))

    def _label_cylinder_contact(self, contact, name):
        contact.name = name
        # find active face: the one with 4 faces
        for face in contact.faces:
            if len(face.edges) == 4:
                active_face = face
                break
        # Label active face and its edges
        active_face.name = name
        for edge in active_face.edges:
            r = self._radial_distance(edge.center)
            if np.isclose(r, 0):
                edge.name = name

    def _label_directed_contact(self, contact, name):
        contact.name = name
        # find active face, the one whosse center is farthest from electrode center
        max_r = 0
        for face in contact.faces:
            r = self._radial_distance(face.center)
            if r > max_r:
                max_r = r
                active_face = face
        # Label active face and its edges
        active_face.name = name
        for edge in active_face.edges:
            """ Diagnostic to understand missing edge meshes
            if not np.isclose(self._radial_distance(edge.center), 0.65):
                edge.name = name
            """
            edge.name = name

    def _print_contact(self, contact):
        print(f'\nCONTACT {contact.name}\n')
        for i, f in enumerate(contact.faces):
            print(f'Face_{i}: {str(f.name):12}',
                  f'rad={self._radial_distance(f.center):0.4f}\t',
                  f'ang={np.arctan2(f.center.y, f.center.x) * 180 / np.pi:0.4f}\t',
                  f'z={f.center.z:0.4f}')
            for j, e in enumerate(f.edges):
                print(f'edge_{j}: {str(e.name):12}',
                      f'rad={self._radial_distance(e.center):0.4f}\t',
                      f'ang={np.arctan2(e.center.y, e.center.x) * 180 / np.pi:0.4f}\t',
                      f'z={e.center.z:0.4f}')
            print('')

    def _contact_directed(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
        radius = self._parameters.lead_diameter * 0.5
        """ Diagnostic to understand missing edge meshes
        radius += 0.5
        """
        height = self._parameters.contact_length
        body = occ.Cylinder(p=self._origin, d=self._direction, r=radius, h=height)
        # tilted y-vector marker is in YZ-plane and orthogonal to _direction
        new_direction = (0, self._direction[2], -self._direction[1])
        eraser = occ.HalfSpace(p=self._origin, n=new_direction)
        angle = 90 - self._parameters.segment_contact_angle / 2
        axis = occ.Axis(p=self._origin, d=self._direction)

        contact = body - eraser.Rotate(axis, angle) - eraser.Rotate(axis, -angle)

        # Having trouble with body-seam lying inside contacts
        # Going to try placing the first contact's gap at the origin (+x axis)
        contact_angle = np.arctan2(contact.center.y, contact.center.x) * 180 / np.pi
        gap_angle = 360 / self._parameters.n_segments_per_level - self._parameters.segment_contact_angle
        rotation_angle = (gap_angle + self._parameters.segment_contact_angle) / 2 - contact_angle
        return contact.Rotate(axis, rotation_angle)
