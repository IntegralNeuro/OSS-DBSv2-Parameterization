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
    segment_contact_angle = 100.0  # grad
    n_segments_per_level = 3
    levels = 3
    segmented_levels = [1,3]  # from 1 to levels
    tip_contact = True
    _n_contacts = 7
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
    _n_contacts = 7

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
        contacts = self._contacts()
        # TODO check
        electrode = occ.Glue([self.__body() - contacts, contacts])
        axis = occ.Axis(p=(0, 0, 0), d=self._direction)
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
        #self._n_contacts = self._parameters.n_segments_per_level * self._parameters.segmented_levels + (self._parameters.levels - self._parameters.segmented_levels)

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
        point = (0, 0, 0)
        height = self._parameters.contact_length
        axis = occ.Axis(p=point, d=self._direction)

        center = tuple(np.array(direction) * radius)
        # define half space at tip_center
        # to construct a hemisphere as part of the contact tip
        if self._parameters.tip_contact:
            half_space = netgen.occ.HalfSpace(p=center, n=direction)
            contact_tip = occ.Sphere(c=center, r=radius) * half_space
            h_pt2 = self._parameters.tip_length - radius
            contact_pt2 = occ.Cylinder(p=center, d=direction, r=radius, h=h_pt2)
            contact_1 = contact_tip + contact_pt2
            #contact_last = occ.Cylinder(p=point, d=self._direction, r=radius, h=height)

            ## dumb way
            #variable_name = "contact_" + str(_n_contacts)
            #exec(f"{variable_name} = {occ.Cylinder(p=point, d=self._direction, r=radius, h=height)}")

        contact = occ.Cylinder(p=point, d=self._direction, r=radius, h=height)

        contact_directed = self._contact_directed()

        contacts = []

        # # first level explicitly
        # if self._parameters.tip_contact:
        #     if 0 in self._parameters.segmented_levels:
        #         print("tip contacts cannot be segmented!")
        #         raise SystemExit
        #     else:
        #         contacts.append(contact_1)
        # else:
        #     if 0 in self._parameters.segmented_levels:
        #         contacts.append(contact_directed.Move(v=vectors[0]))
        #         intersegment_angle =  360.0 / self._parameters.n_segments_per_level
        #         for segment_i in range(1, self._parameters.n_segments_per_level):
        #             contacts.append(contact_directed.Rotate(axis, intersegment_angle * segment_i).Move(v=vectors[0]))
        #     else:
        #         contacts.append(contact.Move(v=vectors[0]))

        for level_i in range(0,N_steps):

            if level_i+1 in self._parameters.segmented_levels:
                contacts.append(contact_directed.Move(v=vectors[level_i]))
                intersegment_angle =  360.0 / self._parameters.n_segments_per_level
                for segment_i in range(1, self._parameters.n_segments_per_level):
                    contacts.append(contact_directed.Rotate(axis, intersegment_angle * segment_i).Move(v=vectors[level_i]))
            else:
                if level_i == 0 and self._parameters.tip_contact:
                    contacts.append(contact_1)
                else:
                    contacts.append(contact.Move(v=vectors[level_i]))

        print(contacts)

        for index, contact in enumerate(contacts, 1):
            name = self._boundaries[f"Contact_{index}"]
            print(name)
            contact.bc(name)
            # Label max z value and min z value for contact_14
            if index == self._n_contacts or (self._parameters.tip_contact and index == 1):
                min_edge = get_lowest_edge(contact)
                min_edge.name = name
            # Only label contact edge with maximum z value for contact_1
            if index == 1 or index == self._n_contacts:
                max_edge = get_highest_edge(contact)
                max_edge.name = name
            else:
                # Label all the named contacts appropriately
                for edge in contact.edges:
                    if edge.name is not None:
                        edge.name = name
        return netgen.occ.Fuse(contacts)

    def _contact_directed(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
        point = (0, 0, 0)
        radius = self._parameters.lead_diameter * 0.5
        height = self._parameters.contact_length
        body = occ.Cylinder(p=point, d=self._direction, r=radius, h=height)
        # tilted y-vector marker is in YZ-plane and orthogonal to _direction
        new_direction = (0, self._direction[2], -self._direction[1])
        eraser = occ.HalfSpace(p=point, n=new_direction)
        angle = 90 - self._parameters.segment_contact_angle / 2
        axis = occ.Axis(p=point, d=self._direction)

        contact = body - eraser.Rotate(axis, angle) - eraser.Rotate(axis, -angle)
        # Centering contact to label edges
        contact = contact.Rotate(axis, angle)
        # TODO refactor / wrap in function
        # Find  max z, min z, max x, and max y values and label min x and min y edge
        max_z_val = max_y_val = max_x_val = float("-inf")
        min_z_val = float("inf")
        for edge in contact.edges:
            if edge.center.z > max_z_val:
                max_z_val = edge.center.z
            if edge.center.z < min_z_val:
                min_z_val = edge.center.z
            if edge.center.x > max_x_val:
                max_x_val = edge.center.x
                max_x_edge = edge
            if edge.center.y > max_y_val:
                max_y_val = edge.center.y
                max_y_edge = edge
        max_x_edge.name = "max x"
        max_y_edge.name = "max y"
        # Label only the outer edges of the contact with min z and max z values
        for edge in contact.edges:
            if np.isclose(edge.center.z, max_z_val) and not (
                np.isclose(edge.center.x, radius / 2)
                or np.isclose(edge.center.y, radius / 2)
            ):
                edge.name = "max z"
            elif np.isclose(edge.center.z, min_z_val) and not (
                np.isclose(edge.center.x, radius / 2)
                or np.isclose(edge.center.y, radius / 2)
            ):
                edge.name = "min z"

        # TODO check that the starting axis of the contacts
        # are correct according to the documentation
        contact = contact.Rotate(axis, -angle)

        return contact
