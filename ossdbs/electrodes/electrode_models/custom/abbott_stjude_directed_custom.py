# Abbott/St Jude Directed 6172
from ..electrode import ElectrodeModel
import netgen
import netgen.occ as occ
import numpy as np
import os
import json


class AbbottStjudeDirectedCustom(ElectrodeModel):
    """Abbott/St Jude Directed 6172 electrode.

    Attributes
    ----------
    rotation : float
        Rotation angle in degree of electrode.

    direction : tuple
        Direction vector (x,y,z) of electrode.

    position : tuple
        Position vector (x,y,z) of electrode tip.
    """

    # dimensions [mm]
    TIP_LENGTH = 1.1
    CONTACT_LENGTH = 1.5
    CONTACT_SPACING = 1.5
    LEAD_DIAMETER = 1.3
    TOTAL_LENGTH = 100.0

    def __init__(self,
                 rotation: float = 0.0,
                 direction: tuple = (0, 0, 1),
                 position: tuple = (0, 0, 0)) -> None:
        self.__position = position
        self.__rotation = rotation
        norm = np.linalg.norm(direction)
        self.__direction = tuple(direction / norm) if norm else (0, 0, 1)
        self.__load_parameters()
        self.__boundaries = {'Body': 'Body',
                             'Contact_1': 'Contact_1',
                             'Contact_2': 'Contact_2',
                             'Contact_3': 'Contact_3',
                             'Contact_4': 'Contact_4',
                             'Contact_5': 'Contact_5',
                             'Contact_6': 'Contact_6',
                             'Contact_7': 'Contact_7',
                             'Contact_8': 'Contact_8'}

    def capsule_geometry(self, thickness: float) \
            -> netgen.libngpy._NgOCC.TopoDS_Shape:
        """Generate geometry of capsule layer around electrode.

        Parameters
        ----------
        thickness : float
            Thickness of encapsulating layer.

        Returns
        -------
        geometry : netgen.libngpy._NgOCC.TopoDS_Shape
        """
        center = tuple(np.array(self.__direction) * self.LEAD_DIAMETER * 0.5)
        radius = self.LEAD_DIAMETER * 0.5 + thickness
        height = self.TOTAL_LENGTH - self.TIP_LENGTH
        tip = occ.Sphere(c=center, r=radius)
        lead = occ.Cylinder(p=center, d=self.__direction, r=radius, h=height)
        capsule = tip + lead
        capsule.bc('Capsule')
        capsule.mat('Capsule')
        return capsule.Move(v=self.__position) - self.geometry()

    def geometry(self) -> netgen.libngpy._meshing.Mesh:
        """Generate geometry of electrode.

        Returns
        -------
        geometry : netgen.libngpy._NgOCC.TopoDS_Shape
        """
        contacts = self.__contacts()
        body = self.__body() - contacts
        electrode = occ.Glue([body, contacts])
        axis = occ.Axis(p=(0, 0, 0), d=self.__direction)
        rotated_electrode = electrode.Rotate(axis=axis, ang=self.__rotation)
        return rotated_electrode.Move(v=self.__position)

    def set_contact_names(self, boundaries: dict) -> None:
        """Set the names of electrode contacts.

        Parameters
        ----------
        contact_names : dict
            {'Body': 'body_name',
             'Contact_1': 'contact_name',
             'Contact_2': ...}
        """
        self.__boundaries.update(boundaries)

    def __body(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
        radius = self.LEAD_DIAMETER * 0.5
        center = tuple(np.array(self.__direction) * radius)
        tip = occ.Sphere(c=center, r=radius)
        height = self.TOTAL_LENGTH - self.TIP_LENGTH
        lead = occ.Cylinder(p=center, d=self.__direction, r=radius, h=height)
        body = tip + lead
        body.bc(self.__boundaries['Body'])
        return body

    def __contacts(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:

        distance_1 = self.TIP_LENGTH
        distance_2 = distance_1 + self.CONTACT_LENGTH + self.CONTACT_SPACING
        distance_3 = distance_2 + self.CONTACT_LENGTH + self.CONTACT_SPACING
        distance_4 = distance_3 + self.CONTACT_LENGTH + self.CONTACT_SPACING

        vector_1 = tuple(np.array(self.__direction) * distance_1)
        vector_2 = tuple(np.array(self.__direction) * distance_2)
        vector_3 = tuple(np.array(self.__direction) * distance_3)
        vector_4 = tuple(np.array(self.__direction) * distance_4)

        point = (0, 0, 0)
        radius = self.LEAD_DIAMETER * 0.5
        height = self.CONTACT_LENGTH
        contact = occ.Cylinder(p=point, d=self.__direction, r=radius, h=height)
        axis = occ.Axis(p=point, d=self.__direction)

        contact_directed = self.__contact_directed()
        contacts = [contact.Move(vector_1),
                    contact_directed.Move(vector_2),
                    contact_directed.Rotate(axis, 120).Move(vector_2),
                    contact_directed.Rotate(axis, 240).Move(vector_2),
                    contact_directed.Move(vector_3),
                    contact_directed.Rotate(axis, 120).Move(vector_3),
                    contact_directed.Rotate(axis, 240).Move(vector_3),
                    contact.Move(vector_4)
                    ]

        for index, contact in enumerate(contacts, 1):
            name = self.__boundaries['Contact_{}'.format(index)]
            contact.bc(name)
            for edge in contact.edges:
                edge.name = name

        return occ.Fuse(contacts)

    def __contact_directed(self) -> netgen.libngpy._NgOCC.TopoDS_Shape:
        point = (0, 0, 0)
        radius = self.LEAD_DIAMETER * 0.5
        height = self.CONTACT_LENGTH
        body = occ.Cylinder(p=point, d=self.__direction, r=radius, h=height)
        new_direction = tuple(np.cross(self.__direction_2(), self.__direction))
        eraser = occ.HalfSpace(p=point, n=new_direction)
        delta = 15
        angle = 30 + delta
        axis = occ.Axis(p=point, d=self.__direction)
        return body - eraser.Rotate(axis, angle) - eraser.Rotate(axis, -angle)

    def __direction_2(self):
        x, y, z = self.__direction

        if not x and not y:
            return (0, 1, 0)

        if not x and not z:
            return (0, 0, 1)

        if not y and not z:
            return (0, 1, 0)

        return (x, y, not z)

    def __load_parameters(self) -> None:
        dir_name = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dir_name, 'abbott_stjude_directed_custom.json')

        with open(path, 'r') as json_file:
            parameters = json.load(json_file)

        self.CONTACT_LENGTH = parameters['ContactLength[mm]']
        self.TIP_LENGTH = parameters['TipLength[mm]']
        self.CONTACT_SPACING = parameters['ContactSpacingAxial[mm]']
        self.LEAD_DIAMETER = parameters['LeadDiameter[mm]']
        self.TOTAL_LENGHTH = parameters['LeadLength[mm]']
