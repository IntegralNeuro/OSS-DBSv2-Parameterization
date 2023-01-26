from ossdbs.electrodes import AbbottStjudeActiveTip6146_6149
from tests.geometry_converter import GeometryConverter
import pytest
import netgen
import ngsolve
import json


class TestAbbottStJudeActiveTip6146_6149():
    FILE_PREFIX = "tests/test_data/AbbottStjudeActiveTip6146_6149"

    TESTDATA = [
        # electrode_parameters (Rotation, Translation, Direction), file_path
        ((0.0, (0, 0, 0), (0, 0, 1)), FILE_PREFIX + '_0.json'),
        ((0.0, (0, 0, 0), (0, 0, 0)), FILE_PREFIX + '_0.json'),
        ((3.0, (0, 0, 0), (0, 0, 1)), FILE_PREFIX + '_0.json'),
        ((0.0, (1, -2, 3), (0, 0, 1)), FILE_PREFIX + '_1.json'),
        ((0.0, (1, -2, 3), (0, 0, 0)), FILE_PREFIX + '_1.json'),
        ((3.0, (1, -2, 3), (0, 0, 0)), FILE_PREFIX + '_1.json'),
        ((0.0, (1, -2, 3), (2.0, 0, 1.0)), FILE_PREFIX+'_2.json'),
        ((0.0, (1, -2, 3), (2.0/3.0, 0, 1.0/3.0)), FILE_PREFIX+'_2.json'),
        ]

    def geometry_to_dictionary(self,
                               geometry: netgen.libngpy._NgOCC.TopoDS_Shape
                               ) -> dict:
        return GeometryConverter(geometry).to_dictionary()

    def load_geometry_data(self, path: str) -> dict:
        with open(path, "r") as file:
            geometry_data = json.load(file)
        return geometry_data

    @pytest.mark.parametrize('electrode_parameters, path', TESTDATA)
    def test_generate_geometry(self, electrode_parameters, path) -> None:
        rotation, translation, direction = electrode_parameters
        electrode = AbbottStjudeActiveTip6146_6149(rotation,
                                                   direction,
                                                   translation)
        geometry = electrode.generate_geometry()
        desired = self.load_geometry_data(path=path)
        assert desired == self.geometry_to_dictionary(geometry)

    def test_generate_geometry_default(self):
        electrode = AbbottStjudeActiveTip6146_6149()
        geometry = electrode.generate_geometry()
        desired = self.load_geometry_data(path=self.FILE_PREFIX+'_0.json')
        assert desired == self.geometry_to_dictionary(geometry)

    def test_rename_boundaries(self):
        electrode = AbbottStjudeActiveTip6146_6149()
        electrode.rename_boundaries({'Body': 'RenamedBody',
                                     'Contact_1': 'RenamedContact_1',
                                     'NonExistingPart': 'NonExistingPart'})
        geometry = electrode.generate_geometry()
        netgen_geometry = netgen.occ.OCCGeometry(geometry)
        mesh = ngsolve.Mesh(netgen_geometry.GenerateMesh())
        desired = set(['RenamedBody',
                       'RenamedContact_1',
                       'Contact_2',
                       'Contact_3',
                       'Contact_4'])
        assert desired == set(mesh.GetBoundaries())