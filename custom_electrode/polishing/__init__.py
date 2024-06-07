# Copyright 2023, 2024 Konstantin Butenko, Shruthi Chakravarthy
# Copyright 2023, 2024 Jan Philipp Payonk, Johannes Reding, Julius Zimmermann
# SPDX-License-Identifier: GPL-3.0-or-later

"""Electrode models for DBS."""
from .abbott_stjude import (
    AbbottStJudeActiveTipModel,
    AbbottStJudeActiveTipParameters,
    AbbottStJudeDirectedModel,
    AbbottStJudeParameters,
)
from .boston_scientific_cartesia import (
    BostonScientificCartesiaHXModel,
    BostonScientificCartesiaParameters,
    BostonScientificCartesiaXModel,
)
from .boston_scientific_vercise import (
    BostonScientificVerciseDirectedModel,
    BostonScientificVerciseDirectedParameters,
    BostonScientificVerciseModel,
    BostonScientificVerciseParameters,
)
from .best_electrode_ever import (
    BestElectrodeEverParameters,
    BestElectrodeEverModel
)
from .defaults import (
    AbbottStJudeActiveTip6142_6145,
    AbbottStJudeActiveTip6146_6149,
    AbbottStJudeDirected6172,
    AbbottStJudeDirected6173,
    BostonScientificCartesiaHX,
    BostonScientificCartesiaX,
    BostonScientificVercise,
    BostonScientificVerciseDirected,
    BestElectrodeEver,
    DixiSEEG5,
    DixiSEEG8,
    DixiSEEG10,
    DixiSEEG12,
    DixiSEEG15,
    DixiSEEG18,
    Medtronic3387,
    Medtronic3389,
    Medtronic3391,
    MedtronicSenSightB33005,
    MedtronicSenSightB33015,
    MicroElectrode,
    MicroProbesRodentElectrode,
    MicroProbesSNEX100,
    NeuroPaceDL344_3_5,
    NeuroPaceDL344_10,
    PINSMedicalL301,
    PINSMedicalL302,
    PINSMedicalL303,
    default_electrode_parameters,
)
from .dixi_microtechniques import (
    DixiSEEGModel,
    DixiSEEGParameters,
)
from .electrode_model_template import ElectrodeModel
from .medtronic import (
    MedtronicModel,
    MedtronicParameters,
    MedtronicSenSightModel,
)
from .micro_probes import (
    MicroProbesRodentElectrodeModel,
    MicroProbesRodentElectrodeParameters,
    MicroProbesSNEX100Model,
    MicroProbesSNEX100Parameters,
)
from .microelectrode import MicroElectrodeModel, MicroElectrodeParameters
from .neuro_pace import NeuroPaceModel, NeuroPaceParameters
from .pins_medical import PINSMedicalModel, PINSMedicalParameters

ELECTRODES = {
    "AbbottStJudeActiveTip6142_6145": AbbottStJudeActiveTip6142_6145,
    "AbbottStJudeActiveTip6146_6149": AbbottStJudeActiveTip6146_6149,
    "AbbottStJudeDirected6172": AbbottStJudeDirected6172,
    "AbbottStJudeDirected6173": AbbottStJudeDirected6173,
    "BostonScientificVercise": BostonScientificVercise,
    "BostonScientificVerciseDirected": BostonScientificVerciseDirected,
    "BestElectrodeEver": BestElectrodeEver,
    "BostonScientificCartesiaX": BostonScientificCartesiaX,
    "BostonScientificCartesiaHX": BostonScientificCartesiaHX,
    "DixiSEEG5": DixiSEEG5,
    "DixiSEEG8": DixiSEEG8,
    "DixiSEEG10": DixiSEEG10,
    "DixiSEEG12": DixiSEEG12,
    "DixiSEEG15": DixiSEEG15,
    "DixiSEEG18": DixiSEEG18,
    "Medtronic3387": Medtronic3387,
    "Medtronic3389": Medtronic3389,
    "Medtronic3391": Medtronic3391,
    "MedtronicSenSightB33015": MedtronicSenSightB33015,
    "MedtronicSenSightB33005": MedtronicSenSightB33005,
    "MicroElectrode": MicroElectrode,
    "MicroProbesRodentElectrode": MicroProbesRodentElectrode,
    "MicroProbesSNEX100": MicroProbesSNEX100,
    "NeuroPaceDL344_3_5": NeuroPaceDL344_3_5,
    "NeuroPaceDL344_10": NeuroPaceDL344_10,
    "PINSMedicalL301": PINSMedicalL301,
    "PINSMedicalL302": PINSMedicalL302,
    "PINSMedicalL303": PINSMedicalL303,
}

ELECTRODE_MODELS = {
    "AbbottStJudeActiveTip6142_6145Custom": AbbottStJudeActiveTipModel,
    "AbbottStJudeActiveTip6146_6149Custom": AbbottStJudeActiveTipModel,
    "AbbottStJudeDirected6172Custom": AbbottStJudeDirectedModel,
    "AbbottStJudeDirected6173Custom": AbbottStJudeDirectedModel,
    "BostonScientificVerciseCustom": BostonScientificVerciseModel,
    "BostonScientificVerciseDirectedCustom": BostonScientificVerciseDirectedModel,
    "BestElectrodeEverCustom": BestElectrodeEverModel,
    "BostonScientificCartesiaXCustom": BostonScientificCartesiaXModel,
    "BostonScientificCartesiaHXCustom": BostonScientificCartesiaHXModel,
    "DixiSEEG5Custom": DixiSEEGModel,
    "DixiSEEG8Custom": DixiSEEGModel,
    "DixiSEEG10Custom": DixiSEEGModel,
    "DixiSEEG12Custom": DixiSEEGModel,
    "DixiSEEG15Custom": DixiSEEGModel,
    "DixiSEEG158ustom": DixiSEEGModel,
    "Medtronic3387Custom": MedtronicModel,
    "Medtronic3389Custom": MedtronicModel,
    "Medtronic3391Custom": MedtronicModel,
    "MedtronicSenSightB33015Custom": MedtronicSenSightModel,
    "MedtronicSenSightB33005Custom": MedtronicSenSightModel,
    "MicroElectrodeCustom": MicroElectrodeModel,
    "MicroProbesRodentElectrodeCustom": MicroProbesRodentElectrodeModel,
    "MicroProbesSNEX100Custom": MicroProbesSNEX100Model,
    "NeuroPaceDL344_3_5Custom": NeuroPaceModel,
    "NeuroPaceDL344_10Custom": NeuroPaceModel,
    "PINSMedicalL301Custom": PINSMedicalModel,
    "PINSMedicalL302Custom": PINSMedicalModel,
    "PINSMedicalL303Custom": PINSMedicalModel,
}


ELECTRODE_PARAMETERS = {
    "AbbottStJudeActiveTipModel": AbbottStJudeActiveTipParameters,
    "AbbottStJudeDirectedModel": AbbottStJudeParameters,
    "BostonScientificVerciseModel": BostonScientificVerciseParameters,
    "BostonScientificVerciseDirectedModel": BostonScientificVerciseDirectedParameters,
    "BestElectrodeEverModel": BestElectrodeEverParameters,
    "BostonScientificCartesiaXModel": BostonScientificCartesiaParameters,
    "BostonScientificCartesiaHXModel": BostonScientificCartesiaParameters,
    "DixiSEEGModel": DixiSEEGParameters,
    "MedtronicModel": MedtronicParameters,
    "MedtronicSenSightModel": MedtronicParameters,
    "MicroElectrodeModel": MicroElectrodeParameters,
    "MicroProbesRodentElectrodeModel": MicroProbesRodentElectrodeParameters,
    "MicroProbesSNEX100Model": MicroProbesSNEX100Parameters,
    "NeuroPaceModel": NeuroPaceParameters,
    "PINSMedicalModel": PINSMedicalParameters,
}

__all__ = (
    "ElectrodeModel",
    "AbbottStJudeActiveTip6142_6145",
    "AbbottStJudeActiveTip6146_6149",
    "AbbottStJudeActiveTipModel",
    "AbbottStJudeDirected6172",
    "AbbottStJudeDirected6173",
    "AbbottStJudeDirectedModel",
    "AbbottStJudeActiveTipParameters",
    "AbbottStJudeParameters",
    "BostonScientificVercise",
    "BostonScientificVerciseDirected",
    "BostonScientificVerciseModel",
    "BostonScientificVerciseDirectedModel",
    "BostonScientificVerciseParameters",
    "BostonScientificVerciseDirectedParameters",
    "BestElectrodeEverModel",
    "BestElectrodeEverParameters",
    "BestElectrodeEver",
    "BostonScientificCartesiaX",
    "BostonScientificCartesiaHX",
    "BostonScientificCartesiaXModel",
    "BostonScientificCartesiaHXModel",
    "BostonScientificCartesiaParameters",
    "DixiSEEG5",
    "DixiSEEG8",
    "DixiSEEG10",
    "DixiSEEG12",
    "DixiSEEG15",
    "DixiSEEG18",
    "DixiSEEGModel",
    "DixiSEEGParameters",
    "Medtronic3387",
    "Medtronic3389",
    "Medtronic3391",
    "MedtronicModel",
    "MedtronicSenSightB33015",
    "MedtronicSenSightB33005",
    "MedtronicParameters",
    "MicroElectrode",
    "MicroElectrodeModel",
    "MicroElectrodeParameters",
    "MicroProbesRodentElectrode",
    "MicroProbesRodentElectrodeModel",
    "MicroProbesRodentElectrodeParameters",
    "MicroProbesSNEX100",
    "MicroProbesSNEX100Model",
    "MicroProbesSNEX100Parameters",
    "NeuroPaceDL344_3_5",
    "NeuroPaceDL344_10",
    "NeuroPaceModel",
    "NeuroPaceParameters",
    "PINSMedicalL301",
    "PINSMedicalL302",
    "PINSMedicalL303",
    "PINSMedicalModel",
    "PINSMedicalParameters",
    "default_electrode_parameters",
)
