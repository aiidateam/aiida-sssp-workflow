import importlib

from aiida.plugins import DataFactory

UpfData = DataFactory("pseudo.upf")


def get_extra_parameters_for_lanthanides(element, nbnd) -> dict:
    """
    In rare earth case, increase the initial number of bands,
    otherwise the occupation will not fill up in the highest band
    which always trigger the `PwBaseWorkChain` sanity check.
    The nbnd only take effect for the scf step of PwBandsWorkChain
    since for the bands step, the nbnd is controled by init_nbands_factor
    while `nbnd` will be removed from scf parameters.
    """
    extra_parameters = {
        "SYSTEM": {
            "nspin": 2,
            "starting_magnetization": {
                element: 0.5,
            },
            "nbnd": int(nbnd),
        },
        "ELECTRONS": {
            "diagonalization": "cg",
            "mixing_beta": 0.5,
            "electron_maxstep": 200,
        },
    }

    return extra_parameters


def get_pseudo_N():
    """Return pseudo of nitrogen for lanthanide nitrides"""
    import_path = importlib.resources.path(
        "aiida_sssp_workflow.statics.upf", "N.pbe-n-radius_5.upf"
    )
    with import_path as psp_path, open(psp_path, "rb") as stream:
        pseudo_N = UpfData(stream)

    return pseudo_N


def get_pseudo_O():
    """Return pseudo of oxygen for oxides"""
    import_path = importlib.resources.path(
        "aiida_sssp_workflow.statics.upf", "O.pbe-n-kjpaw_psl.0.1.upf"
    )
    with import_path as psp_path, open(psp_path, "rb") as stream:
        pseudo_O = UpfData(stream)

    return pseudo_O
