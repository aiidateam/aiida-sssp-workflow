import importlib

from aiida.plugins import DataFactory

UpfData = DataFactory("pseudo.upf")


def get_extra_parameters_and_pseudos_for_lanthanoid(element, pseudo_RE):
    import_path = importlib.resources.path(
        "aiida_sssp_workflow.statics.upf", "N.pbe-n-radius_5.upf"
    )
    with import_path as psp_path, open(psp_path, "rb") as stream:
        pseudo_N = UpfData(stream)

    pseudos = {
        element: pseudo_RE,
        "N": pseudo_N,
    }

    # In rare earth case, increase the initial number of bands,
    # otherwise the occupation will not fill up in the highest band
    # which always trigger the `PwBaseWorkChain` sanity check.
    # The nbnd only take effect for the scf step of PwBandsWorkChain
    # since for the bands step, the nbnd is controled by init_nbands_factor
    # while `nbnd` is removed from scf parameters.
    nbands = pseudo_RE.z_valence + pseudo_N.z_valence
    nbands_factor = 2

    extra_parameters = {
        "SYSTEM": {
            "nspin": 2,
            "starting_magnetization": {
                element: 1.0,
            },
            "nbnd": int(nbands * nbands_factor),
        },
    }

    return extra_parameters, pseudos
