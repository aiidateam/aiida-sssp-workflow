import importlib

from aiida import orm
from aiida.plugins import DataFactory

from pseudo_parser.upf_parser import parse_element, parse_pseudo_type

UpfData = DataFactory("pseudo.upf")


def get_pseudo_element_and_type(pseudo):
    """Giving a pseudo, return element and pseudo type as tuple"""
    content = pseudo.get_content()
    element = parse_element(content)
    pseudo_type = parse_pseudo_type(content)

    return element, pseudo_type


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


def clean_workdir(wfnode, all_same_nodes=False, invalid_caching=True):
    """clean the remote folder of all calculation in the workchain node
    return the node pk of cleaned calculation.

    :param wfnode: workflow node to clean its descendat calcjob nodes
    :param all_same_nodes: If `True`, fetch all same node and process the clean. BE CAREFUL!
    :param invalid_caching: If `True`, disable caching the calcjob cleaned, so ph, bands
        not failed because of cleaned node is used as parent scf calculation.
    :return: return the list of nodes pk being cleaned
    """

    def clean(node: orm.CalcJobNode, _invalid_caching=True):
        """clean node workdir"""
        node.outputs.remote_folder._clean()  # pylint: disable=protected-access
        if (
            _invalid_caching
            and "_aiida_cached_from" in node.extras
            and "_aiida_hash" in node.extras
        ):
            # It is implemented in aiida 2.0.0, by setting the is_valid_cache.
            # set the it to disable the caching to precisely control extras.
            # here in order that the correct node is cleaned and caching controlled
            # I only invalid_caching if this node is cached from other node, otherwise
            # that node (should be the node from `_caching` workflow) will not be invalid
            # caching.
            # This ensure that if the calcjob is identically running, it will still be used for
            # further calculation.
            node.delete_extra("_aiida_hash")

        return node.pk

    cleaned_calcs = []
    for descendant_node in wfnode.called_descendants:
        if isinstance(descendant_node, orm.CalcJobNode):
            same_nodes = descendant_node.get_all_same_nodes()
            try:
                if all_same_nodes:
                    # clean all same nodes
                    for n in same_nodes:
                        calc_pk = clean(n, invalid_caching)
                        cleaned_calcs.append(calc_pk)
                else:
                    # clean only this node
                    calc_pk = clean(descendant_node, invalid_caching)
                    cleaned_calcs.append(calc_pk)

            except (IOError, OSError, KeyError) as exc:
                raise RuntimeError(
                    "Failed to clean working dirctory of calcjob"
                ) from exc

    return cleaned_calcs
