import importlib
from typing import Optional

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
        "aiida_sssp_workflow.statics.upf", "N.us.z_5.ld1.theose.v0.upf"
    )
    with import_path as psp_path, open(psp_path, "rb") as stream:
        pseudo_N = UpfData(stream)

    return pseudo_N


def get_pseudo_O():
    """Return pseudo of oxygen for oxides"""
    import_path = importlib.resources.path(
        "aiida_sssp_workflow.statics.upf", "O.paw.z_6.ld1.psl.v0.1.upf"
    )
    with import_path as psp_path, open(psp_path, "rb") as stream:
        pseudo_O = UpfData(stream)

    return pseudo_O


def clean_workdir(node: orm.CalcJobNode) -> Optional[int]:
    """clean remote workdir of nonmenon calcjob"""
    # I have to do only clean nonmenon calcjob since I regard it as a bug that
    # the workdir of cached node point to the identical remote path of nomenon calcjob node.
    # It is like a soft link or shallow copy in caching. This lead to clean the remote path
    # of cached node also destroy the remote path of nonmenon node that may still be used for
    # other subsequent calcjob as `parent_folder`, i.e PH calculation.
    if "_aiida_cached_from" not in node.extras:
        node.outputs.remote_folder._clean()  # pylint: disable=protected-access
        return node.pk
    else:
        return None


def invalid_cache(node: orm.CalcJobNode) -> Optional[int]:
    """invalid cache of cached calcjob, so it will not be used for further caching"""

    if "_aiida_cached_from" in node.extras and "_aiida_hash" in node.extras:
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
    else:
        return None


def operate_calcjobs(wnode, operator, all_same_nodes=False):
    """iterate over desendant calcjob nodes of given workflow node and apply operator.

    :param wnode: workflow node to operating on its descendat calcjob nodes
    :param operator: a function operate to the descendants calcjob node
    :param all_same_nodes: If `True`, fetch all same node and process the operation. BE CAREFUL!
    :return: return the list of nodes pk being cleaned
    """

    cleaned_calcs = []
    for descendant_node in wnode.called_descendants:
        if isinstance(descendant_node, orm.CalcJobNode):
            # the nodes waid for operated.
            nodes = descendant_node.get_all_same_nodes()
            try:
                if not all_same_nodes:
                    # only operated on this node
                    nodes = [descendant_node]

                for n in nodes:
                    pk = operator(n)
                    cleaned_calcs += [pk] if pk else []

            except (IOError, OSError, KeyError) as exc:
                raise RuntimeError(
                    "Failed to clean working dirctory of calcjob"
                ) from exc

    return cleaned_calcs
