#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running EquationOfStateWorkChain example
You can import the necessary nodes of examples-node-archive
(and config pw-6.6 code) to run this example.
"""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import submit
from aiida.plugins import CalculationFactory, WorkflowFactory

PwRelaxWorkChain = WorkflowFactory("quantumespresso.pw.relax")
EOSWorkChain = WorkflowFactory("sssp_workflow.eos")
birch_murnaghan_fit = CalculationFactory("sssp_workflow.birch_murnaghan_fit")


def run_eos(code, structure, pseudos):
    """eos run for silicon"""
    inputs = {
        "structure": structure,
        "scale_count": orm.Int(7),
        "kpoints_distance": orm.Float(0.1),
        "scf": {
            "pw": {
                "code": code,
                "pseudos": pseudos,
                "parameters": orm.Dict(
                    dict={
                        "SYSTEM": {
                            "degauss": 0.00735,
                            "ecutrho": 1600,
                            "ecutwfc": 200,
                            "occupations": "smearing",
                            "smearing": "marzari-vanderbilt",
                        },
                        "ELECTRONS": {
                            "conv_thr": 1e-10,
                        },
                    }
                ),
                "metadata": {
                    "options": {
                        "resources": {"num_machines": 1},
                        "max_wallclock_seconds": 1800 * 2,
                        "withmpi": True,
                    },
                },
            },
        },
    }
    node = submit(EOSWorkChain, **inputs)

    return node


def run_relax(code, structure, pseudos):
    _BASE_PARA = {
        "pw": {
            "code": code,
            "pseudos": pseudos,
            "parameters": orm.Dict(
                dict={
                    "SYSTEM": {
                        "degauss": 0.00735,
                        "ecutrho": 1600,
                        "ecutwfc": 200,
                        "occupations": "smearing",
                        "smearing": "marzari-vanderbilt",
                    },
                    "ELECTRONS": {
                        "conv_thr": 1e-10,
                    },
                }
            ),
            "settings": orm.Dict(dict={"CMDLINE": ["-ndiag", "1"]}),
            "metadata": {
                "options": {
                    "resources": {"num_machines": 1},
                    "max_wallclock_seconds": 1800 * 2,
                    "withmpi": True,
                },
            },
        },
        "kpoints_distance": orm.Float(0.1),
    }
    inputs = {
        "structure": structure,
        "base": _BASE_PARA,
        "base_final_scf": _BASE_PARA,
    }
    node = submit(PwRelaxWorkChain, **inputs)

    return node


if __name__ == "__main__":
    import importlib_resources
    from aiida.orm import load_code, load_node
    from aiida.plugins import DbImporterFactory, WorkflowFactory

    code = load_code("qe-6.6-pw@daint-mc")

    # Retrieve and convert to primitive cell
    # cod = DbImporterFactory('cod')()
    # result = cod.query(id='1010134')
    # cif = result[0].get_cif_node().store()
    # # print(cif)
    # structure = cif.get_structure(store=True, primitive_cell=True) # provenance lost

    structure = load_node(14903)

    # pseudos are from SSSP-v1.1 precision
    fpath = importlib_resources.path(
        "aiida_sssp_workflow.REF.UPFs", "Si.pbe-n-rrkjus_psl.1.0.0.UPF"
    )
    with fpath as path:
        filename = str(path)
        upf_silicon = orm.UpfData.get_or_create(filename)[0]
        si_pseudo = upf_silicon

    fpath = importlib_resources.path("aiida_sssp_workflow.REF.UPFs", "F.oncvpsp.upf")
    with fpath as path:
        filename = str(path)
        upf_silicon = orm.UpfData.get_or_create(filename)[0]
        f_pseudo = upf_silicon

    pseudos = {
        "Si": si_pseudo,
        "F": f_pseudo,
    }

    node = run_relax(code, structure, pseudos)
    #
    # relax_structure = load_node(14881)
    # # node = run_eos(code, relax_structure, pseudos)
    # #
    # node = load_node(14909)
    # volume_energy = node.outputs.output_parameters  # This keep the provenance
    # res = birch_murnaghan_fit(
    #     volume_energy)
    #
    # print(res)  # node.outputs
