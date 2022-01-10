#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running verification workchain example
"""
import os

from aiida import orm

from aiida.plugins import WorkflowFactory, DataFactory
from aiida.engine import run_get_node

UpfData = DataFactory('pseudo.upf')
VerificationWorkChain = WorkflowFactory('sssp_workflow.verification')

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_static')

def run_verification(pw_code, ph_code, upf):
    inputs = {
        'pw_code': pw_code,
        'ph_code': ph_code,
        'pseudo': upf,
        'protocol': orm.Str('test'),
        'properties_list': orm.List(list=[
            # 'delta_factor',
            'convergence:cohesive_energy',
            # 'convergence:phonon_frequencies',
            # 'convergence:pressure',
        ]),
        'options': orm.Dict(
                dict={
                    'resources': {
                        'num_machines': 1,
                        'num_mpiprocs_per_machine': 4,
                    },
                    'max_wallclock_seconds': 1800 * 3,
                    'withmpi': True,
                }),
        # 'parallelization': orm.Dict(dict={}),
        'clean_workdir_level': orm.Int(0),
    }

    res, node = run_get_node(VerificationWorkChain, **inputs)
    return res, node

if __name__ == '__main__':
    from aiida.orm import load_code

    pw_code = load_code('pw-6.7@localhost')
    ph_code = load_code('ph-6.7@localhost')

    upf = {}
    pp_label = 'psl/Si.pbe-n-rrkjus_psl.1.0.0.UPF'
    pp_name = pp_label.split('/')[1]
    pp_path = os.path.join(STATIC_DIR, pp_name)
    with open(pp_path, 'rb') as stream:
        pseudo = UpfData(stream)
        upf['si'] = pseudo

    for element, upf in upf.items():
        res, node = run_verification(pw_code, ph_code, upf)
        node.description = pp_label
        print(node)
