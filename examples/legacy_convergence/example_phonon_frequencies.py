#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running phonon frequencies convergence workflow
"""
import numpy as np
import os

from aiida import orm
from aiida.plugins import WorkflowFactory, DataFactory

from aiida.engine import run_get_node

UpfData = DataFactory('pseudo.upf')
ConvergencePhononFrequenciesWorkChain = WorkflowFactory(
    'sssp_workflow.legacy_convergence.phonon_frequencies')

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../', '_static')


def run_phonon_cov(pw_code, ph_code, upf, dual=4.0):
    inputs = {
        'pw_code': pw_code,
        'ph_code': ph_code,
        'pseudo': upf,
        'protocol': orm.Str('test'),
        'dual': orm.Float(dual),
        'options': orm.Dict(
                dict={
                    'resources': {
                        'num_machines': 1
                    },
                    'max_wallclock_seconds': 1800 * 3,
                    'withmpi': False,
                }),
        'parallelization': orm.Dict(dict={}),
        'clean_workdir': orm.Bool(False),
    }
    res, node = run_get_node(ConvergencePhononFrequenciesWorkChain, **inputs)

    return res, node


if __name__ == '__main__':
    from aiida.orm import load_code

    pw_code = load_code('pw-6.7@localhost')
    ph_code = load_code('ph-6.7@localhost')

    upf_sg15 = {}
    # sg15/Si_ONCV_PBE-1.2.upf
    pp_name = 'Si_ONCV_PBE-1.2.upf'
    pp_path = os.path.join(STATIC_DIR, pp_name)
    with open(pp_path, 'rb') as stream:
        pseudo = UpfData(stream)
        upf_sg15['si'] = pseudo

    for element, upf in upf_sg15.items():
        res, node = run_phonon_cov(pw_code, ph_code, upf, dual=4.0)
        node.description = f'sg15/{element}'
        print(node)
