# -*- coding: utf-8 -*-
"""doc"""


def test_update_dict():
    """test update_dict"""
    import copy

    from aiida_sssp_workflow.utils import update_dict

    old_dict = {
        "SYSTEM": {
            "degauss": 0.02,
            "occupations": "smearing",
            "smearing": "cold",
        },
        "ELECTRONS": {
            "conv_thr": 1e-10,
        },
    }
    old_expect = copy.deepcopy(old_dict)

    new_inputs = {
        "SYSTEM": {
            "ecutrho": 1600,
            "ecutwfc": 200,
            "smearing": "fd",
        },
    }

    new_dict = update_dict(old_dict, new_inputs)

    assert new_dict["SYSTEM"]["degauss"] == 0.02
    assert new_dict["SYSTEM"]["smearing"] == "fd"
    assert new_dict["SYSTEM"]["ecutrho"] == 1600
    assert new_dict["ELECTRONS"]["conv_thr"] == 1e-10

    # test that the original one is not modified
    assert old_expect == old_dict


def test_to_valid_key():
    """test to_valid_key"""
    from aiida_sssp_workflow.utils import to_valid_key

    assert to_valid_key("Si_ONCV_PBE-1.2") == "Si_ONCV_PBE_1_2"
