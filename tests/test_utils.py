def test_update_dict():
    from aiida_sssp_workflow.utils import update_dict
    import copy

    u = {
        'SYSTEM': {
            'degauss': 0.02,
            'occupations': 'smearing',
            'smearing': 'marzari-vanderbilt',
        },
        'ELECTRONS': {
            'conv_thr': 1e-10,
        },
    }
    u_expect = copy.deepcopy(u)

    d = {
        'SYSTEM': {
            'ecutrho': 1600,
            'ecutwfc': 200,
            'smearing': 'fd',
        },
    }

    r = update_dict(u, d)

    assert r['SYSTEM']['degauss'] == 0.02
    assert r['SYSTEM']['smearing'] == 'fd'
    assert r['SYSTEM']['ecutrho'] == 1600
    assert r['ELECTRONS']['conv_thr'] == 1e-10
    assert u == u_expect


def test_cif_from_element():
    from aiida_sssp_workflow.utils import get_standard_cif_filename_from_element

    fn = get_standard_cif_filename_from_element('Si')
    assert 'Si.cif' in fn

    fn = get_standard_cif_filename_from_element('SiF4')
    assert 'SiF4.cif' in fn

    fn = get_standard_cif_filename_from_element('La')
    assert 'LaN.cif' in fn
