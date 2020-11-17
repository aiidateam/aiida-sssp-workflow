"""
Test case of functions for bands distance calculation
"""
from aiida import orm
from aiida.plugins import CalculationFactory

from aiida_sssp_workflow.calculations.calculate_bands_distance import calculate_eta, retrieve_bands

# calculate_bands_distance = CalculationFactory('sssp_workflow.calculate_bands_distance')

RY_TO_EV = 13.6056980659

def test_calc_eta_identity():
    bandsdata = orm.load_node('aa9a6d48-bbc2-4104-977c-4264c725d947')
    band_parameters = orm.load_node('6946d96b-8656-4046-ab04-d8f11ff817c4')
    efermi = orm.Float(band_parameters['fermi_energy'])
    start_band = orm.Int(0)
    num_electrons = orm.Int(8)
    smearing_v = orm.Float(0.0)    # nearly a step function
    smearing_10 = orm.Float(0.02 * RY_TO_EV)

    res = retrieve_bands(bandsdata, start_band, num_electrons, efermi, is_metal=orm.Bool(False))
    bands_a = res['bands']
    efermi_a = res['efermi']
    res = retrieve_bands(bandsdata, start_band, num_electrons, efermi, is_metal=orm.Bool(False))
    bands_b = res['bands']
    efermi_b = res['efermi']
    fermi_shift = orm.Float(0.0)

    outputs = calculate_eta(bands_a, bands_b, efermi_a, efermi_b, fermi_shift, smearing_v)
    print(outputs['eta'], outputs['shift'])
    # assert outputs['eta_v'] == 0.0
    # assert outputs['eta_10'] == 0.0


# def test_bands_identity():
#     bandsdata = orm.load_node('aa9a6d48-bbc2-4104-977c-4264c725d947')
#     band_parameters = orm.load_node('6946d96b-8656-4046-ab04-d8f11ff817c4')
#     efermi = orm.Float(band_parameters['fermi_energy'])
#     start_band = orm.Int(0)
#     num_electrons = orm.Int(8)
#     smearing_v = orm.Float(0.0)    # nearly a step function
#     smearing_10 = 0.02 * RY_TO_EV
#
#     res = retrieve_bands(bandsdata, start_band, num_electrons, efermi, is_metal=orm.Bool(False))
#     bands_a = res['bands']
#     efermi_a = res['efermi']
#     res = retrieve_bands(bandsdata, start_band, num_electrons, efermi, is_metal=orm.Bool(False))
#     bands_b = res['bands']
#     efermi_b = res['efermi']
#
#     outputs = calculate_bands_distance(bands_a, bands_b, efermi_a, efermi_b, smearing_v, smearing_10)
#     assert outputs['eta_v'] == 0.0
#     assert outputs['eta_10'] == 0.0
#     assert outputs['max_v'] == 0.0
#     assert outputs['max_10'] == 0.0
#     shift_v = outputs['shift_v']
#     shift_10 = outputs['shift_10']
#     print(f'rigid energy shift of eta_v is {shift_v}')
#     print(f'rigid energy shift of eta_10 is {shift_10}')

if __name__ == '__main__':
    test_calc_eta_identity()