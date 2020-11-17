"""
Example for functions for bands distance calculation
"""
from aiida import orm
from aiida.plugins import CalculationFactory

from aiida_sssp_workflow.calculations.calculate_bands_distance import calculate_eta_and_max_diff, \
    retrieve_bands, calculate_bands_distance

# calculate_bands_distance = CalculationFactory('sssp_workflow.calculate_bands_distance')

RY_TO_EV = 13.6056980659

def calc_eta_identity(bandsdata, band_parameters):
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

    outputs = calculate_eta_and_max_diff(bands_a, bands_b, efermi_a, efermi_b, fermi_shift, smearing_v)
    return outputs

def calc_eta_identity_with_shift(bandsdata, band_parameters):
    efermi = orm.Float(band_parameters['fermi_energy'])
    start_band = orm.Int(0)
    num_electrons = orm.Int(8)
    smearing_v = orm.Float(0.0)    # nearly a step function
    smearing_10 = orm.Float(0.02 * RY_TO_EV)

    res = retrieve_bands(bandsdata, start_band, num_electrons, efermi, is_metal=orm.Bool(False))
    bands_a = res['bands']
    efermi_a = res['efermi']
    res = retrieve_bands(bandsdata, start_band, num_electrons, efermi, is_metal=orm.Bool(False))
    bands_b = orm.ArrayData()
    bands_b.set_array('kpoints', res['bands'].get_array('kpoints'))
    bands_b.set_array('bands', res['bands'].get_array('bands') + 0.01)
    efermi_b = res['efermi']
    fermi_shift = orm.Float(0.0)

    outputs = calculate_eta_and_max_diff(bands_a, bands_b, efermi_a, efermi_b, fermi_shift, smearing_v)
    return outputs

if __name__ == '__main__':
    # bandsdata of 8 electrons silicon
    bandsdata_a = orm.load_node('aa9a6d48-bbc2-4104-977c-4264c725d947')
    band_parameters_a = orm.load_node('6946d96b-8656-4046-ab04-d8f11ff817c4')

    # bandsdata of 12 elecnrons silicon
    bandsdata_b = orm.load_node('a9ef4515-59b6-41a5-8667-6608f7adb258')
    band_parameters_b = orm.load_node('5bfeda4c-a87c-4680-9b24-85254dccabdc')

    res = calculate_bands_distance(bandsdata_a, bandsdata_b, band_parameters_a, band_parameters_b, orm.Float(0.02 * RY_TO_EV))
    print(res.get('eta_v'))
    print(res.get('shift_v'))
    print(res.get('max_diff_v'))
    print(res.get('eta_10'))
    print(res.get('shift_10'))
    print(res.get('max_diff_10'))

    # res = calc_eta_identity(bandsdata, band_parameters)
    # print(res.get('eta'))
    # print(res.get('shift'))
    # print(res.get('max_diff'))
    #
    # res = calc_eta_identity_with_shift(bandsdata, band_parameters)
    # print(res.get('eta'))
    # print(res.get('shift')) # should be 0.01
    # print(res.get('max_diff'))  # should be 0.01