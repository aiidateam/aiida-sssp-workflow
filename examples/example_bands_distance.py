"""
Example for functions for bands distance calculation
"""
from aiida import orm
from aiida.plugins import CalculationFactory

from aiida_sssp_workflow.calculations.calculate_bands_distance import calculate_eta_and_max_diff, \
    retrieve_bands, calculate_bands_distance

# calculate_bands_distance = CalculationFactory('sssp_workflow.calculate_bands_distance')

RY_TO_EV = 13.6056980659

def calc_band_distance(wc_node1, wc_node2, is_metal):
    bandsdata_a = wc_node1.outputs.band_structure
    band_parameters_a = wc_node1.outputs.band_parameters

    bandsdata_b = wc_node2.outputs.band_structure
    band_parameters_b = wc_node2.outputs.band_parameters

    res = calculate_bands_distance(bandsdata_a, bandsdata_b, band_parameters_a, band_parameters_b,
                                   orm.Float(0.02 * RY_TO_EV), orm.Bool(is_metal))
    print(f'etav: {res.get("eta_v")}')
    print(f'shift_v: {res.get("shift_v")}')
    print(f'max_diff_v: {res.get("max_diff_v")}')
    print(f'eta_10: {res.get("eta_10")}')
    print(f'shift_10: {res.get("shift_10")}')
    print(f'max_diff_10: {res.get("max_diff_10")}')


if __name__ == '__main__':
    wc1 = orm.load_node('3d69f296-89f3-4cb3-854d-4e433344a96a')
    wc2 = orm.load_node('15582ab4-1ba0-4398-8fb7-8831661a3a6a')
    calc_band_distance(wc1, wc2, is_metal=True)