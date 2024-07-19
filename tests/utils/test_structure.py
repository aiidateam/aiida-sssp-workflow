import pytest
import itertools
import numpy as np

from aiida.engine import run_get_node
from aiida import orm

from aiida_sssp_workflow.utils import get_default_configuration, get_standard_structure
from aiida_sssp_workflow.utils.element import ALL_ELEMENTS, UNSUPPORTED_ELEMENTS

ALL_SUPPORTED_ELEMENTS = [e for e in ALL_ELEMENTS if e not in UNSUPPORTED_ELEMENTS]


@pytest.mark.parametrize(
    "element, property, configuration",
    [
        ("Fe", "band", "GS"),
        ("Fe", "convergence", "DC"),
        ("Te", "convergence", "DC"),
        ("Te", "band", "GS"),
        ("La", "band", "LAN"),
        ("La", "convergence", "DC"),
    ],
)
def test_get_default_configuration(element, property, configuration):
    """Test get_default_configuration function."""
    r, _ = run_get_node(get_default_configuration, element=element, property=property)

    assert isinstance(r, orm.Str)
    assert r.value == configuration


@pytest.mark.parametrize(
    "element, configuration",
    [
        ("Fe", "GS"),
        ("Fe", "DC"),
        ("Fe", "BCC"),
        ("Te", "DC"),
        ("Te", "GS"),
        ("La", "XO"),
        ("La", "LAN"),
        ("La", "DC"),
    ],
)
def test_get_standard_structure(element, configuration, data_regression):
    """Data regression test for get_standard_structure function."""
    r, _ = run_get_node(
        get_standard_structure, element=element, configuration=configuration
    )
    ase_r = r.get_ase()

    data_regression.check({k: np.round(v, 3).tolist() for k, v in ase_r.arrays.items()})


@pytest.mark.parametrize(
    "element, target_property",
    [
        (e, p)
        for e, p in itertools.product(ALL_SUPPORTED_ELEMENTS, ["band", "convergence"])
    ],
)
def test_get_standard_structure_for_available_properties(element, target_property):
    """Go through all elements covered configuration from get_default_configuration"""
    # Loop over all elements
    configuration = get_default_configuration(
        element,
        property=target_property,
    )

    structure = get_standard_structure(
        element,
        configuration=configuration,
    )

    assert isinstance(structure, orm.StructureData)


@pytest.mark.parametrize(
    "element, target_property",
    [
        (e, p)
        for e, p in itertools.product(UNSUPPORTED_ELEMENTS, ["band", "convergence"])
    ],
)
def test_get_standard_structure_raise_for_unsupported(element, target_property):
    """Go through all elements covered configuration from get_default_configuration"""
    # Loop over all elements
    configuration = get_default_configuration(
        element,
        property=target_property,
    )

    with pytest.raises(ValueError, match="Unknown configuration N/A"):
        get_standard_structure(
            element,
            configuration=configuration,
        )
