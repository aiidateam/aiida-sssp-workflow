import pytest
from aiida.engine import run_get_node
from aiida import orm

from aiida_sssp_workflow.utils import get_default_configuration, get_standard_structure


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

    data_regression.check(r.get_cif().get_content())
