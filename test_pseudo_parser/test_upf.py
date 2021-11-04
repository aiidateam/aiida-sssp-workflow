# -*- coding: utf-8 -*-
import pytest

from pseudo_parser.upf_parser import parse_z_valence, parse_element, parse_pseudo_type


@pytest.mark.parametrize('content', (
    'z_valence="1"',
    'z_valence="1.0"',
    'z_valence="1.000"',
    'z_valence="1.00E+00"',
    'z_valence="1."',
    "z_valence='1.0'",
    'z_valence="    1"',
    'z_valence="1    "',
    '1.0     Z valence',
))
def test_parse_z_valence(content):
    """Test the ``parse_z_valence`` method."""
    assert parse_z_valence(content) == 1


@pytest.mark.parametrize('content', (
    'element="O"',
    'element="O "',
))
def test_parse_element(content):
    """Test the ``parse_element`` method"""
    assert parse_element(content) == 'O'


@pytest.mark.parametrize('content', (
    'pseudo_type="US"',
    'pseudo_type="US "',
    '   US                  Ultrasoft pseudopotential',
))
def test_parse_pseudo_type(content):
    """Test the ``parse_pseudo_type`` method"""
    assert parse_pseudo_type(content) == 'US'
