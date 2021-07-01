# -*- coding: utf-8 -*-
"""helper workfunction"""
from aiida.engine import calcfunction


@calcfunction
def echo_calcfunction(x, y):
    return x + y
