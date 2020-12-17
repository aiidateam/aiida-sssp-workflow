"""helper workfunction"""
from aiida.engine import workfunction


@workfunction
def echo_workfunction(x):
    return x
