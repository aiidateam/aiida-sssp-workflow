# -*- coding: utf-8 -*-
"""setup.py"""
import json
import setuptools  # this is the "magic" import
from numpy.distutils.core import setup, Extension

flib = Extension(
    name='sssp.efermi_module',
    sources=['aiida_sssp_workflow/efermi.pyf', 'aiida_sssp_workflow/efermi.f'])

if __name__ == '__main__':
    # Provide static information in setup.json
    # such that it can be discovered automatically
    with open('setup.json', 'r') as info:
        kwargs = json.load(info)
    setup(
        packages=setuptools.find_packages(exclude=['tests*']),
        # this doesn't work when placed in setup.json (something to do with str type)
        package_data={
            '': ['*'],
            'aiida_sssp_workflow':
            ['REF/CIFs/*.cif', 'REF/CIFs_REN/*.cif', 'REF/UPFs/*.UPF'],
        },
        ext_modules=[flib],
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        **kwargs)
