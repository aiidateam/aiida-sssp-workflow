# -*- coding: utf-8 -*-
"""Define the setup for the `aiida-pseudo` plugin."""
try:
    import fastentrypoints  # pylint: disable=unused-import
except ImportError:
    # This should only occur when building the package, i.e. for `python setup.py sdist/bdist_wheel`
    pass


def setup_package():
    """Install the `aiida-pseudo` package."""
    import json
    from setuptools import setup, find_packages

    filename_setup_json = 'setup.json'
    filename_description = 'README.md'

    with open(filename_setup_json, 'r') as handle:
        setup_json = json.load(handle)

    with open(filename_description, 'r') as handle:
        description = handle.read()

    setup(packages=find_packages(),
          package_data={
              '': ['*'],
              'aiida_sssp_workflow': [
                  'REF/CIFs/*.cif', 'REF/CIFs_REN/*.cif', 'REF/UPFs/*.UPF',
                  'sssp_protocol.yml'
              ],
          },
          long_description=description,
          long_description_content_type='text/markdown',
          **setup_json)


if __name__ == '__main__':
    setup_package()
