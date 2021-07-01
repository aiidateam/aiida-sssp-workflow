# -*- coding: utf-8 -*-
"""Check that version numbers match.

Check version number in setup.json and aiida_sssp_workflow/__init__.py and make sure
they match.
"""
import os
import json
import sys

this_path = os.path.split(os.path.realpath(__file__))[0]

# Get content of setup.json
setup_name = 'setup.json'  #pylint: disable=invalid-name
setup_path = os.path.join(this_path, os.pardir, setup_name)
with open(setup_path) as f:
    setup_content = json.load(f)

# Get version from python package
sys.path.insert(0, os.path.join(this_path, os.pardir))
import aiida_sssp_workflow  # pylint: disable=wrong-import-position
version = aiida_sssp_workflow.__version__  #pylint: disable=invalid-name

if version != setup_content['version']:
    print('Version number mismatch detected:')
    print(f"Version number in '{setup_name}': {setup_content['version']}")
    print(f"Version number in 'aiida_sssp_workflow/__init__.py': {version}")
    sys.exit(1)

# Overwrite version in setup.json
#setup_content['version'] = version
#with open(setup_path, 'w') as f:
#	json.dump(setup_content, f, indent=4, sort_keys=True)
